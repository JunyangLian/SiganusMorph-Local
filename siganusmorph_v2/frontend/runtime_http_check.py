from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results" / "v2_user_outputs"
JSON_REPORT = RESULTS_DIR / "v2_frontend_runtime_http_check.json"
CSV_REPORT = RESULTS_DIR / "v2_frontend_runtime_http_check.csv"
FRONTEND_URL = "http://127.0.0.1:5173"
BACKEND_URL = "http://127.0.0.1:8000"


def _rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _sanitize(text: str) -> str:
    text = text.replace(str(PROJECT_ROOT), "<project_root>")
    text = text.replace(str(PROJECT_ROOT).replace("\\", "/"), "<project_root>")
    return text


def _get(url: str, timeout: float = 5.0) -> tuple[int | None, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    status_code: int | None = None,
    evidence: str = "",
    notes: str = "",
) -> None:
    checks.append(
        {
            "check_name": name,
            "passed": bool(passed),
            "status": "pass" if passed else "fail",
            "status_code": status_code,
            "evidence": evidence,
            "notes": _sanitize(notes[:1200]),
        }
    )


def _wait_for(url: str, timeout_seconds: float = 20.0) -> tuple[bool, int | None, str]:
    deadline = time.monotonic() + timeout_seconds
    last_status: int | None = None
    last_body = ""
    while time.monotonic() < deadline:
        last_status, last_body = _get(url, timeout=2.0)
        if last_status == 200:
            return True, last_status, last_body
        time.sleep(0.25)
    return False, last_status, last_body


def _start_backend_if_needed() -> tuple[subprocess.Popen[str] | None, str]:
    status, body = _get(f"{BACKEND_URL}/api/health", timeout=2.0)
    if status == 200 and '"status":"ok"' in body.replace(" ", ""):
        return None, "existing_backend"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "siganusmorph_v2.backend.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--log-level",
            "warning",
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc, "started_backend"


def _start_frontend_if_needed() -> tuple[subprocess.Popen[str] | None, str]:
    status, body = _get(FRONTEND_URL, timeout=2.0)
    if status == 200 and "src/main.tsx" in body:
        return None, "existing_frontend"
    npm = "npm.cmd" if sys.platform.startswith("win") else "npm"
    proc = subprocess.Popen(
        [npm, "run", "dev"],
        cwd=FRONTEND_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc, "started_frontend"


def _stop(proc: subprocess.Popen[str] | None) -> tuple[str, str]:
    if proc is None:
        return "", ""
    if proc.poll() is None:
        if sys.platform.startswith("win"):
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, text=True)
        else:
            proc.terminate()
        try:
            return proc.communicate(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                return proc.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                return "", "Process tree termination timed out."
    try:
        return proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        return "", "Process output collection timed out."


def _kill_windows_listeners(port: int) -> str:
    if not sys.platform.startswith("win"):
        return ""
    try:
        proc = subprocess.run(["netstat", "-ano", "-p", "tcp"], capture_output=True, text=True, timeout=8)
    except Exception as exc:  # noqa: BLE001
        return f"netstat failed: {exc}"
    killed: list[str] = []
    for line in proc.stdout.splitlines():
        if f":{port} " not in line or "LISTENING" not in line:
            continue
        parts = line.split()
        if not parts:
            continue
        pid = parts[-1]
        if not pid.isdigit():
            continue
        subprocess.run(["taskkill", "/PID", pid, "/T", "/F"], capture_output=True, text=True)
        killed.append(pid)
    return ",".join(sorted(set(killed)))


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []
    backend_proc: subprocess.Popen[str] | None = None
    frontend_proc: subprocess.Popen[str] | None = None
    backend_source = ""
    frontend_source = ""
    backend_stdout = backend_stderr = frontend_stdout = frontend_stderr = ""

    try:
        backend_proc, backend_source = _start_backend_if_needed()
        backend_ready, backend_status, backend_body = _wait_for(f"{BACKEND_URL}/api/health")
        _add_check(
            checks,
            "backend_health_available_for_frontend",
            backend_ready and '"version":"2.0.0"' in backend_body.replace(" ", ""),
            backend_status,
            f"{BACKEND_URL}/api/health",
            backend_body,
        )

        frontend_proc, frontend_source = _start_frontend_if_needed()
        frontend_ready, root_status, root_html = _wait_for(FRONTEND_URL)
        _add_check(
            checks,
            "vite_dev_server_serves_root",
            frontend_ready and '<div id="root">' in root_html and "/src/main.tsx" in root_html,
            root_status,
            FRONTEND_URL,
            root_html,
        )

        main_status, main_module = _get(f"{FRONTEND_URL}/src/main.tsx")
        _add_check(
            checks,
            "vite_transforms_main_module",
            main_status == 200 and "createRoot" in main_module and "App" in main_module,
            main_status,
            "/src/main.tsx",
            main_module,
        )

        app_status, app_module = _get(f"{FRONTEND_URL}/src/App.tsx")
        _add_check(
            checks,
            "vite_transforms_app_module",
            app_status == 200 and all(label in app_module for label in ["首页", "单鱼测量", "批量测量", "结果导出", "设置"]),
            app_status,
            "/src/App.tsx",
            app_module,
        )

        single_status, single_module = _get(f"{FRONTEND_URL}/src/pages/SingleFishPage.tsx")
        _add_check(
            checks,
            "vite_transforms_single_fish_page",
            single_status == 200 and all(label in single_module for label in ["上传图片", "样本编号", "样品重量", "开始测量"]),
            single_status,
            "/src/pages/SingleFishPage.tsx",
            single_module,
        )

        token_status, token_css = _get(f"{FRONTEND_URL}/src/styles/tokens.css")
        _add_check(
            checks,
            "apple_design_tokens_served",
            token_status == 200
            and all(token in token_css.lower() for token in ["#f5f5f7", "#ffffff", "#1d1d1f", "#707070", "#0071e3", "#0066cc", "#e8e8ed"]),
            token_status,
            "/src/styles/tokens.css",
            token_css,
        )

        user_visible_payload = "\n".join([root_html])
        forbidden = [
            r"\b[A-Z]:[\\/]",
            r"Traceback",
            r'File "',
            r"corrected_keypoints",
            r"final_analysis_dataset",
        ]
        hits = [pattern for pattern in forbidden if re.search(pattern, user_visible_payload, flags=re.IGNORECASE)]
        _add_check(
            checks,
            "frontend_user_visible_runtime_hides_internal_paths_and_tracebacks",
            not hits,
            None,
            "frontend root HTML",
            ", ".join(hits),
        )
    finally:
        frontend_stdout, frontend_stderr = _stop(frontend_proc)
        backend_stdout, backend_stderr = _stop(backend_proc)
        if frontend_source == "started_frontend":
            killed = _kill_windows_listeners(5173)
            if killed:
                frontend_stderr = f"{frontend_stderr}\nKilled lingering frontend listener PID(s): {killed}".strip()
        if backend_source == "started_backend":
            killed = _kill_windows_listeners(8000)
            if killed:
                backend_stderr = f"{backend_stderr}\nKilled lingering backend listener PID(s): {killed}".strip()

    passed = sum(1 for check in checks if check["passed"])
    failed = len(checks) - passed
    report = {
        "version": "2.0.0",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "backend_source": backend_source,
        "frontend_source": frontend_source,
        "frontend_url": FRONTEND_URL,
        "backend_url": BACKEND_URL,
        "total_checks": len(checks),
        "passed_checks": passed,
        "failed_checks": failed,
        "backend_stdout_tail": _sanitize(backend_stdout)[-1200:],
        "backend_stderr_tail": _sanitize(backend_stderr)[-1200:],
        "frontend_stdout_tail": _sanitize(frontend_stdout)[-1200:],
        "frontend_stderr_tail": _sanitize(frontend_stderr)[-1200:],
        "checks": checks,
    }
    JSON_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with CSV_REPORT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["check_name", "passed", "status", "status_code", "evidence", "notes"])
        writer.writeheader()
        writer.writerows(checks)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Frontend runtime HTTP report written to: {_rel(JSON_REPORT)}")
    print(f"Frontend runtime HTTP CSV written to: {_rel(CSV_REPORT)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
