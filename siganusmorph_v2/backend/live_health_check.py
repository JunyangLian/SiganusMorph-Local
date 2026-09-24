from __future__ import annotations

import csv
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "v2_user_outputs"
JSON_REPORT = RESULTS_DIR / "v2_live_backend_health_check.json"
CSV_REPORT = RESULTS_DIR / "v2_live_backend_health_check.csv"
HEALTH_APP_NAME = "SiganusMorph Local V2.0"
HEALTH_VERSION = "2.0.0"


def _project_relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _sanitize_runtime_text(text: str) -> str:
    if not text:
        return ""
    sanitized = text.replace(str(PROJECT_ROOT), "<project_root>")
    sanitized = sanitized.replace(str(PROJECT_ROOT).replace("\\", "/"), "<project_root>")
    return sanitized


def _find_free_port(start: int = 8000, max_tries: int = 50) -> int:
    for port in range(start, start + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No free localhost port found in range {start}-{start + max_tries - 1}")


def _add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    evidence: str = "",
    notes: str = "",
) -> None:
    checks.append(
        {
            "check_name": name,
            "passed": bool(passed),
            "status": "pass" if passed else "fail",
            "evidence": evidence,
            "notes": notes,
        }
    )


def _request_json(url: str) -> tuple[int | None, dict[str, Any] | None, str]:
    try:
        with urllib.request.urlopen(url, timeout=2.0) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body), body
    except urllib.error.HTTPError as exc:
        return exc.code, None, str(exc)
    except Exception as exc:  # noqa: BLE001 - report startup failures without leaking stack traces
        return None, None, str(exc)


def _terminate(proc: subprocess.Popen[str]) -> tuple[str, str]:
    if proc.poll() is None:
        proc.terminate()
        try:
            return proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            return proc.communicate(timeout=5)
    return proc.communicate(timeout=5)


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []
    port = _find_free_port()
    health_url = f"http://127.0.0.1:{port}/api/health"
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "siganusmorph_v2.backend.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    command_display = [
        "python",
        "-m",
        "uvicorn",
        "siganusmorph_v2.backend.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    proc: subprocess.Popen[str] | None = None
    health_status: int | None = None
    health_payload: dict[str, Any] | None = None
    last_error = ""
    stdout = ""
    stderr = ""

    try:
        proc = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                last_error = f"uvicorn exited before health check, returncode={proc.returncode}"
                break
            health_status, health_payload, last_error = _request_json(health_url)
            if health_status == 200 and health_payload:
                break
            time.sleep(0.25)

        _add_check(
            checks,
            "uvicorn_process_started",
            proc.poll() is None or health_status == 200,
            "python -m uvicorn siganusmorph_v2.backend.app:app",
            last_error if health_status != 200 else "",
        )
        _add_check(
            checks,
            "health_endpoint_http_200",
            health_status == 200,
            health_url,
            last_error,
        )
        _add_check(
            checks,
            "health_payload_status_ok",
            bool(health_payload) and health_payload.get("status") == "ok",
            json.dumps(health_payload, ensure_ascii=False) if health_payload else "",
            "",
        )
        _add_check(
            checks,
            "health_payload_version",
            bool(health_payload) and health_payload.get("version") == HEALTH_VERSION,
            HEALTH_VERSION,
            json.dumps(health_payload, ensure_ascii=False) if health_payload else "",
        )
        _add_check(
            checks,
            "health_payload_app_name",
            bool(health_payload) and health_payload.get("app") == HEALTH_APP_NAME,
            HEALTH_APP_NAME,
            json.dumps(health_payload, ensure_ascii=False) if health_payload else "",
        )
    finally:
        if proc is not None:
            stdout, stderr = _terminate(proc)

    passed = sum(1 for check in checks if check["passed"])
    failed = len(checks) - passed
    report = {
        "version": HEALTH_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "server_url": f"http://127.0.0.1:{port}",
        "health_url": health_url,
        "command": command_display,
        "total_checks": len(checks),
        "passed_checks": passed,
        "failed_checks": failed,
        "stdout_tail": _sanitize_runtime_text(stdout)[-2000:],
        "stderr_tail": _sanitize_runtime_text(stderr)[-2000:],
        "checks": checks,
    }

    JSON_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with CSV_REPORT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["check_name", "passed", "status", "evidence", "notes"])
        writer.writeheader()
        writer.writerows(checks)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Live backend health report written to: {_project_relative(JSON_REPORT)}")
    print(f"Live backend health CSV written to: {_project_relative(CSV_REPORT)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
