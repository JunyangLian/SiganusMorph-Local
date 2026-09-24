from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results" / "v2_user_outputs"
SCREENSHOT_DIR = RESULTS_DIR / "visual_smoke"
JSON_REPORT = RESULTS_DIR / "v2_frontend_visual_workflow_check.json"
CSV_REPORT = RESULTS_DIR / "v2_frontend_visual_workflow_check.csv"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph_v2.backend.live_api_flow_check import _png_bytes, _request_json, _upload_png  # noqa: E402
from siganusmorph_v2.backend.live_health_check import _find_free_port, _sanitize_runtime_text  # noqa: E402


def _add_check(checks: list[dict[str, Any]], name: str, passed: bool, evidence: str = "", notes: str = "") -> None:
    checks.append(
        {
            "check_name": name,
            "passed": bool(passed),
            "status": "pass" if passed else "fail",
            "evidence": evidence,
            "notes": _sanitize_runtime_text(str(notes))[:1200],
        }
    )


def _get(url: str, timeout: float = 5.0) -> tuple[int | None, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001 - audit reports the failure safely
        return None, str(exc)


def _wait_for_http(url: str, proc: subprocess.Popen[str], timeout_seconds: float = 30.0) -> tuple[bool, str]:
    deadline = time.monotonic() + timeout_seconds
    last = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False, f"process exited early with returncode={proc.returncode}"
        status, body = _get(url, timeout=2.0)
        if status == 200:
            return True, body
        last = body
        time.sleep(0.25)
    return False, last


def _start_backend(port: int) -> subprocess.Popen[str]:
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
    return subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _start_frontend(port: int) -> subprocess.Popen[str]:
    vite = FRONTEND_ROOT / "node_modules" / ".bin" / ("vite.cmd" if sys.platform.startswith("win") else "vite")
    command = [str(vite), "--host", "127.0.0.1", "--port", str(port)]
    return subprocess.Popen(
        command,
        cwd=FRONTEND_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _safe_terminate(proc: subprocess.Popen[str] | None) -> tuple[str, str]:
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
        try:
            proc.kill()
            return proc.communicate(timeout=3)
        except Exception as exc:  # noqa: BLE001
            return "", f"Process termination output collection failed: {exc}"


def _browser_executable() -> str | None:
    candidates = [
        Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
        Path("C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"),
        Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
        Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _create_measured_session(backend_url: str) -> str:
    upload_status, upload_json, upload_text = _upload_png(backend_url, _png_bytes())
    session_id = upload_json.get("session_id", "")
    if upload_status != 200 or not session_id:
        raise RuntimeError(f"Upload failed: {upload_text[:500]}")
    corners = [[24, 24], [335, 24], [335, 215], [24, 215]]
    cal_status, cal_json, cal_text = _request_json(
        "POST",
        f"{backend_url}/api/calibrate",
        {"session_id": session_id, "manual_corners": corners},
    )
    if cal_status != 200 or cal_json.get("status") != "success":
        raise RuntimeError(f"Manual calibration failed: {cal_text[:500]}")
    measure_status, measure_json, measure_text = _request_json("POST", f"{backend_url}/api/measure", {"session_id": session_id})
    if measure_status != 200 or measure_json.get("coordinate_space") != "warped_image":
        raise RuntimeError(f"Measurement failed: {measure_text[:500]}")
    return session_id


def _screenshot(page: Any, name: str) -> str:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SCREENSHOT_DIR / name
    page.screenshot(path=path, full_page=True)
    return path.relative_to(PROJECT_ROOT).as_posix()


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []
    backend_port = 8000
    frontend_port = _find_free_port(start=5190)
    backend_url = f"http://127.0.0.1:{backend_port}"
    frontend_url = f"http://127.0.0.1:{frontend_port}"
    backend_proc: subprocess.Popen[str] | None = None
    frontend_proc: subprocess.Popen[str] | None = None
    backend_stdout = backend_stderr = frontend_stdout = frontend_stderr = ""
    session_id = ""
    screenshots: dict[str, str] = {}

    try:
        backend_status, backend_notes = _get(f"{backend_url}/api/health", timeout=2.0)
        if backend_status == 200:
            backend_ready = True
        else:
            backend_proc = _start_backend(backend_port)
            backend_ready, backend_notes = _wait_for_http(f"{backend_url}/api/health", backend_proc)
        _add_check(checks, "backend_available_for_visual_workflow", backend_ready, f"{backend_url}/api/health", backend_notes[:500])
        if not backend_ready:
            raise RuntimeError("Backend did not become healthy.")

        session_id = _create_measured_session(backend_url)
        _add_check(checks, "measured_session_prepared_for_visual_restore", bool(session_id), "backend manual session", session_id)

        frontend_proc = _start_frontend(frontend_port)
        frontend_ready, frontend_notes = _wait_for_http(frontend_url, frontend_proc)
        _add_check(checks, "vite_frontend_available_for_browser", frontend_ready, frontend_url, frontend_notes[:500])
        if not frontend_ready:
            raise RuntimeError("Frontend did not become available.")

        executable_path = _browser_executable()
        _add_check(checks, "browser_executable_available", bool(executable_path), executable_path or "", "")
        if not executable_path:
            raise RuntimeError("No Chrome/Edge executable found for browser-level visual workflow check.")

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, executable_path=executable_path)
            page = browser.new_page(viewport={"width": 1440, "height": 960})
            try:
                page.goto(frontend_url, wait_until="networkidle", timeout=30000)
                home_visible = page.get_by_text("SiganusMorph V2.0").count() > 0 and page.get_by_text("首页").count() > 0
                _add_check(checks, "homepage_visible_in_browser", home_visible, frontend_url, "")
                screenshots["homepage"] = _screenshot(page, "v2_browser_homepage.png")

                page.get_by_role("button", name="单鱼测量", exact=True).click(timeout=10000)
                page.get_by_role("heading", name="单鱼测量").wait_for(timeout=10000)
                single_open = page.get_by_text("上传图片").count() > 0 and page.get_by_text("样本编号 specimen_id").count() > 0
                _add_check(checks, "single_fish_page_opened_in_browser", single_open, "单鱼测量 nav", "")

                page.get_by_placeholder("粘贴 session_id").fill(session_id)
                page.get_by_role("button", name="恢复会话").click(timeout=10000)
                page.get_by_role("heading", name="Measurement Canvas").wait_for(timeout=30000)
                canvas_visible = page.locator("svg.measurement-svg").count() > 0
                _add_check(checks, "measurement_canvas_visible_after_session_restore", canvas_visible, "svg.measurement-svg", "")

                image_href = page.locator("svg.measurement-svg image").first.get_attribute("href")
                warped_image_visible = bool(image_href and "warped.png" in image_href and session_id in image_href)
                _add_check(checks, "warped_image_rendered_in_canvas", warped_image_visible, image_href or "", "")

                circle_count = page.locator("svg.measurement-svg circle").count()
                _add_check(checks, "formal_points_rendered_in_canvas", circle_count >= 10, "svg.measurement-svg circle", f"circle_count={circle_count}")

                first_circle = page.locator("svg.measurement-svg circle").first
                first_circle.scroll_into_view_if_needed(timeout=10000)
                box = first_circle.bounding_box(timeout=10000)
                if not box:
                    raise RuntimeError("Could not locate a draggable point bounding box.")
                x = box["x"] + box["width"] / 2
                y = box["y"] + box["height"] / 2
                page.mouse.move(x + 1, y + 1)
                page.wait_for_timeout(250)
                hover_label_visible = page.locator("svg.measurement-svg text").count() > 0
                _add_check(checks, "hover_label_visible_on_point", hover_label_visible, "svg.measurement-svg text", "")

                page.mouse.down()
                page.mouse.move(x + 24, y + 14, steps=4)
                page.wait_for_timeout(250)
                crosshair_visible = page.locator("g.crosshair line").count() >= 2
                _add_check(checks, "crosshair_visible_during_drag", crosshair_visible, "g.crosshair line", "")
                page.mouse.up()

                local_preview_visible = page.locator(".metric").count() >= 3
                _add_check(checks, "local_measurement_preview_visible", local_preview_visible, ".metric", "")

                with page.expect_response(lambda response: "/api/update-points" in response.url and response.status == 200, timeout=30000) as update_info:
                    page.get_by_role("button", name="Apply changes").click(timeout=10000)
                update_json = update_info.value.json()
                _add_check(
                    checks,
                    "apply_changes_calls_update_points",
                    update_json.get("coordinate_space") == "warped_image",
                    "/api/update-points",
                    json.dumps(update_json, ensure_ascii=False)[:500],
                )

                with page.expect_response(lambda response: "/api/export" in response.url and response.status == 200, timeout=30000) as export_info:
                    page.get_by_role("button", name="保存并导出").click(timeout=10000)
                export_json = export_info.value.json()
                exported_files = export_json.get("files") or {}
                export_paths_ok = bool(exported_files) and all(str(path).startswith("results/v2_user_outputs/") for path in exported_files.values())
                _add_check(
                    checks,
                    "export_from_browser_writes_v2_outputs",
                    export_paths_ok,
                    "/api/export",
                    json.dumps(export_json, ensure_ascii=False)[:500],
                )
                screenshots["single_fish_after_drag_export"] = _screenshot(page, "v2_browser_single_fish_after_drag_export.png")

                page.get_by_role("button", name="结果导出", exact=True).click(timeout=10000)
                page.get_by_role("heading", name="结果导出").wait_for(timeout=10000)
                saved_to = str(export_json.get("saved_to") or "")
                try:
                    page.get_by_text(saved_to).first.wait_for(timeout=8000)
                except PlaywrightTimeoutError:
                    pass
                export_page_visible = page.get_by_role("heading", name="结果导出").count() > 0 and (
                    page.get_by_text(saved_to).count() > 0 or page.get_by_text("results/v2_user_outputs/exports").count() > 0
                )
                _add_check(checks, "export_page_lists_browser_export", export_page_visible, "结果导出", saved_to)
            finally:
                browser.close()
    except PlaywrightTimeoutError as exc:
        _add_check(checks, "browser_visual_workflow_completed", False, "", f"Playwright timeout: {exc}")
    except Exception as exc:  # noqa: BLE001
        _add_check(checks, "browser_visual_workflow_completed", False, "", str(exc))
    finally:
        if frontend_proc is not None:
            frontend_stdout, frontend_stderr = _safe_terminate(frontend_proc)
        if backend_proc is not None:
            backend_stdout, backend_stderr = _safe_terminate(backend_proc)

    passed = sum(1 for check in checks if check["passed"])
    failed = len(checks) - passed
    report = {
        "version": "2.0.0",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "backend_url": backend_url,
        "frontend_url": frontend_url,
        "session_id": session_id,
        "screenshots": screenshots,
        "total_checks": len(checks),
        "passed_checks": passed,
        "failed_checks": failed,
        "visual_workflow_ready": failed == 0,
        "backend_stdout_tail": _sanitize_runtime_text(backend_stdout)[-1200:],
        "backend_stderr_tail": _sanitize_runtime_text(backend_stderr)[-1200:],
        "frontend_stdout_tail": _sanitize_runtime_text(frontend_stdout)[-1200:],
        "frontend_stderr_tail": _sanitize_runtime_text(frontend_stderr)[-1200:],
        "checks": checks,
    }
    JSON_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with CSV_REPORT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["check_name", "passed", "status", "evidence", "notes"])
        writer.writeheader()
        writer.writerows(checks)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Visual workflow report written to: {JSON_REPORT.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Visual workflow CSV written to: {CSV_REPORT.relative_to(PROJECT_ROOT).as_posix()}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
