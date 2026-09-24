from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph_v2.backend.live_health_check import (  # noqa: E402
    _find_free_port,
    _project_relative,
    _sanitize_runtime_text,
    _terminate,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "v2_user_outputs"
JSON_REPORT = RESULTS_DIR / "v2_live_api_flow_check.json"
CSV_REPORT = RESULTS_DIR / "v2_live_api_flow_check.csv"


def _png_bytes() -> bytes:
    image = Image.new("RGB", (360, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((70, 80, 280, 165), fill=(205, 212, 206), outline=(50, 70, 80), width=2)
    draw.polygon([(280, 120), (335, 75), (325, 120), (335, 168)], fill=(195, 204, 198), outline=(50, 70, 80))
    draw.ellipse((105, 105, 116, 116), fill=(20, 20, 20))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


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
            "notes": notes,
        }
    )


def _request_json(method: str, url: str, payload: dict[str, Any] | None = None) -> tuple[int | None, dict[str, Any], str]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body), body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {}
        return exc.code, parsed, body
    except Exception as exc:  # noqa: BLE001 - report user-safe reason in audit output
        return None, {}, str(exc)


def _upload_png(base_url: str, image_bytes: bytes) -> tuple[int | None, dict[str, Any], str]:
    boundary = f"----siganusmorph-v2-{uuid.uuid4().hex}"
    fields = {
        "specimen_id": "v2_live_http",
        "weight_g": "155.33",
    }
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(f"--{boundary}\r\n".encode("utf-8"))
        parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        parts.append(f"{value}\r\n".encode("utf-8"))
    parts.append(f"--{boundary}\r\n".encode("utf-8"))
    parts.append(
        b'Content-Disposition: form-data; name="file"; filename="live_http_fish.png"\r\n'
        b"Content-Type: image/png\r\n\r\n"
    )
    parts.append(image_bytes)
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(parts)
    request = urllib.request.Request(
        f"{base_url}/api/upload",
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            text = response.read().decode("utf-8")
            return response.status, json.loads(text), text
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = {}
        return exc.code, parsed, text
    except Exception as exc:  # noqa: BLE001
        return None, {}, str(exc)


def _wait_for_health(base_url: str, proc: subprocess.Popen[str]) -> tuple[bool, str]:
    deadline = time.monotonic() + 20.0
    last_error = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False, f"uvicorn exited early, returncode={proc.returncode}"
        status, payload, text = _request_json("GET", f"{base_url}/api/health")
        if status == 200 and payload.get("status") == "ok":
            return True, text
        last_error = text
        time.sleep(0.25)
    return False, last_error


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []
    port = _find_free_port(start=8775)
    base_url = f"http://127.0.0.1:{port}"
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
    command_display = ["python", "-m", "uvicorn", "siganusmorph_v2.backend.app:app", "--host", "127.0.0.1", "--port", str(port)]
    stdout = ""
    stderr = ""
    session_id = ""
    exported_files: dict[str, str] = {}

    proc = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        health_ok, health_notes = _wait_for_health(base_url, proc)
        _add_check(checks, "server_health_ready", health_ok, 200 if health_ok else None, f"{base_url}/api/health", health_notes)
        if not health_ok:
            raise RuntimeError("Live server did not become healthy.")

        upload_status, upload_json, upload_text = _upload_png(base_url, _png_bytes())
        session_id = upload_json.get("session_id", "")
        _add_check(
            checks,
            "upload_over_http",
            upload_status == 200 and bool(session_id) and upload_json.get("status") == "uploaded",
            upload_status,
            "POST /api/upload",
            upload_text[:500],
        )

        corners = [[24, 24], [335, 24], [335, 215], [24, 215]]
        cal_status, cal_json, cal_text = _request_json("POST", f"{base_url}/api/calibrate", {"session_id": session_id, "manual_corners": corners})
        _add_check(
            checks,
            "manual_calibration_over_http",
            cal_status == 200
            and cal_json.get("status") == "success"
            and cal_json.get("mm_per_pixel_source") == "manual_board_corners"
            and bool(cal_json.get("warped_image_url")),
            cal_status,
            "POST /api/calibrate",
            cal_text[:500],
        )

        measure_status, measure_json, measure_text = _request_json("POST", f"{base_url}/api/measure", {"session_id": session_id})
        formal_points = measure_json.get("formal_points") or {}
        measurements = measure_json.get("measurements") or {}
        _add_check(
            checks,
            "measure_over_http",
            measure_status == 200
            and measure_json.get("coordinate_space") == "warped_image"
            and len(formal_points) >= 10
            and bool(measurements),
            measure_status,
            "POST /api/measure",
            measure_text[:500],
        )

        first_point_name = next((name for name, value in formal_points.items() if isinstance(value, list) and len(value) >= 2), "")
        if first_point_name:
            updated_points = json.loads(json.dumps(formal_points))
            updated_points[first_point_name][0] = float(updated_points[first_point_name][0]) + 1.0
        else:
            updated_points = formal_points
        update_status, update_json, update_text = _request_json(
            "POST",
            f"{base_url}/api/update-points",
            {
                "session_id": session_id,
                "formal_points": updated_points,
                "measurement_overrides": {"live_api_flow_check": True, "modified_point": first_point_name},
            },
        )
        _add_check(
            checks,
            "update_points_over_http",
            update_status == 200 and update_json.get("unsaved_changes") is True and update_json.get("coordinate_space") == "warped_image",
            update_status,
            "POST /api/update-points",
            update_text[:500],
        )

        export_status, export_json, export_text = _request_json(
            "POST",
            f"{base_url}/api/export",
            {"session_id": session_id, "formats": ["csv", "xlsx", "json", "preview_png"]},
        )
        exported_files = export_json.get("files") or {}
        expected_formats = {"csv", "xlsx", "json", "preview_png"}
        files_exist = all((PROJECT_ROOT / path).exists() for path in exported_files.values())
        _add_check(
            checks,
            "export_over_http",
            export_status == 200 and expected_formats.issubset(exported_files.keys()) and files_exist,
            export_status,
            "POST /api/export",
            export_text[:500],
        )

        session_status, session_json, session_text = _request_json("GET", f"{base_url}/api/session/{session_id}")
        _add_check(
            checks,
            "session_restore_over_http",
            session_status == 200
            and session_json.get("session_id") == session_id
            and session_json.get("coordinate_space") == "warped_image"
            and "original_path" not in json.dumps(session_json, ensure_ascii=False),
            session_status,
            "GET /api/session/{session_id}",
            session_text[:500],
        )

        exports_status, exports_json, exports_text = _request_json("GET", f"{base_url}/api/exports")
        export_ids = [entry.get("session_id") for entry in exports_json.get("exports", [])]
        _add_check(
            checks,
            "exports_listing_over_http",
            exports_status == 200 and session_id in export_ids,
            exports_status,
            "GET /api/exports",
            exports_text[:500],
        )
    except Exception as exc:  # noqa: BLE001
        _add_check(checks, "live_api_flow_completed", False, None, "", str(exc))
    finally:
        stdout, stderr = _terminate(proc)

    passed = sum(1 for check in checks if check["passed"])
    failed = len(checks) - passed
    report = {
        "version": "2.0.0",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "server_url": base_url,
        "command": command_display,
        "session_id": session_id,
        "exported_files": exported_files,
        "total_checks": len(checks),
        "passed_checks": passed,
        "failed_checks": failed,
        "stdout_tail": _sanitize_runtime_text(stdout)[-2000:],
        "stderr_tail": _sanitize_runtime_text(stderr)[-2000:],
        "checks": checks,
    }

    JSON_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with CSV_REPORT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["check_name", "passed", "status", "status_code", "evidence", "notes"])
        writer.writeheader()
        writer.writerows(checks)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Live API flow report written to: {_project_relative(JSON_REPORT)}")
    print(f"Live API flow CSV written to: {_project_relative(CSV_REPORT)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
