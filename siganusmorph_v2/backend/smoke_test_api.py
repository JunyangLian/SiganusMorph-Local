from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph_v2.backend.app import app  # noqa: E402
from siganusmorph_v2.backend.services.session_store import load_session, session_dir  # noqa: E402

REPORT_PATH = PROJECT_ROOT / "results" / "v2_user_outputs" / "v2_backend_smoke_test_report.json"


def _png_bytes(*, fish_like: bool = False) -> bytes:
    image = Image.new("RGB", (360, 240), "white")
    draw = ImageDraw.Draw(image)
    if fish_like:
        draw.ellipse((70, 80, 280, 165), fill=(205, 212, 206), outline=(50, 70, 80), width=2)
        draw.polygon([(280, 120), (335, 75), (325, 120), (335, 168)], fill=(195, 204, 198), outline=(50, 70, 80))
        draw.ellipse((105, 105, 116, 116), fill=(20, 20, 20))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _upload(client: TestClient, filename: str, content: bytes) -> dict[str, Any]:
    response = client.post(
        "/api/upload",
        data={"specimen_id": "v2_smoke", "weight_g": "155.33"},
        files={"file": (filename, content, "image/png")},
    )
    return {
        "status_code": response.status_code,
        "json": response.json(),
    }


def _check_passed(check: dict[str, Any]) -> bool:
    if "passed" in check:
        return bool(check["passed"])
    if "blocked" in check:
        return bool(check["blocked"])
    if "status_code" in check:
        return int(check["status_code"]) == 200
    return False


def main() -> None:
    client = TestClient(app)
    report: dict[str, Any] = {"version": "2.0.0", "checks": []}

    health = client.get("/api/health")
    report["checks"].append({"name": "health", "status_code": health.status_code, "json": health.json()})

    settings = client.get("/api/settings")
    report["checks"].append(
        {
            "name": "settings",
            "status_code": settings.status_code,
            "json": settings.json(),
            "passed": settings.status_code == 200 and settings.json().get("version") == "2.0.0",
        }
    )

    invalid_session = client.get("/api/session/not-a-uuid")
    report["checks"].append(
        {
            "name": "invalid_session_id_rejected",
            "status_code": invalid_session.status_code,
            "json": invalid_session.json(),
            "passed": invalid_session.status_code == 400,
        }
    )

    unsafe_upload = _upload(client, "..\\..\\unsafe_name.png", _png_bytes())
    unsafe_session_id = unsafe_upload["json"].get("session_id")
    unsafe_session = load_session(unsafe_session_id) if unsafe_session_id else {}
    raw_image = unsafe_session.get("raw_image", {})
    raw_original = Path(raw_image.get("original_path", "")).resolve()
    unsafe_session_root = session_dir(unsafe_session_id).resolve() if unsafe_session_id else Path("")
    report["checks"].append(
        {
            "name": "unsafe_upload_filename_sanitized",
            "upload_status_code": unsafe_upload["status_code"],
            "stored_filename": raw_image.get("filename"),
            "stored_path_inside_session": unsafe_session_root in [raw_original, *raw_original.parents],
            "passed": unsafe_upload["status_code"] == 200
            and raw_image.get("filename") == "unsafe_name.png"
            and unsafe_session_root in [raw_original, *raw_original.parents],
        }
    )

    blocked_upload = _upload(client, "blank_gate.png", _png_bytes())
    blocked_session = blocked_upload["json"].get("session_id")
    blocked_cal = client.post("/api/calibrate", json={"session_id": blocked_session})
    blocked_measure = client.post("/api/measure", json={"session_id": blocked_session})
    report["checks"].append(
        {
            "name": "calibration_failure_blocks_measurement",
            "upload_status_code": blocked_upload["status_code"],
            "calibrate_status_code": blocked_cal.status_code,
            "calibrate_json": blocked_cal.json(),
            "measure_status_code": blocked_measure.status_code,
            "blocked": blocked_measure.status_code == 400,
        }
    )

    success_upload = _upload(client, "manual_calibration.png", _png_bytes(fish_like=True))
    success_session = success_upload["json"].get("session_id")
    manual_corners = [[24, 24], [335, 24], [335, 215], [24, 215]]
    success_cal = client.post("/api/calibrate", json={"session_id": success_session, "manual_corners": manual_corners})
    success_cal_json = success_cal.json()
    report["checks"].append(
        {
            "name": "manual_corner_calibration",
            "upload_status_code": success_upload["status_code"],
            "calibrate_status_code": success_cal.status_code,
            "calibrate_json": success_cal_json,
            "passed": success_cal.status_code == 200
            and success_cal_json.get("status") == "success"
            and success_cal_json.get("mm_per_pixel_source") == "manual_board_corners",
        }
    )

    premature_export = client.post(
        "/api/export",
        json={"session_id": success_session, "formats": ["csv"]},
    )
    report["checks"].append(
        {
            "name": "export_blocked_before_measurement",
            "status_code": premature_export.status_code,
            "json": premature_export.json(),
            "blocked": premature_export.status_code == 400,
        }
    )

    measure = client.post("/api/measure", json={"session_id": success_session})
    measure_json: dict[str, Any]
    try:
        measure_json = measure.json()
    except Exception as exc:  # noqa: BLE001 - diagnostic only.
        measure_json = {"error": str(exc)}
    measure_check: dict[str, Any] = {
        "name": "measure_after_manual_calibration",
        "status_code": measure.status_code,
        "json": measure_json,
        "passed": measure.status_code == 200 and bool(measure_json.get("formal_points")),
        "note": "Synthetic fish-like image may fail segmentation/preannotation; endpoint plumbing is still exercised.",
    }
    report["checks"].append(measure_check)

    if measure_check["passed"]:
        formal_points = measure_json.get("formal_points", {})
        update = client.post(
            "/api/update-points",
            json={
                "session_id": success_session,
                "formal_points": formal_points,
                "measurement_overrides": {"source": "v2_backend_smoke_test"},
            },
        )
        expected_formats = ["csv", "xlsx", "json", "preview_png"]
        export = client.post(
            "/api/export",
            json={"session_id": success_session, "formats": expected_formats},
        )
        export_json = export.json()
        export_files = export_json.get("files", {}) if isinstance(export_json, dict) else {}
        exported_files_exist = all((PROJECT_ROOT / export_files.get(fmt, "")).exists() for fmt in expected_formats)
        report["checks"].append(
            {
                "name": "update_points_and_export",
                "update_status_code": update.status_code,
                "export_status_code": export.status_code,
                "export_json": export_json,
                "expected_formats": expected_formats,
                "exported_files_exist": exported_files_exist,
                "passed": update.status_code == 200
                and export.status_code == 200
                and export_json.get("version") == "2.0.0"
                and all(fmt in export_files for fmt in expected_formats)
                and exported_files_exist,
            }
        )

        exports = client.get("/api/exports")
        exports_json = exports.json()
        report["checks"].append(
            {
                "name": "list_exports",
                "status_code": exports.status_code,
                "json": exports_json,
                "passed": exports.status_code == 200
                and exports_json.get("version") == "2.0.0"
                and any(item.get("session_id") == success_session for item in exports_json.get("exports", [])),
            }
        )

        session_lookup = client.get(f"/api/session/{success_session}")
        session_json = session_lookup.json()
        serialized_session = json.dumps(session_json, ensure_ascii=False)
        report["checks"].append(
            {
                "name": "session_lookup_public_payload",
                "status_code": session_lookup.status_code,
                "json": session_json,
                "passed": session_lookup.status_code == 200
                and session_json.get("version") == "2.0.0"
                and session_json.get("measurement", {}).get("coordinate_space") == "warped_image"
                and "original_path" not in serialized_session
                and "warped_image_path" not in serialized_session,
            }
        )

    report["total_checks"] = len(report["checks"])
    report["passed_checks"] = sum(1 for check in report["checks"] if _check_passed(check))
    report["failed_checks"] = report["total_checks"] - report["passed_checks"]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Smoke test report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
