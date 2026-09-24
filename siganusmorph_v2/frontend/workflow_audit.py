from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_ROOT = Path(__file__).resolve().parent
SRC_ROOT = FRONTEND_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "results" / "v2_user_outputs"
JSON_REPORT = RESULTS_DIR / "v2_frontend_workflow_audit.json"
CSV_REPORT = RESULTS_DIR / "v2_frontend_workflow_audit.csv"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def has_all(text: str, needles: list[str]) -> bool:
    return all(needle in text for needle in needles)


def add_check(checks: list[dict[str, object]], name: str, passed: bool, evidence: str, notes: str = "") -> None:
    checks.append(
        {
            "check_name": name,
            "passed": bool(passed),
            "status": "pass" if passed else "fail",
            "evidence": evidence,
            "notes": notes,
        }
    )


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    app = read(SRC_ROOT / "App.tsx")
    api = read(SRC_ROOT / "api" / "client.ts")
    single = read(SRC_ROOT / "pages" / "SingleFishPage.tsx")
    canvas = read(SRC_ROOT / "components" / "MeasurementCanvas.tsx")
    batch = read(SRC_ROOT / "pages" / "BatchPage.tsx")
    export_page = read(SRC_ROOT / "pages" / "ExportPage.tsx")
    settings = read(SRC_ROOT / "pages" / "SettingsPage.tsx")

    checks: list[dict[str, object]] = []

    add_check(
        checks,
        "navigation_pages_are_wired",
        has_all(app, ["HomePage", "SingleFishPage", "BatchPage", "ExportPage", "SettingsPage", "setPage"]),
        "src/App.tsx",
    )
    add_check(
        checks,
        "ordinary_user_nav_labels_are_chinese",
        has_all(app, ["首页", "单鱼测量", "批量测量", "结果导出", "设置", "蓝子鱼形态测量系统"]),
        "src/App.tsx",
    )
    add_check(
        checks,
        "api_client_has_required_backend_endpoints",
        has_all(api, ["/api/health", "/api/upload", "/api/calibrate", "/api/measure", "/api/update-points", "/api/export"]),
        "src/api/client.ts",
    )
    add_check(
        checks,
        "api_client_hides_tracebacks_from_user_errors",
        has_all(api, ["safeErrorMessage", "Traceback", "处理失败"]),
        "src/api/client.ts",
    )
    add_check(
        checks,
        "single_fish_has_specimen_and_weight_inputs",
        has_all(single, ["specimenId", "weightG", "样本编号 specimen_id", "样品重量 weight_g"]),
        "src/pages/SingleFishPage.tsx",
    )
    add_check(
        checks,
        "single_fish_runs_upload_calibrate_measure_flow",
        has_all(single, ["uploadImage(file", "calibrate(uploaded.session_id)", "measure(uploaded.session_id)"]),
        "src/pages/SingleFishPage.tsx",
    )
    add_check(
        checks,
        "calibration_failure_blocks_measurement",
        has_all(single, ['cal.status !== "success"', "return;", "校准失败"]),
        "src/pages/SingleFishPage.tsx",
    )
    add_check(
        checks,
        "single_fish_uses_warped_image_for_canvas",
        has_all(single, ["calibration?.warped_image_url", "assetUrl(calibration.warped_image_url)", "warped_image_width", "warped_image_height"]),
        "src/pages/SingleFishPage.tsx",
    )
    add_check(
        checks,
        "single_fish_apply_changes_calls_update_points",
        has_all(single, ["applyPoints", "updatePoints(upload.session_id", "setMeasurement(updated)"]),
        "src/pages/SingleFishPage.tsx",
    )
    add_check(
        checks,
        "single_fish_export_calls_backend_and_remembers_history",
        has_all(single, ["exportSession(upload.session_id)", "rememberExport", "保存并导出"]),
        "src/pages/SingleFishPage.tsx",
    )
    add_check(
        checks,
        "measurement_canvas_renders_warped_image_svg",
        has_all(canvas, ["<svg", "<image", "href={imageUrl}", "viewBox={`0 0 ${imageWidth} ${imageHeight}`"]),
        "src/components/MeasurementCanvas.tsx",
    )

    draggable_points = [
        "P1_snout_tip",
        "P2_eye_anterior",
        "P3_operculum_posterior",
        "P4_peduncle_anterior",
        "P5_caudal_base",
        "P6_caudal_fork",
        "P7U_upper_lobe_tip",
        "P7L_lower_lobe_tip",
        "body_depth_upper",
        "body_depth_lower",
        "peduncle_depth_upper",
        "peduncle_depth_lower",
    ]
    add_check(
        checks,
        "measurement_canvas_has_required_draggable_points",
        has_all(canvas, draggable_points),
        "src/components/MeasurementCanvas.tsx",
    )
    drag_array_match = re.search(r"const DRAGGABLE = \[(.*?)\];", canvas, flags=re.DOTALL)
    drag_array_text = drag_array_match.group(1) if drag_array_match else ""
    add_check(
        checks,
        "p7v_is_derived_not_directly_draggable",
        "deriveP7V" in canvas and "P7V" in canvas and "P7V" not in drag_array_text,
        "src/components/MeasurementCanvas.tsx",
    )
    add_check(
        checks,
        "display_to_image_coordinate_mapping_present",
        has_all(canvas, ["eventToImage", "getBoundingClientRect", "clientX", "clientY", "imageWidth", "imageHeight"]),
        "src/components/MeasurementCanvas.tsx",
    )
    add_check(
        checks,
        "large_invisible_hit_radius_present",
        has_all(canvas, ["nearestPoint", "hitRadius", "18 *", "bestDist"]),
        "src/components/MeasurementCanvas.tsx",
    )
    add_check(
        checks,
        "drag_updates_local_state_before_backend_apply",
        has_all(canvas, ["handlePointerMove", "setLocalPoints", "active", "onApply(localPoints"]),
        "src/components/MeasurementCanvas.tsx",
    )
    add_check(
        checks,
        "hover_label_crosshair_magnifier_present",
        has_all(canvas, ["hover", "crosshair", "magnifier", "backgroundPosition", "x=", "y="]),
        "src/components/MeasurementCanvas.tsx",
    )
    add_check(
        checks,
        "keyboard_fine_tuning_present",
        has_all(canvas, ["handleKeyDown", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "shiftKey"]),
        "src/components/MeasurementCanvas.tsx",
    )
    add_check(
        checks,
        "local_measurement_preview_present",
        has_all(canvas, ["localMeasurements", "TL_compressed_virtual_mm", "body_depth_mm", "caudal_peduncle_depth_mm"]),
        "src/components/MeasurementCanvas.tsx",
    )
    add_check(
        checks,
        "batch_page_has_queue_status_and_export_flow",
        has_all(batch, ["uploadImage", "calibrate", "measure", "exportSession", "calibration_failed", "needs_review"]),
        "src/pages/BatchPage.tsx",
    )
    add_check(
        checks,
        "export_page_lists_backend_exports",
        has_all(export_page, ["listExports", "loadExportHistory", "CSV", "Excel", "JSON"]),
        "src/pages/ExportPage.tsx",
    )
    add_check(
        checks,
        "settings_page_reads_backend_settings",
        has_all(settings, ["getSettings", "model_paths", "user_output_root", "admin_output_root"]),
        "src/pages/SettingsPage.tsx",
    )

    public_text = "\n".join([app, single, canvas, batch, export_page, settings])
    forbidden_patterns = [r"\bdebug\b", r"\baudit\b", r"\bexperimental\b", r"Traceback", r"\b[A-Z]:[\\/]"]
    forbidden_hits = []
    for pattern in forbidden_patterns:
        if re.search(pattern, public_text, flags=re.IGNORECASE):
            forbidden_hits.append(pattern)
    add_check(
        checks,
        "ordinary_user_frontend_avoids_debug_and_local_paths",
        not forbidden_hits,
        "src/App.tsx + pages + MeasurementCanvas",
        "\n".join(forbidden_hits),
    )

    passed = sum(1 for item in checks if item["passed"])
    report = {
        "version": "2.0.0",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_checks": len(checks),
        "passed_checks": passed,
        "failed_checks": len(checks) - passed,
        "workflow_static_ready": passed == len(checks),
        "notes": "This audit verifies source/runtime wiring. It does not replace visual/manual validation inside the Tauri window.",
        "checks": checks,
    }
    JSON_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with CSV_REPORT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["check_name", "passed", "status", "evidence", "notes"])
        writer.writeheader()
        writer.writerows(checks)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Frontend workflow audit written to: {JSON_REPORT.relative_to(PROJECT_ROOT)}")
    print(f"Frontend workflow audit CSV written to: {CSV_REPORT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
