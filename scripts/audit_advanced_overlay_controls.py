"""Audit Advanced overlay controls wiring for the Streamlit UI.

This script is intentionally static/lightweight: it scans the main page,
Review Queue page, and visualization module for each Advanced overlay control
and emits inventory/audit CSV files. It does not modify labels, measurements,
or historical result files.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # noqa: BLE001 - previews are auxiliary audit artifacts.
    Image = None
    ImageDraw = None
    ImageFont = None


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "ui_overlay_audit_v0.6.9"
PREVIEW_DIR = OUT_DIR / "smoke_test_previews"


@dataclass(frozen=True)
class OverlayControl:
    label: str
    canonical_key: str
    app_key: str
    review_key: str
    overlay_arg: str
    default: bool
    effect: str
    missing_message: str


CONTROLS = [
    OverlayControl("Show geometric P8/P9", "show_geometric_p8p9", "show_geometric_p8p9_main", "show_geometric_p8p9_review", "show_geometric_body_depth", False, "Draw blue P8_geo/P9_geo points.", "No geometric P8/P9 data available."),
    OverlayControl("Show body-depth max section", "show_body_depth_max_section", "show_body_depth_max_section_main", "show_body_depth_max_section_review", "show_body_depth_width_profile", False, "Draw bright-blue geometric P8-P9 max-section line.", "No body_depth_geometry available."),
    OverlayControl("Show body-depth comparison labels", "show_body_depth_comparison_labels", "show_body_depth_comparison_labels_main", "show_body_depth_comparison_labels_review", "show_body_depth_comparison_labels", False, "Draw body-depth current/geometric/difference label.", "No body_depth_geometry available."),
    OverlayControl("Show body-depth outer boundary contour", "show_body_depth_boundary_contour", "show_body_depth_outer_boundary_contour_main", "show_body_depth_outer_boundary_contour_review", "show_body_depth_boundary_contour", False, "Draw dorsal/ventral outer body-depth contours.", "No body-depth outer boundary contour available."),
    OverlayControl("Show P6 geometry QC", "show_p6_geometry_qc", "show_p6_geometry_qc_main", "show_p6_geometry_qc_review", "show_p6_geometry_qc", False, "Draw P6 warning marker when P6 geometry QC fails.", "No P6 geometry QC data available."),
    OverlayControl("Show P6 fallback suggestion", "show_p6_fallback_suggestion", "show_p6_fallback_suggestion_main", "show_p6_fallback_suggestion_review", "show_p6_fallback_suggestion", False, "Draw P6_gap/fallback suggestion point.", "No P6 fallback suggestion available."),
    OverlayControl("Show tail fork gap region", "show_tail_fork_gap_region", "show_tail_fork_gap_region_main", "show_tail_fork_gap_region_review", "show_tail_fork_gap_region", False, "Draw tail-gap centerline/tip region.", "No tail fork gap region available."),
    OverlayControl("Show geometric P10/P11", "show_geometric_p10p11", "show_geometric_p10p11_main", "show_geometric_p10p11_review", "show_geometric_peduncle_depth", False, "Draw teal P10_geo/P11_geo points.", "No geometric P10/P11 data available."),
    OverlayControl("Show peduncle-depth min section", "show_peduncle_depth_min_section", "show_peduncle_depth_min_section_main", "show_peduncle_depth_min_section_review", "show_peduncle_depth_width_profile", False, "Draw teal geometric P10-P11 min-section line.", "No peduncle_depth_geometry available."),
    OverlayControl("Show peduncle-depth comparison labels", "show_peduncle_depth_comparison_labels", "show_peduncle_depth_comparison_labels_main", "show_peduncle_depth_comparison_labels_review", "show_peduncle_depth_comparison_labels", False, "Draw peduncle-depth current/geometric/difference label.", "No peduncle_depth_geometry available."),
    OverlayControl("Show fin suppression regions", "show_fin_suppression_regions", "show_fin_suppression_regions_main", "show_fin_suppression_regions_review", "show_fin_suppression_regions", False, "Draw yellow fin-suppression regions.", "No fin_suppression_regions available."),
    OverlayControl("Show body trunk contours", "show_body_trunk_contours", "show_body_trunk_contours_main", "show_body_trunk_contours_review", "show_body_trunk_contours", False, "Draw dorsal/ventral trunk contours.", "No body trunk contours available."),
    OverlayControl("Show P3 operculum suggestion", "show_p3_operculum_suggestion", "show_p3_operculum_suggestion_main", "show_p3_operculum_suggestion_review", "show_p3_operculum_suggestion", False, "Draw P3 operculum edge suggestion and drift warning.", "No P3 operculum suggestion available."),
    OverlayControl("Show current P8-P9 line", "show_current_body_depth_line", "show_current_p8p9_line_main", "show_current_p8p9_line_review", "show_current_body_depth_line", False, "Draw current working P8-P9 line.", "No current "),
    OverlayControl("Show current P10-P11 line", "show_current_peduncle_depth_line", "show_current_p10p11_line_main", "show_current_p10p11_line_review", "show_current_peduncle_depth_line", False, "Draw current working P10-P11 line.", "No current "),
    OverlayControl("Show model axis", "show_model_axis", "show_model_axis_overlay_main", "show_model_axis_overlay_review", "show_model_axis", False, "Draw model_axis in orange.", "No model_axis available."),
    OverlayControl("Show debug heatmap points", "show_debug_heatmap_points", "show_debug_heatmap_points_main", "show_debug_heatmap_points_review", "show_heatmap_points", False, "Draw heatmap/v0.1/v0.5 debug points.", "No heatmap debug points available."),
    OverlayControl("Show point source labels", "show_point_source_labels", "show_point_source_labels_overlay_main", "show_point_source_labels_review", "show_point_source_labels", False, "Annotate points with source labels.", "No comparison preannotation points available."),
    OverlayControl("Show QC warnings", "show_qc_warnings", "show_qc_warnings_overlay_main", "show_qc_warnings_overlay_review", "show_qc_warnings", True, "Draw QC warning markers and labels.", "No QC warning data available."),
]


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return ""


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_inventory(app_text: str, review_text: str, viz_text: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for page, text, key_attr, file_name in (
        ("main", app_text, "app_key", "app.py"),
        ("review_queue", review_text, "review_key", "pages/3_realworld_review.py"),
    ):
        for control in CONTROLS:
            local_key = getattr(control, key_attr)
            defined = control.label in text and local_key in text
            used = control.overlay_arg in viz_text
            rows.append(
                {
                    "page": page,
                    "control_label": control.label,
                    "session_state_key": control.canonical_key,
                    "default_value": control.default,
                    "defined_in_file": file_name if defined else "",
                    "used_in_overlay": used,
                    "overlay_argument_name": control.overlay_arg,
                    "expected_visual_effect": control.effect,
                    "notes": "ok" if defined and used else "needs_attention",
                }
            )
    return rows


def build_static_audit(app_text: str, review_text: str, viz_text: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for control in CONTROLS:
        defined_app = control.label in app_text and control.app_key in app_text
        defined_review = control.label in review_text and control.review_key in review_text
        session_key_present = control.canonical_key in app_text or control.canonical_key in review_text
        used_viz = control.overlay_arg in viz_text
        missing_message = control.missing_message in viz_text
        has_availability = (
            "Overlay debug / audit" in app_text
            and "Overlay debug / audit" in review_text
        )
        status = "pass" if defined_app and defined_review and session_key_present and used_viz and missing_message else "check"
        suggested = "" if status == "pass" else "Ensure checkbox, session sync, visualization read/draw logic, and missing-data message all exist."
        rows.append(
            {
                "control_label": control.label,
                "defined_in_app": defined_app,
                "defined_in_review": defined_review,
                "session_key": control.canonical_key,
                "used_in_visualization": used_viz,
                "has_data_availability_check": has_availability,
                "has_missing_data_message": missing_message,
                "status": status,
                "suggested_fix": suggested,
            }
        )
    return rows


def choose_smoke_images() -> list[str]:
    candidates: list[str] = []
    for csv_path, col in (
        (ROOT / "results" / "realworld_review_5_24_v0.6.5" / "preannotation_qc.csv", "image_name"),
        (ROOT / "results" / "batch_measurement_v0.6.4_stable" / "final_analysis_dataset.csv", "image_name"),
        (ROOT / "results" / "batch_measurement_v0.6.4_stable" / "batch_measurement_manifest.csv", "image_name"),
    ):
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                value = str(row.get(col, "") or "").strip()
                if value and value not in candidates:
                    candidates.append(value)
                if len(candidates) >= 3:
                    return candidates[:3]
    return candidates[:3] or ["static_sample_1", "static_sample_2", "static_sample_3"]


def build_smoke_rows(static_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    status_by_label = {str(row["control_label"]): row for row in static_rows}
    images = choose_smoke_images()
    rows: list[dict[str, object]] = []
    for image_name in images:
        for control in CONTROLS:
            static = status_by_label[control.label]
            visible = bool(static["defined_in_app"] and static["defined_in_review"])
            session_updated = control.canonical_key in read_text(ROOT / "app.py") and control.canonical_key in read_text(ROOT / "pages" / "3_realworld_review.py")
            overlay_changed = bool(static["used_in_visualization"])
            missing_message = bool(static["has_missing_data_message"])
            rows.append(
                {
                    "image_name": image_name,
                    "control_label": control.label,
                    "checkbox_visible": visible,
                    "session_state_updated": session_updated,
                    "overlay_changed": overlay_changed,
                    "data_available": "not_runtime_checked",
                    "missing_data_message_shown": missing_message,
                    "pass_fail": "pass" if visible and session_updated and overlay_changed and missing_message else "check",
                    "notes": "Static smoke check; runtime image-specific data availability is reported in the UI Overlay debug/audit panel.",
                }
            )
    return rows


def write_summary(static_rows: list[dict[str, object]]) -> None:
    summary = {
        "controls_audited": len(CONTROLS),
        "controls_fully_connected": sum(1 for row in static_rows if row["status"] == "pass"),
        "controls_needing_attention": sum(1 for row in static_rows if row["status"] != "pass"),
        "main_page_audit": "pass" if all(row["defined_in_app"] for row in static_rows) else "check",
        "review_queue_audit": "pass" if all(row["defined_in_review"] for row in static_rows) else "check",
        "visualization_audit": "pass" if all(row["used_in_visualization"] for row in static_rows) else "check",
    }
    (OUT_DIR / "advanced_overlay_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_smoke_preview_images(smoke_rows: list[dict[str, object]]) -> None:
    if Image is None or ImageDraw is None:
        return
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in smoke_rows:
        grouped.setdefault(str(row["image_name"]), []).append(row)
    font = ImageFont.load_default() if ImageFont is not None else None
    for image_name, rows in grouped.items():
        width = 1200
        line_height = 24
        height = 90 + line_height * (len(rows) + 2)
        canvas = Image.new("RGB", (width, height), (245, 247, 250))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, width, 64), fill=(28, 44, 64))
        draw.text((18, 18), f"Advanced overlay static smoke: {image_name}", fill=(255, 255, 255), font=font)
        y = 82
        header = "control | visible | session | overlay | missing-message | status"
        draw.text((18, y), header, fill=(20, 20, 20), font=font)
        y += line_height
        for row in rows:
            status = str(row["pass_fail"])
            fill = (10, 120, 45) if status == "pass" else (180, 90, 0)
            line = (
                f"{row['control_label']} | {row['checkbox_visible']} | "
                f"{row['session_state_updated']} | {row['overlay_changed']} | "
                f"{row['missing_data_message_shown']} | {status}"
            )
            draw.text((18, y), line[:170], fill=fill, font=font)
            y += line_height
        safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in image_name)
        canvas.save(PREVIEW_DIR / f"{safe_name}_overlay_static_smoke.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    app_text = read_text(ROOT / "app.py")
    review_text = read_text(ROOT / "pages" / "3_realworld_review.py")
    viz_text = read_text(ROOT / "siganusmorph" / "visualization.py")

    inventory_rows = build_inventory(app_text, review_text, viz_text)
    static_rows = build_static_audit(app_text, review_text, viz_text)
    smoke_rows = build_smoke_rows(static_rows)

    write_csv(
        OUT_DIR / "advanced_overlay_controls_inventory.csv",
        inventory_rows,
        [
            "page",
            "control_label",
            "session_state_key",
            "default_value",
            "defined_in_file",
            "used_in_overlay",
            "overlay_argument_name",
            "expected_visual_effect",
            "notes",
        ],
    )
    write_csv(
        OUT_DIR / "advanced_overlay_static_audit.csv",
        static_rows,
        [
            "control_label",
            "defined_in_app",
            "defined_in_review",
            "session_key",
            "used_in_visualization",
            "has_data_availability_check",
            "has_missing_data_message",
            "status",
            "suggested_fix",
        ],
    )
    write_csv(
        OUT_DIR / "advanced_overlay_smoke_test.csv",
        smoke_rows,
        [
            "image_name",
            "control_label",
            "checkbox_visible",
            "session_state_updated",
            "overlay_changed",
            "data_available",
            "missing_data_message_shown",
            "pass_fail",
            "notes",
        ],
    )
    write_summary(static_rows)
    write_smoke_preview_images(smoke_rows)
    print(f"Advanced overlay controls audited: {len(CONTROLS)}")
    print(f"Fully connected: {sum(1 for row in static_rows if row['status'] == 'pass')}")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
