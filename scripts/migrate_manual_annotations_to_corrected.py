"""Migrate first-pass manual annotations into the v0.3 corrected-label store.

The first manual pass is the current 16-keypoint ground truth. This script
copies those labels into corrected_keypoints without using model predictions as
training labels.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph.config import BODY_AXIS_POINT_KEYS, CAUDAL_TIP_CANDIDATE_KEYS, KEYPOINT_DEFS  # noqa: E402
from siganusmorph.image_utils import ensure_rgb  # noqa: E402
from siganusmorph.io_utils import sanitize_filename_stem  # noqa: E402
from siganusmorph.measurements import build_result_row, calculate_measurements  # noqa: E402
from siganusmorph.real_dataset import (  # noqa: E402
    corrected_annotation_dir,
    relpath,
)
from siganusmorph.visualization import draw_keypoints_and_measurements  # noqa: E402


SOURCE_DIR = PROJECT_ROOT / "results" / "real_annotation_5_15" / "keypoints"
SOURCE_ANNOTATION_DIR = PROJECT_ROOT / "results" / "real_annotation_5_15" / "annotations"
OUTPUT_ROOT = corrected_annotation_dir(PROJECT_ROOT)
REPORT_PATH = OUTPUT_ROOT / "migration_report.csv"
MEASUREMENTS_CSV = OUTPUT_ROOT / "measurements_corrected.csv"
MEASUREMENTS_XLSX = OUTPUT_ROOT / "measurements_corrected.xlsx"
KEY_ORDER = [f"{definition.code}_{definition.name}" for definition in KEYPOINT_DEFS]
DEFAULT_MM_PER_PIXEL = 0.1


def base_stem_from_image_name(image_name: str) -> str:
    return Path(image_name).stem.removesuffix("_warped")


def warped_image_path(image_name: str) -> Path:
    return PROJECT_ROOT / "data" / "real_images_warped" / f"{base_stem_from_image_name(image_name)}_warped.png"


def full_to_short_keypoints(full_keypoints: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    short: dict[str, dict[str, float]] = {}
    for definition in KEYPOINT_DEFS:
        full_key = f"{definition.code}_{definition.name}"
        value = full_keypoints.get(full_key)
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            continue
        short[definition.name] = {"x": float(value[0]), "y": float(value[1])}
    return short


def full_keypoints_from_short(short: Mapping[str, Mapping[str, float]]) -> dict[str, list[float]]:
    full: dict[str, list[float]] = {}
    for definition in KEYPOINT_DEFS:
        point = short.get(definition.name)
        if point is None:
            continue
        full[f"{definition.code}_{definition.name}"] = [float(point["x"]), float(point["y"])]
    return full


def complete_keypoints(full_keypoints: Mapping[str, Any]) -> tuple[bool, list[str]]:
    missing = []
    for key in KEY_ORDER:
        value = full_keypoints.get(key)
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            missing.append(key)
    return not missing, missing


def clear_previous_tables() -> None:
    for path in (MEASUREMENTS_CSV, MEASUREMENTS_XLSX, OUTPUT_ROOT / "correction_summary.csv", REPORT_PATH):
        if path.exists():
            path.unlink()
    for folder in (OUTPUT_ROOT / "keypoints", OUTPUT_ROOT / "annotations"):
        folder.mkdir(parents=True, exist_ok=True)
        for path in folder.glob("real_*"):
            if path.is_file():
                path.unlink()


def serialize_derived_points(measurements: Mapping[str, Any]) -> dict[str, list[float]]:
    derived = measurements.get("derived_points", {})
    out: dict[str, list[float]] = {}
    if isinstance(derived, Mapping):
        for key, value in derived.items():
            point = value
            if isinstance(point, Mapping):
                out[str(key)] = [float(point["x"]), float(point["y"])]
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                out[str(key)] = [float(point[0]), float(point[1])]
    return out


def migrate_one(json_path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    image_name = str(payload.get("image_name") or f"{json_path.stem.removesuffix('_keypoints')}_warped.png")
    specimen_id = str(payload.get("specimen_id", ""))
    source_keypoints = payload.get("keypoints", {})
    if not isinstance(source_keypoints, Mapping):
        source_keypoints = {}
    is_complete, missing = complete_keypoints(source_keypoints)
    report: dict[str, Any] = {
        "source_json": relpath(json_path, PROJECT_ROOT),
        "image_name": image_name,
        "specimen_id": specimen_id,
        "migrated": False,
        "missing_keypoints": ";".join(missing),
        "output_json": "",
        "review_reason": "",
    }
    if not is_complete:
        report["review_reason"] = "missing_16_keypoints"
        return report, None

    image_path = warped_image_path(image_name)
    if not image_path.exists():
        report["review_reason"] = f"missing_warped_image:{relpath(image_path, PROJECT_ROOT)}"
        return report, None

    keypoints = full_to_short_keypoints(source_keypoints)
    mm_per_pixel = float(payload.get("mm_per_pixel") or DEFAULT_MM_PER_PIXEL)
    axis_mode_request = str(payload.get("axis_mode_request") or "auto")
    measurements = calculate_measurements(keypoints, mm_per_pixel, axis_mode_request)
    result_row = build_result_row(
        image_name=image_name,
        specimen_id=specimen_id,
        source_type="real",
        scale_method=str(payload.get("scale_method") or "prewarped_board_coordinate"),
        mm_per_pixel=mm_per_pixel,
        keypoints=keypoints,
        measurements=measurements,
        needs_review=bool(payload.get("needs_review", False)),
        notes=str(payload.get("notes", "")),
    )
    result_row["annotation_mode"] = "manual_original"
    result_row["annotation_status"] = "corrected_and_confirmed"
    result_row["review_reason"] = str(payload.get("review_reason", ""))

    corrected_full = full_keypoints_from_short(keypoints)
    metadata = {
        "annotation_mode": "manual_original",
        "annotation_status": "corrected_and_confirmed",
        "migrated_from_previous_annotation": True,
        "source_annotation_path": str(json_path),
        "model_keypoints_raw": {},
        "mask_suggestions": {},
        "corrected_keypoints": corrected_full,
        "derived_points": measurements.get("derived_points", {}),
        "keypoint_edit_log": [],
        "qc_results": {},
        "body_axis_points_order": list(BODY_AXIS_POINT_KEYS),
        "caudal_tip_candidates": list(CAUDAL_TIP_CANDIDATE_KEYS),
    }
    keypoint_dir = OUTPUT_ROOT / "keypoints"
    annotation_dir = OUTPUT_ROOT / "annotations"
    keypoint_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    base = sanitize_filename_stem(base_stem_from_image_name(image_name))
    annotated_path = annotation_dir / f"{base}_annotated.png"
    source_annotated = SOURCE_ANNOTATION_DIR / f"{base}_annotated.png"
    if source_annotated.exists():
        shutil.copy2(source_annotated, annotated_path)
    else:
        with Image.open(image_path) as image:
            rgb = ensure_rgb(image.convert("RGB"))
        annotated = draw_keypoints_and_measurements(
            rgb,
            keypoints,
            measurements,
            image_name=image_name,
            specimen_id=specimen_id,
            axis_mode_selected=str(measurements.get("axis_mode_selected", "")),
        )
        Image.fromarray(ensure_rgb(annotated)).save(annotated_path)
    out_payload = {
        "annotation_schema_version": payload.get("annotation_schema_version", "v0.3_16kp_p7v"),
        "board_version": payload.get("board_version", "a3_v2_charuco_plumb"),
        "image_name": image_name,
        "source_type": "real",
        "specimen_id": specimen_id,
        "annotator": str(payload.get("annotator", "")),
        "created_at": str(payload.get("created_at", "")),
        "annotation_mode": "manual_original",
        "annotation_status": "corrected_and_confirmed",
        "migrated_from_previous_annotation": True,
        "source_annotation_path": str(json_path),
        "axis_mode_selected": measurements.get("axis_mode_selected", ""),
        "axis_mode_request": axis_mode_request,
        "scale_method": str(payload.get("scale_method") or "prewarped_board_coordinate"),
        "mm_per_pixel": mm_per_pixel,
        "keypoints": corrected_full,
        "model_keypoints_raw": {},
        "mask_suggestions": {},
        "corrected_keypoints": corrected_full,
        "derived_points": serialize_derived_points(measurements),
        "keypoint_edit_log": [],
        "qc_results": {},
        "body_axis_points_order": list(BODY_AXIS_POINT_KEYS),
        "caudal_tip_candidates": list(CAUDAL_TIP_CANDIDATE_KEYS),
        "caudal_tip_selected": measurements.get("caudal_tip_selected", ""),
        "needs_review": bool(payload.get("needs_review", False)),
        "review_reason": str(payload.get("review_reason", "")),
        "notes": str(payload.get("notes", "")),
        "measurements": measurements,
        "preannotation": metadata,
    }
    json_out = keypoint_dir / f"{base}_keypoints.json"
    json_out.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report["migrated"] = True
    report["output_json"] = relpath(json_out, PROJECT_ROOT)
    return report, result_row


def main() -> None:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Manual annotation directory not found: {SOURCE_DIR}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    clear_previous_tables()

    rows: list[dict[str, Any]] = []
    measurement_rows: list[dict[str, Any]] = []
    for path in sorted(SOURCE_DIR.glob("real_*_keypoints.json")):
        report, result_row = migrate_one(path)
        rows.append(report)
        if result_row is not None:
            measurement_rows.append(result_row)

    if measurement_rows:
        measurements_df = pd.DataFrame(measurement_rows)
        measurements_df.to_csv(MEASUREMENTS_CSV, index=False, encoding="utf-8-sig")
        measurements_df.to_excel(MEASUREMENTS_XLSX, index=False)
    correction_summary = OUTPUT_ROOT / "correction_summary.csv"
    correction_summary.write_text(
        "image_name,specimen_id,keypoint_name,model_x,model_y,corrected_x,corrected_y,movement_px,movement_mm,has_mask_suggestion,suggestion_x,suggestion_y,corrected_to_suggestion_px,qc_flag\n",
        encoding="utf-8-sig",
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source_json", "image_name", "specimen_id", "migrated", "missing_keypoints", "output_json", "review_reason"]
    with REPORT_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    migrated_count = sum(1 for row in rows if row["migrated"])
    print(f"Migrated {migrated_count}/{len(rows)} manual annotations to corrected_keypoints.")
    print(f"Output directory: {relpath(OUTPUT_ROOT, PROJECT_ROOT)}")
    print(f"Report: {relpath(REPORT_PATH, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
