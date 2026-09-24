"""Synthetic annotation trial helpers.

Synthetic trial data is intentionally kept separate from real specimen data so
it can be used for UI, export, and measurement sanity checks without leaking
into formal training sets.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from PIL import Image

from .config import (
    BODY_AXIS_POINT_KEYS,
    CAUDAL_TIP_CANDIDATE_KEYS,
    DERIVED_POINT_DEFS,
    KEYPOINT_DEFS,
    RESULT_COLUMNS,
)
from .image_utils import ensure_rgb
from .io_utils import sanitize_filename_stem


ANNOTATION_SCHEMA_VERSION = "v0.3_16kp_p7v"
BOARD_VERSION = "a3_v2_charuco_plumb"
SYNTHETIC_DIR = Path("data") / "synthetic_test_images"
SYNTHETIC_OUTPUT_DIR = Path("results") / "synthetic_annotation_trial"
SYNTHETIC_MEASUREMENTS_CSV = "measurements_synthetic.csv"
SYNTHETIC_MEASUREMENTS_XLSX = "measurements_synthetic.xlsx"
SYNTHETIC_SUMMARY_CSV = "annotation_summary.csv"


def synthetic_image_dir(project_root: Path) -> Path:
    return project_root / SYNTHETIC_DIR


def synthetic_output_dir(project_root: Path) -> Path:
    return project_root / SYNTHETIC_OUTPUT_DIR


def list_synthetic_trial_images(project_root: Path) -> list[Path]:
    root = synthetic_image_dir(project_root)
    return sorted(root.glob("synthetic_*.png")) if root.exists() else []


def is_synthetic_trial_image(image_name: str) -> bool:
    path = Path(image_name)
    return path.name.startswith("synthetic_") and path.suffix.lower() == ".png"


def ensure_synthetic_trial_dirs(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "warped_images": root / "warped_images",
        "keypoints": root / "keypoints",
        "annotations": root / "annotations",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def review_reason_from_measurements(measurements: Mapping[str, Any] | None) -> str:
    if not measurements:
        return ""
    curvature = measurements.get("curvature_index_polyline")
    if curvature is not None and float(curvature) > 1.05:
        return "curvature_index_polyline > 1.05; fish body is noticeably curved and should be reviewed."
    return ""


def _point_list(point: Mapping[str, float]) -> list[float]:
    return [float(point["x"]), float(point["y"])]


def _format_keypoints(keypoints: Mapping[str, Mapping[str, float]]) -> dict[str, list[float]]:
    formatted: dict[str, list[float]] = {}
    for definition in KEYPOINT_DEFS:
        point = keypoints.get(definition.name)
        if point is not None:
            formatted[f"{definition.code}_{definition.name}"] = _point_list(point)
    return formatted


def _format_derived_points(measurements: Mapping[str, Any]) -> dict[str, list[float]]:
    formatted: dict[str, list[float]] = {}
    derived_points = measurements.get("derived_points", {})
    if not isinstance(derived_points, Mapping):
        return formatted
    for definition in DERIVED_POINT_DEFS:
        key = f"{definition.code}_{definition.name}"
        point = derived_points.get(key) or derived_points.get(definition.name)
        if isinstance(point, Mapping):
            formatted[key] = _point_list(point)
    return formatted


def _save_image(image: np.ndarray | Image.Image, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, Image.Image):
        image.convert("RGB").save(output_path)
    else:
        Image.fromarray(ensure_rgb(image)).save(output_path)
    return output_path


def _ordered_dataframe(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    for column in RESULT_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df.loc[:, list(RESULT_COLUMNS)]


def upsert_synthetic_measurement_row(row: Mapping[str, Any], trial_root: Path) -> dict[str, Path]:
    trial_root.mkdir(parents=True, exist_ok=True)
    csv_path = trial_root / SYNTHETIC_MEASUREMENTS_CSV
    xlsx_path = trial_root / SYNTHETIC_MEASUREMENTS_XLSX

    if csv_path.exists():
        existing = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    else:
        existing = pd.DataFrame(columns=list(RESULT_COLUMNS))
    for column in RESULT_COLUMNS:
        if column not in existing.columns:
            existing[column] = ""

    image_name = str(row.get("image_name", ""))
    specimen_id = str(row.get("specimen_id", ""))
    if not existing.empty:
        keep = ~(
            (existing["image_name"].astype(str) == image_name)
            & (existing["specimen_id"].astype(str) == specimen_id)
        )
        existing = existing.loc[keep, list(RESULT_COLUMNS)]

    new_row = _ordered_dataframe([row])
    combined = new_row if existing.empty else pd.concat([existing, new_row], ignore_index=True)
    combined = combined.loc[:, list(RESULT_COLUMNS)]
    combined.to_csv(csv_path, index=False, encoding="utf-8-sig")
    combined.to_excel(xlsx_path, index=False)
    return {"csv": csv_path, "xlsx": xlsx_path}


def save_synthetic_trial_artifacts(
    project_root: Path,
    image_name: str,
    working_image: np.ndarray | Image.Image,
    annotated_image: np.ndarray | Image.Image,
    keypoints: Mapping[str, Mapping[str, float]],
    measurements: Mapping[str, Any],
    result_row: Mapping[str, Any],
    specimen_id: str,
    annotator: str,
    notes: str,
    axis_mode_request: str,
    scale_method: str,
    mm_per_pixel: float,
    board_version: str = BOARD_VERSION,
) -> dict[str, Path]:
    trial_root = synthetic_output_dir(project_root)
    paths = ensure_synthetic_trial_dirs(trial_root)
    stem = sanitize_filename_stem(Path(image_name).stem)
    created_at = datetime.now().isoformat(timespec="seconds")

    warped_name = f"{stem}_warped.png"
    keypoints_name = f"{stem}_keypoints.json"
    annotated_name = f"{stem}_annotated.png"
    warped_path = _save_image(working_image, paths["warped_images"] / warped_name)
    annotated_path = _save_image(annotated_image, paths["annotations"] / annotated_name)

    review_reason = review_reason_from_measurements(measurements)
    needs_review = bool(result_row.get("needs_review", False)) or bool(review_reason)
    payload = {
        "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
        "board_version": board_version,
        "image_name": image_name,
        "warped_image_name": warped_name,
        "source_type": "synthetic",
        "specimen_id": specimen_id,
        "annotator": annotator,
        "created_at": created_at,
        "axis_mode_selected": measurements.get("axis_mode_selected", ""),
        "axis_mode_request": axis_mode_request,
        "scale_method": scale_method,
        "mm_per_pixel": float(mm_per_pixel),
        "keypoints": _format_keypoints(keypoints),
        "derived_points": _format_derived_points(measurements),
        "body_axis_points_order": list(BODY_AXIS_POINT_KEYS),
        "caudal_tip_candidates": list(CAUDAL_TIP_CANDIDATE_KEYS),
        "caudal_tip_selected": measurements.get("caudal_tip_selected", ""),
        "needs_review": needs_review,
        "review_reason": review_reason,
        "notes": notes,
        "measurements": dict(measurements),
    }
    json_path = paths["keypoints"] / keypoints_name
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    measurement_paths = upsert_synthetic_measurement_row(result_row, trial_root)
    write_annotation_summary(project_root, trial_root)
    return {
        "warped": warped_path,
        "keypoints": json_path,
        "annotations": annotated_path,
        "csv": measurement_paths["csv"],
        "xlsx": measurement_paths["xlsx"],
        "summary": trial_root / SYNTHETIC_SUMMARY_CSV,
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def build_annotation_summary_rows(project_root: Path, trial_root: Path | None = None) -> list[dict[str, Any]]:
    trial_root = trial_root or synthetic_output_dir(project_root)
    images = list_synthetic_trial_images(project_root)
    rows: list[dict[str, Any]] = []
    required_keypoint_count = len(KEYPOINT_DEFS)

    for image_path in images:
        stem = sanitize_filename_stem(image_path.stem)
        json_path = trial_root / "keypoints" / f"{stem}_keypoints.json"
        annotated_path = trial_root / "annotations" / f"{stem}_annotated.png"
        warped_path = trial_root / "warped_images" / f"{stem}_warped.png"
        payload = _read_json(json_path) if json_path.exists() else None
        keypoints = payload.get("keypoints", {}) if isinstance(payload, Mapping) else {}
        derived_points = payload.get("derived_points", {}) if isinstance(payload, Mapping) else {}
        manual_keypoint_count = len(keypoints) if isinstance(keypoints, Mapping) else 0
        p7v_exists = isinstance(derived_points, Mapping) and "P7V_caudal_fin_posterior_endpoint" in derived_points
        complete = (
            json_path.exists()
            and annotated_path.exists()
            and warped_path.exists()
            and manual_keypoint_count == required_keypoint_count
            and p7v_exists
        )
        rows.append(
            {
                "image_name": image_path.name,
                "source_type": "synthetic",
                "json_exists": json_path.exists(),
                "manual_keypoint_count": manual_keypoint_count,
                "manual_keypoints_ok": manual_keypoint_count == required_keypoint_count,
                "p7v_exists": p7v_exists,
                "warped_exists": warped_path.exists(),
                "annotated_exists": annotated_path.exists(),
                "complete": complete,
                "needs_review": bool(payload.get("needs_review", False)) if isinstance(payload, Mapping) else "",
                "review_reason": payload.get("review_reason", "") if isinstance(payload, Mapping) else "",
                "json_path": str(json_path),
                "warped_path": str(warped_path),
                "annotated_path": str(annotated_path),
            }
        )
    return rows


def write_annotation_summary(project_root: Path, trial_root: Path | None = None) -> Path:
    trial_root = trial_root or synthetic_output_dir(project_root)
    trial_root.mkdir(parents=True, exist_ok=True)
    rows = build_annotation_summary_rows(project_root, trial_root)
    output_path = trial_root / SYNTHETIC_SUMMARY_CSV
    fields = [
        "image_name",
        "source_type",
        "json_exists",
        "manual_keypoint_count",
        "manual_keypoints_ok",
        "p7v_exists",
        "warped_exists",
        "annotated_exists",
        "complete",
        "needs_review",
        "review_reason",
        "json_path",
        "warped_path",
        "annotated_path",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def check_synthetic_annotation_trial(project_root: Path) -> dict[str, Any]:
    trial_root = synthetic_output_dir(project_root)
    summary_path = write_annotation_summary(project_root, trial_root)
    rows = build_annotation_summary_rows(project_root, trial_root)
    measurements_csv = trial_root / SYNTHETIC_MEASUREMENTS_CSV
    measurement_rows = 0
    if measurements_csv.exists():
        measurement_rows = len(pd.read_csv(measurements_csv, dtype=str, keep_default_na=False))

    errors: list[str] = []
    if len(rows) != 20:
        errors.append(f"Expected 20 synthetic source images, found {len(rows)}.")
    for row in rows:
        if not row["complete"]:
            missing_parts: list[str] = []
            if not row["json_exists"]:
                missing_parts.append("JSON")
            if not row["warped_exists"]:
                missing_parts.append("warped image")
            if not row["annotated_exists"]:
                missing_parts.append("annotated image")
            if not row["manual_keypoints_ok"]:
                missing_parts.append(f"16 manual points (found {row['manual_keypoint_count']})")
            if not row["p7v_exists"]:
                missing_parts.append("P7V")
            errors.append(f"{row['image_name']}: missing/invalid {', '.join(missing_parts)}.")
    if measurement_rows != 20:
        errors.append(f"{SYNTHETIC_MEASUREMENTS_CSV} should have 20 rows, found {measurement_rows}.")

    return {
        "ok": not errors,
        "errors": errors,
        "summary_path": summary_path,
        "measurement_rows": measurement_rows,
        "completed_count": sum(1 for row in rows if row["complete"]),
        "expected_count": len(rows),
        "rows": rows,
    }


def _iter_keypoint_jsons(project_root: Path) -> Iterable[Path]:
    yield from (project_root / "results" / "keypoints").glob("*_keypoints.json")
    yield from (synthetic_output_dir(project_root) / "keypoints").glob("*_keypoints.json")


def export_training_data(project_root: Path, source_type: str) -> dict[str, Any]:
    if source_type not in {"synthetic", "real", "both"}:
        raise ValueError("source_type must be one of: synthetic, real, both")

    export_root = project_root / "results" / "training_exports"
    export_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = export_root / f"siganusmorph_keypoints_{source_type}.jsonl"
    manifest_path = export_root / f"siganusmorph_keypoints_{source_type}_manifest.csv"
    records: list[dict[str, Any]] = []

    for path in _iter_keypoint_jsons(project_root):
        payload = _read_json(path)
        if not isinstance(payload, Mapping):
            continue
        payload_source = payload.get("source_type") or payload.get("metadata", {}).get("source_type", "")
        payload_source = str(payload_source or "unknown")
        if source_type != "both" and payload_source != source_type:
            continue
        records.append({"path": path, "payload": dict(payload), "source_type": payload_source})

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record["payload"], ensure_ascii=False) + "\n")

    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_name", "source_type", "json_path"])
        writer.writeheader()
        for record in records:
            payload = record["payload"]
            writer.writerow(
                {
                    "image_name": payload.get("image_name", ""),
                    "source_type": record["source_type"],
                    "json_path": str(record["path"]),
                }
            )

    return {"count": len(records), "jsonl": jsonl_path, "manifest": manifest_path}
