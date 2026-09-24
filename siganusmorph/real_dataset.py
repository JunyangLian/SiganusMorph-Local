"""Helpers for the 2026-05-15 real rabbitfish dataset workflow."""

from __future__ import annotations

import csv
import json
import math
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from .annotation import load_keypoints_json
from .aruco_utils import detect_aruco_markers, select_board_config_for_markers, warp_board_by_aruco
from .config import (
    A3_V2_CHARUCO_PLUMB_CONFIG,
    BODY_AXIS_POINT_KEYS,
    CAUDAL_TIP_CANDIDATE_KEYS,
    DERIVED_POINT_DEFS,
    KEYPOINT_DEFS,
    RESULT_COLUMNS,
)
from .image_utils import ensure_rgb, load_image_file
from .io_utils import sanitize_filename_stem


REAL_SOURCE_DIR = Path(r"C:\Users\羊叽叽\Desktop\caiyang\5.15")
REAL_RAW_DIR = Path("data") / "real_images_raw_png"
REAL_WARPED_DIR = Path("data") / "real_images_warped"
REAL_MANIFEST = Path("data") / "real_images_manifest.csv"
SPECIMEN_GROUP_SUMMARY = Path("data") / "specimen_group_summary.csv"
REAL_ANNOTATION_DIR = Path("results") / "real_annotation_5_15"
CORRECTED_ANNOTATION_DIR = Path("results") / "real_annotation_5_15_v0.3_corrected"
REAL_MEASUREMENTS_CSV = "measurements_real.csv"
REAL_MEASUREMENTS_XLSX = "measurements_real.xlsx"
CORRECTED_MEASUREMENTS_CSV = "measurements_corrected.csv"
CORRECTED_MEASUREMENTS_XLSX = "measurements_corrected.xlsx"
REAL_ANNOTATION_SUMMARY_CSV = "annotation_summary.csv"
REAL_CORRECTION_SUMMARY_CSV = "correction_summary.csv"
READINESS_REPORT_CSV = "readiness_report.csv"
YOLO_DATASET_DIR = Path("datasets") / "siganusmorph_real_5_15_yolopose"
CURATION_COLUMNS = [
    "keep_for_annotation",
    "keep_for_training",
    "duplicate_group",
    "exclude_reason",
    "quality_score",
    "curation_notes",
]
ANNOTATION_SCHEMA_VERSION = "v0.3_16kp_p7v"
BOARD_VERSION = "a3_v2_charuco_plumb"
RANDOM_SEED = 20260515
CLASS_ID = 0
CLASS_NAME = "rabbitfish"


def relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def real_manifest_path(project_root: Path) -> Path:
    return project_root / REAL_MANIFEST


def real_raw_dir(project_root: Path) -> Path:
    return project_root / REAL_RAW_DIR


def real_warped_dir(project_root: Path) -> Path:
    return project_root / REAL_WARPED_DIR


def real_annotation_dir(project_root: Path) -> Path:
    return project_root / REAL_ANNOTATION_DIR


def corrected_annotation_dir(project_root: Path) -> Path:
    return project_root / CORRECTED_ANNOTATION_DIR


def yolo_dataset_dir(project_root: Path) -> Path:
    return project_root / YOLO_DATASET_DIR


def is_real_warped_image(image_name: str) -> bool:
    path = Path(image_name)
    return path.name.startswith("real_") and path.name.endswith("_warped.png")


def base_real_stem(image_name: str) -> str:
    stem = Path(image_name).stem
    return stem.removesuffix("_warped")


def list_real_warped_images(project_root: Path, annotation_only: bool = True) -> list[Path]:
    root = real_warped_dir(project_root)
    if not root.exists():
        return []
    if not annotation_only:
        return sorted(root.glob("real_*_warped.png"))

    df = load_manifest(project_root)
    if df.empty:
        return sorted(root.glob("real_*_warped.png"))

    selected: list[Path] = []
    for _, row in df.iterrows():
        if str(row.get("keep_for_annotation", "true")).lower() != "true":
            continue
        if str(row.get("warp_success", "")).lower() != "true":
            continue
        warped = str(row.get("warped_image_path", "")).strip()
        if not warped:
            continue
        path = project_root / warped
        if path.exists():
            selected.append(path)
    return sorted(selected)


def load_manifest(project_root: Path) -> pd.DataFrame:
    path = real_manifest_path(project_root)
    if not path.exists():
        return pd.DataFrame()
    return ensure_manifest_curation_columns(pd.read_csv(path, dtype=str, keep_default_na=False))


def save_manifest(project_root: Path, df: pd.DataFrame) -> Path:
    path = real_manifest_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = ensure_manifest_curation_columns(df)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _string_bool(value: Any, default: str = "true") -> str:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return "true"
    if text in {"false", "0", "no", "n"}:
        return "false"
    return default


def ensure_manifest_curation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure manifest has non-destructive curation fields."""
    if df.empty:
        return df
    df = df.copy()
    defaults = {
        "keep_for_annotation": "true",
        "keep_for_training": "true",
        "duplicate_group": "",
        "exclude_reason": "",
        "quality_score": "",
        "curation_notes": "",
    }
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default
        else:
            df[column] = df[column].fillna("").astype(str)
            if column in {"keep_for_annotation", "keep_for_training"}:
                df[column] = df[column].apply(lambda value: _string_bool(value, default))

    if "warp_success" in df.columns:
        warp_failed = df["warp_success"].astype(str).str.lower().eq("false")
        df.loc[warp_failed & df["exclude_reason"].eq(""), "exclude_reason"] = "warp_failed"
        df.loc[warp_failed, "keep_for_training"] = "false"
    return df


def manifest_row_for_image(project_root: Path, image_name: str) -> dict[str, str]:
    df = load_manifest(project_root)
    if df.empty:
        return {}
    base = base_real_stem(image_name)
    candidates = {image_name, f"{base}.png", f"{base}_warped.png"}
    mask = df["image_name"].isin(candidates) if "image_name" in df.columns else pd.Series([], dtype=bool)
    if mask.any():
        return df.loc[mask].iloc[0].to_dict()
    return {}


def imported_manifest_columns() -> list[str]:
    return [
        "image_name",
        "original_file_name",
        "original_path",
        "source_type",
        "specimen_id",
        "view_id",
        "imported_path",
        "width",
        "height",
        "warped_image_path",
        "warp_success",
        "needs_review",
        "review_reason",
        "notes",
        *CURATION_COLUMNS,
    ]


def read_existing_manual_fields(project_root: Path) -> dict[str, dict[str, str]]:
    df = load_manifest(project_root)
    if df.empty:
        return {}
    fields = ["specimen_id", "view_id", "notes", *CURATION_COLUMNS]
    existing: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        key = str(row.get("original_file_name", ""))
        if not key:
            continue
        existing[key] = {field: str(row.get(field, "")) for field in fields}
    return existing


def discover_heic_files(source_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".heic", ".heif"}
    )


def check_specimen_groups(project_root: Path) -> dict[str, Any]:
    df = load_manifest(project_root)
    if df.empty:
        raise FileNotFoundError(f"Manifest not found: {real_manifest_path(project_root)}")

    df["specimen_id"] = df.get("specimen_id", "").astype(str).str.strip()
    missing_count = int((df["specimen_id"] == "").sum())
    grouped = df[df["specimen_id"] != ""].groupby("specimen_id", sort=True)
    rows: list[dict[str, Any]] = []
    for specimen_id, group in grouped:
        image_names = list(group["image_name"].astype(str))
        count = len(image_names)
        notes = ""
        if count == 1:
            notes = "single_image"
        elif count >= 6:
            notes = "many_images_check_grouping"
        rows.append(
            {
                "specimen_id": specimen_id,
                "num_images": count,
                "image_names": ";".join(image_names),
                "notes": notes,
            }
        )
    summary = pd.DataFrame(rows, columns=["specimen_id", "num_images", "image_names", "notes"])
    output_path = project_root / SPECIMEN_GROUP_SUMMARY
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")
    return {
        "missing_count": missing_count,
        "num_specimens": int(summary["specimen_id"].nunique()) if not summary.empty else 0,
        "summary_path": output_path,
        "summary": summary,
    }


def _load_camera_calibration(project_root: Path) -> tuple[np.ndarray, np.ndarray] | None:
    candidates = [
        project_root / "camera_calibration.yaml",
        project_root / "data" / "camera_calibration.yaml",
        project_root / "data" / "calibration_board" / "camera_calibration.yaml",
    ]
    for path in candidates:
        if not path.exists():
            continue
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        camera_matrix = fs.getNode("camera_matrix").mat()
        dist_coeffs = fs.getNode("dist_coeffs").mat()
        fs.release()
        if camera_matrix is not None and dist_coeffs is not None:
            return camera_matrix, dist_coeffs
    return None


def warp_real_images(project_root: Path) -> dict[str, Any]:
    df = load_manifest(project_root)
    if df.empty:
        raise FileNotFoundError(f"Manifest not found: {real_manifest_path(project_root)}")

    output_dir = real_warped_dir(project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = _load_camera_calibration(project_root)
    successes = 0
    failures = 0

    for index, row in df.iterrows():
        image_name = str(row["image_name"])
        raw_path = project_root / str(row["imported_path"])
        out_name = f"{Path(image_name).stem}_warped.png"
        out_path = output_dir / out_name
        try:
            image = load_image_file(raw_path)
            if calibration is not None:
                camera_matrix, dist_coeffs = calibration
                image = cv2.undistort(ensure_rgb(image), camera_matrix, dist_coeffs)
            detection = detect_aruco_markers(image)
            markers = detection.get("markers", {})
            board_config = select_board_config_for_markers(markers)
            warped, _transform, _info = warp_board_by_aruco(image, markers, board_config)
            Image.fromarray(ensure_rgb(warped)).save(out_path)
            df.loc[index, "warped_image_path"] = relpath(out_path, project_root)
            df.loc[index, "warp_success"] = "true"
            df.loc[index, "needs_review"] = "false"
            df.loc[index, "review_reason"] = ""
            successes += 1
        except Exception as exc:  # noqa: BLE001 - script records per-image failures.
            df.loc[index, "warped_image_path"] = ""
            df.loc[index, "warp_success"] = "false"
            df.loc[index, "needs_review"] = "true"
            df.loc[index, "review_reason"] = f"warp_failed: {exc}"
            failures += 1

    save_manifest(project_root, df)
    return {"successes": successes, "failures": failures, "manifest": real_manifest_path(project_root)}


def _format_keypoints(keypoints: Mapping[str, Mapping[str, float]]) -> dict[str, list[float]]:
    formatted: dict[str, list[float]] = {}
    for definition in KEYPOINT_DEFS:
        point = keypoints.get(definition.name)
        if point is not None:
            formatted[f"{definition.code}_{definition.name}"] = [float(point["x"]), float(point["y"])]
    return formatted


def _format_derived_points(measurements: Mapping[str, Any]) -> dict[str, list[float]]:
    formatted: dict[str, list[float]] = {}
    derived = measurements.get("derived_points", {})
    if not isinstance(derived, Mapping):
        return formatted
    for definition in DERIVED_POINT_DEFS:
        key = f"{definition.code}_{definition.name}"
        point = derived.get(key) or derived.get(definition.name)
        if isinstance(point, Mapping):
            formatted[key] = [float(point["x"]), float(point["y"])]
    return formatted


def _ordered_result_dataframe(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    for column in RESULT_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df.loc[:, list(RESULT_COLUMNS)]


def _upsert_measurement_row(
    row: Mapping[str, Any],
    annotation_root: Path,
    csv_name: str,
    xlsx_name: str,
) -> dict[str, Path]:
    annotation_root.mkdir(parents=True, exist_ok=True)
    csv_path = annotation_root / csv_name
    xlsx_path = annotation_root / xlsx_name
    if csv_path.exists():
        existing = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    else:
        existing = pd.DataFrame(columns=list(RESULT_COLUMNS))
    for column in RESULT_COLUMNS:
        if column not in existing.columns:
            existing[column] = ""
    image_name = str(row.get("image_name", ""))
    if not existing.empty:
        existing = existing.loc[existing["image_name"].astype(str) != image_name, list(RESULT_COLUMNS)]
    new_row = _ordered_result_dataframe([row])
    combined = new_row if existing.empty else pd.concat([existing, new_row], ignore_index=True)
    combined.to_csv(csv_path, index=False, encoding="utf-8-sig")
    combined.to_excel(xlsx_path, index=False)
    return {"csv": csv_path, "xlsx": xlsx_path}


def upsert_real_measurement_row(row: Mapping[str, Any], annotation_root: Path) -> dict[str, Path]:
    return _upsert_measurement_row(row, annotation_root, REAL_MEASUREMENTS_CSV, REAL_MEASUREMENTS_XLSX)


def upsert_corrected_measurement_row(row: Mapping[str, Any], annotation_root: Path) -> dict[str, Path]:
    return _upsert_measurement_row(row, annotation_root, CORRECTED_MEASUREMENTS_CSV, CORRECTED_MEASUREMENTS_XLSX)


def _xy_from_payload(value: Any) -> tuple[float, float] | None:
    if isinstance(value, Mapping) and "x" in value and "y" in value:
        return float(value["x"]), float(value["y"])
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return None


def upsert_correction_summary(
    annotation_root: Path,
    image_name: str,
    specimen_id: str,
    annotation_metadata: Mapping[str, Any],
    mm_per_pixel: float,
) -> Path:
    """Write per-keypoint raw-to-corrected movement rows for later review."""
    path = annotation_root / REAL_CORRECTION_SUMMARY_CSV
    columns = [
        "image_name",
        "specimen_id",
        "keypoint_name",
        "model_x",
        "model_y",
        "corrected_x",
        "corrected_y",
        "movement_px",
        "movement_mm",
        "has_mask_suggestion",
        "suggestion_x",
        "suggestion_y",
        "corrected_to_suggestion_px",
        "qc_flag",
        "preannotation_mode",
        "hybrid_model_version",
        "num_points_from_heatmap",
        "num_points_from_v034",
        "num_points_from_v01",
        "num_points_from_v05",
        "num_points_from_v041",
        "num_points_from_mask_rule",
        "num_points_manually_moved",
        "moved_keypoints",
        "mean_manual_displacement_mm",
        "max_manual_displacement_mm",
    ]
    if path.exists():
        existing = pd.read_csv(path, dtype=str, keep_default_na=False)
    else:
        existing = pd.DataFrame(columns=columns)
    for column in columns:
        if column not in existing.columns:
            existing[column] = ""
    existing = existing.loc[existing["image_name"].astype(str) != image_name, columns]

    raw_points = annotation_metadata.get("model_keypoints_raw", {})
    corrected_points = annotation_metadata.get("corrected_keypoints", {})
    suggestions = annotation_metadata.get("mask_suggestions", {})
    qc = annotation_metadata.get("qc_results", {})
    flags = qc.get("flagged_keypoints", {}) if isinstance(qc, Mapping) else {}
    if not isinstance(raw_points, Mapping) or not isinstance(corrected_points, Mapping):
        return path
    if not isinstance(suggestions, Mapping):
        suggestions = {}
    if not isinstance(flags, Mapping):
        flags = {}

    rows: list[dict[str, Any]] = []
    for definition in KEYPOINT_DEFS:
        key = f"{definition.code}_{definition.name}"
        raw = _xy_from_payload(raw_points.get(key))
        corrected = _xy_from_payload(corrected_points.get(key))
        suggestion = _xy_from_payload(suggestions.get(key))
        if raw is None or corrected is None:
            continue
        movement = math.hypot(corrected[0] - raw[0], corrected[1] - raw[1])
        if suggestion is not None:
            suggestion_distance: Any = math.hypot(corrected[0] - suggestion[0], corrected[1] - suggestion[1])
            sx: Any = round(suggestion[0], 3)
            sy: Any = round(suggestion[1], 3)
            has_suggestion = True
        else:
            suggestion_distance = ""
            sx = ""
            sy = ""
            has_suggestion = False
        rows.append(
            {
                "image_name": image_name,
                "specimen_id": specimen_id,
                "keypoint_name": key,
                "model_x": round(raw[0], 3),
                "model_y": round(raw[1], 3),
                "corrected_x": round(corrected[0], 3),
                "corrected_y": round(corrected[1], 3),
                "movement_px": round(movement, 3),
                "movement_mm": round(movement * float(mm_per_pixel), 3),
                "has_mask_suggestion": has_suggestion,
                "suggestion_x": sx,
                "suggestion_y": sy,
                "corrected_to_suggestion_px": round(float(suggestion_distance), 3) if suggestion_distance != "" else "",
                "qc_flag": flags.get(key, ""),
                "preannotation_mode": annotation_metadata.get("preannotation_mode", ""),
                "hybrid_model_version": annotation_metadata.get("hybrid_model_version", ""),
                "num_points_from_heatmap": annotation_metadata.get("num_points_from_heatmap", ""),
                "num_points_from_v034": annotation_metadata.get("num_points_from_v034", ""),
                "num_points_from_v01": annotation_metadata.get("num_points_from_v01", ""),
                "num_points_from_v05": annotation_metadata.get("num_points_from_v05", ""),
                "num_points_from_v041": annotation_metadata.get("num_points_from_v041", ""),
                "num_points_from_mask_rule": annotation_metadata.get("num_points_from_mask_rule", ""),
                "num_points_manually_moved": annotation_metadata.get("num_points_manually_moved", ""),
                "moved_keypoints": annotation_metadata.get("moved_keypoints", ""),
                "mean_manual_displacement_mm": annotation_metadata.get("mean_manual_displacement_mm", ""),
                "max_manual_displacement_mm": annotation_metadata.get("max_manual_displacement_mm", ""),
            }
        )
    combined = pd.concat([existing, pd.DataFrame(rows, columns=columns)], ignore_index=True) if rows else existing
    combined.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_real_annotation_artifacts(
    project_root: Path,
    image_name: str,
    annotated_image: np.ndarray,
    keypoints: Mapping[str, Mapping[str, float]],
    measurements: Mapping[str, Any],
    result_row: Mapping[str, Any],
    specimen_id: str,
    annotator: str,
    notes: str,
    axis_mode_request: str,
    scale_method: str,
    mm_per_pixel: float,
    annotation_mode: str = "manual",
    annotation_metadata: Mapping[str, Any] | None = None,
    review_reason: str = "",
) -> dict[str, Path]:
    annotation_root = real_annotation_dir(project_root)
    keypoint_dir = annotation_root / "keypoints"
    annotation_dir = annotation_root / "annotations"
    keypoint_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    base = sanitize_filename_stem(base_real_stem(image_name))
    annotated_path = annotation_dir / f"{base}_annotated.png"
    Image.fromarray(ensure_rgb(annotated_image)).save(annotated_path)
    annotation_metadata = dict(annotation_metadata or {})
    payload = {
        "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
        "board_version": BOARD_VERSION,
        "image_name": image_name,
        "source_type": "real",
        "specimen_id": specimen_id,
        "annotator": annotator,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "annotation_mode": annotation_mode,
        "preannotation_mode": annotation_metadata.get("preannotation_mode", ""),
        "hybrid_model_version": annotation_metadata.get("hybrid_model_version", ""),
        "axis_mode_selected": measurements.get("axis_mode_selected", ""),
        "axis_mode_request": axis_mode_request,
        "scale_method": scale_method,
        "mm_per_pixel": float(mm_per_pixel),
        "keypoints": _format_keypoints(keypoints),
        "derived_points": _format_derived_points(measurements),
        "body_axis_points_order": list(BODY_AXIS_POINT_KEYS),
        "caudal_tip_candidates": list(CAUDAL_TIP_CANDIDATE_KEYS),
        "caudal_tip_selected": measurements.get("caudal_tip_selected", ""),
        "needs_review": bool(result_row.get("needs_review", False)),
        "review_reason": review_reason,
        "notes": notes,
        "measurements": dict(measurements),
        "preannotation": annotation_metadata,
        "model_keypoints_raw": annotation_metadata.get("model_keypoints_raw", {}),
        "heatmap_keypoints": annotation_metadata.get("heatmap_keypoints", {}),
        "heatmap_confidences": annotation_metadata.get("heatmap_confidences", {}),
        "heatmap_debug_info": annotation_metadata.get("heatmap_debug_info", {}),
        "v034_keypoints": annotation_metadata.get("v034_keypoints", {}),
        "mask_suggestions": annotation_metadata.get("mask_suggestions", {}),
        "geometric_suggestions": annotation_metadata.get("geometric_suggestions", {}),
        "local_normal_suggestions": annotation_metadata.get("local_normal_suggestions", {}),
        "hybrid_keypoints": annotation_metadata.get("hybrid_keypoints", {}),
        "v06_keypoints": annotation_metadata.get("v06_keypoints", {}),
        "v06_point_sources": annotation_metadata.get("v06_point_sources", {}),
        "v06_qc_results": annotation_metadata.get("v06_qc_results", {}),
        "v01_heatmap_keypoints": annotation_metadata.get("v01_heatmap_keypoints", {}),
        "v01_heatmap_confidences": annotation_metadata.get("v01_heatmap_confidences", {}),
        "v05_heatmap_keypoints": annotation_metadata.get("v05_heatmap_keypoints", {}),
        "v05_heatmap_confidences": annotation_metadata.get("v05_heatmap_confidences", {}),
        "v041_hybrid_keypoints": annotation_metadata.get("v041_hybrid_keypoints", {}),
        "point_sources": annotation_metadata.get("point_sources", {}),
        "hybrid_qc_results": annotation_metadata.get("hybrid_qc_results", {}),
        "local_normal_measurements": annotation_metadata.get("local_normal_measurements", {}),
        "curvature_qc": annotation_metadata.get("curvature_qc", {}),
        "measurement_axes": annotation_metadata.get("measurement_axes", {}),
        "tail_geometry_qc": annotation_metadata.get("tail_geometry_qc", {}),
        "tail_qc": annotation_metadata.get("tail_qc", {}),
        "body_depth_geometry": annotation_metadata.get("body_depth_geometry", {}),
        "body_depth_qc": annotation_metadata.get("body_depth_qc", {}),
        "peduncle_depth_qc": annotation_metadata.get("peduncle_depth_qc", {}),
        "axis_qc": annotation_metadata.get("axis_qc", {}),
        "corrected_keypoints": annotation_metadata.get("corrected_keypoints", _format_keypoints(keypoints)),
        "keypoint_edit_log": annotation_metadata.get("keypoint_edit_log", []),
        "qc_results": annotation_metadata.get("qc_results", {}),
    }
    for key in (
        "segmentation_success",
        "segmentation_quality",
        "fish_bbox_x1",
        "fish_bbox_y1",
        "fish_bbox_x2",
        "fish_bbox_y2",
        "mask_area_px",
        "p1_source",
        "p1_model_x",
        "p1_model_y",
        "p1_mask_x",
        "p1_mask_y",
        "p1_correction_applied",
        "keypoint_correction_log",
        "P7U_source",
        "P7L_source",
        "P6_source",
        "P6_mask_x",
        "P6_mask_y",
        "P6_mask_quality",
        "P6_gap_x",
        "P6_gap_y",
        "P6_gap_quality",
        "P6_concavity_x",
        "P6_concavity_y",
        "P6_concavity_quality",
        "P6_final_x",
        "P6_final_y",
        "P6_review_reason",
        "P6_geometry_qc_pass",
        "P6_needs_review",
        "P6_fallback_used",
        "P6_geometry_review_reason",
        "P6_distance_to_P7U_ok",
        "P6_distance_to_P7L_ok",
        "P6_tail_leaf_balance_ok",
        "tail_detail_mask_quality",
        "tail_mask_gap_visible",
        "tail_mask_gap_filled",
        "P7U_mask_x",
        "P7U_mask_y",
        "P7L_mask_x",
        "P7L_mask_y",
        "tail_mask_quality",
        "tail_axis_source",
        "tail_axis_used_without_P6",
        "tail_axis_used_without_model_P6",
        "tail_axis_dx",
        "tail_axis_dy",
        "tail_region_bbox",
        "P7U_candidate_count",
        "P7U_candidate_method",
        "P7U_mask_quality",
        "P7U_review_reason",
        "p7v_valid",
        "p7v_invalid_reason",
        "P2_eye_x",
        "P2_eye_y",
        "P2_eye_quality",
        "P2_eye_bbox",
        "P2_eye_candidate_count",
        "P2_source_suggestion",
        "P2_model_to_eye_distance_mm",
        "P3_edge_x",
        "P3_edge_y",
        "P3_edge_quality",
        "P3_edge_strength",
        "P3_edge_method",
        "P3_model_to_edge_distance_mm",
        "P5_transition_x",
        "P5_transition_y",
        "P5_transition_quality",
        "P5_transition_method",
        "P5_model_to_transition_distance_mm",
        "tail_width_profile",
        "tail_width_transition_score",
        "local_structure_qc_pass",
        "local_structure_review_reason",
        "axis_centerline_quality",
        "USE_MASK_AXIS_POINTS",
        "C1_mask_x",
        "C1_mask_y",
        "C1_mask_quality",
        "C1_source",
        "C2_mask_x",
        "C2_mask_y",
        "C2_mask_quality",
        "C2_source",
        "C3_mask_x",
        "C3_mask_y",
        "C3_mask_quality",
        "C3_source",
        "C4_mask_x",
        "C4_mask_y",
        "C4_mask_quality",
        "C4_source",
        "body_depth_refined_mm",
        "body_depth_original_mm",
        "body_depth_refined_minus_original_mm",
        "body_depth_geometric_mm",
        "body_depth_geometric_qc_pass",
        "body_depth_geometric_review_reason",
        "caudal_peduncle_depth_refined_mm",
        "caudal_peduncle_depth_original_mm",
        "caudal_peduncle_depth_refined_minus_original_mm",
        "curvature_qc_level",
        "max_axis_deviation_mm",
        "axis_bend_angle_deg",
        "curvature_high_review_required",
        "measurements_needs_review",
        "curvature_review_points",
    ):
        if key in annotation_metadata:
            payload[key] = annotation_metadata[key]
    json_path = keypoint_dir / f"{base}_keypoints.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    measurement_paths = upsert_real_measurement_row(result_row, annotation_root)
    correction_summary_path = upsert_correction_summary(
        annotation_root,
        image_name,
        specimen_id,
        annotation_metadata,
        mm_per_pixel,
    )
    summary_path = write_real_annotation_summary(project_root)
    return {
        "keypoints": json_path,
        "annotations": annotated_path,
        "csv": measurement_paths["csv"],
        "xlsx": measurement_paths["xlsx"],
        "correction_summary": correction_summary_path,
        "summary": summary_path,
    }


def save_corrected_annotation_artifacts(
    project_root: Path,
    image_name: str,
    annotated_image: np.ndarray,
    keypoints: Mapping[str, Mapping[str, float]],
    measurements: Mapping[str, Any],
    result_row: Mapping[str, Any],
    specimen_id: str,
    annotator: str,
    notes: str,
    axis_mode_request: str,
    scale_method: str,
    mm_per_pixel: float,
    annotation_metadata: Mapping[str, Any],
    review_reason: str = "",
) -> dict[str, Path]:
    """Save user-confirmed auto-assisted corrected labels for v0.3 training."""
    annotation_root = corrected_annotation_dir(project_root)
    keypoint_dir = annotation_root / "keypoints"
    annotation_dir = annotation_root / "annotations"
    keypoint_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    base = sanitize_filename_stem(base_real_stem(image_name))
    annotated_path = annotation_dir / f"{base}_annotated.png"
    Image.fromarray(ensure_rgb(annotated_image)).save(annotated_path)

    annotation_metadata = dict(annotation_metadata or {})
    annotation_metadata["annotation_status"] = "corrected_and_confirmed"
    annotation_mode = str(annotation_metadata.get("annotation_mode") or result_row.get("annotation_mode") or "auto_assisted")
    payload = {
        "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
        "board_version": BOARD_VERSION,
        "image_name": image_name,
        "source_type": "real",
        "specimen_id": specimen_id,
        "annotator": annotator,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "annotation_status": "corrected_and_confirmed",
        "annotation_mode": annotation_mode,
        "preannotation_mode": annotation_metadata.get("preannotation_mode", ""),
        "hybrid_model_version": annotation_metadata.get("hybrid_model_version", ""),
        "migrated_from_previous_annotation": bool(annotation_metadata.get("migrated_from_previous_annotation", False)),
        "source_annotation_path": str(annotation_metadata.get("source_annotation_path", "")),
        "axis_mode_selected": measurements.get("axis_mode_selected", ""),
        "axis_mode_request": axis_mode_request,
        "scale_method": scale_method,
        "mm_per_pixel": float(mm_per_pixel),
        "keypoints": _format_keypoints(keypoints),
        "model_keypoints_raw": annotation_metadata.get("model_keypoints_raw", {}),
        "heatmap_keypoints": annotation_metadata.get("heatmap_keypoints", {}),
        "heatmap_confidences": annotation_metadata.get("heatmap_confidences", {}),
        "heatmap_debug_info": annotation_metadata.get("heatmap_debug_info", {}),
        "v034_keypoints": annotation_metadata.get("v034_keypoints", {}),
        "mask_suggestions": annotation_metadata.get("mask_suggestions", {}),
        "geometric_suggestions": annotation_metadata.get("geometric_suggestions", {}),
        "local_normal_suggestions": annotation_metadata.get("local_normal_suggestions", {}),
        "hybrid_keypoints": annotation_metadata.get("hybrid_keypoints", {}),
        "point_sources": annotation_metadata.get("point_sources", {}),
        "hybrid_qc_results": annotation_metadata.get("hybrid_qc_results", {}),
        "local_normal_measurements": annotation_metadata.get("local_normal_measurements", {}),
        "curvature_qc": annotation_metadata.get("curvature_qc", {}),
        "measurement_axes": annotation_metadata.get("measurement_axes", {}),
        "tail_geometry_qc": annotation_metadata.get("tail_geometry_qc", {}),
        "tail_qc": annotation_metadata.get("tail_qc", {}),
        "body_depth_geometry": annotation_metadata.get("body_depth_geometry", {}),
        "body_depth_qc": annotation_metadata.get("body_depth_qc", {}),
        "peduncle_depth_qc": annotation_metadata.get("peduncle_depth_qc", {}),
        "axis_qc": annotation_metadata.get("axis_qc", {}),
        "corrected_keypoints": annotation_metadata.get("corrected_keypoints", _format_keypoints(keypoints)),
        "derived_points": _format_derived_points(measurements),
        "keypoint_edit_log": annotation_metadata.get("keypoint_edit_log", []),
        "qc_results": annotation_metadata.get("qc_results", {}),
        "body_axis_points_order": list(BODY_AXIS_POINT_KEYS),
        "caudal_tip_candidates": list(CAUDAL_TIP_CANDIDATE_KEYS),
        "caudal_tip_selected": measurements.get("caudal_tip_selected", ""),
        "needs_review": bool(result_row.get("needs_review", False)),
        "review_reason": review_reason,
        "notes": notes,
        "measurements": dict(measurements),
        "preannotation": annotation_metadata,
    }
    for key in (
        "segmentation_success",
        "segmentation_quality",
        "fish_bbox_x1",
        "fish_bbox_y1",
        "fish_bbox_x2",
        "fish_bbox_y2",
        "mask_area_px",
        "p1_source",
        "p1_model_x",
        "p1_model_y",
        "p1_mask_x",
        "p1_mask_y",
        "p1_correction_applied",
        "keypoint_correction_log",
        "P7U_source",
        "P7L_source",
        "P6_source",
        "P6_mask_x",
        "P6_mask_y",
        "P6_mask_quality",
        "P6_gap_x",
        "P6_gap_y",
        "P6_gap_quality",
        "P6_concavity_x",
        "P6_concavity_y",
        "P6_concavity_quality",
        "P6_final_x",
        "P6_final_y",
        "P6_review_reason",
        "P6_geometry_qc_pass",
        "P6_needs_review",
        "P6_fallback_used",
        "P6_geometry_review_reason",
        "P6_distance_to_P7U_ok",
        "P6_distance_to_P7L_ok",
        "P6_tail_leaf_balance_ok",
        "tail_detail_mask_quality",
        "tail_mask_gap_visible",
        "tail_mask_gap_filled",
        "P7U_mask_x",
        "P7U_mask_y",
        "P7L_mask_x",
        "P7L_mask_y",
        "tail_mask_quality",
        "tail_axis_source",
        "tail_axis_used_without_P6",
        "tail_axis_used_without_model_P6",
        "tail_axis_dx",
        "tail_axis_dy",
        "tail_region_bbox",
        "P7U_candidate_count",
        "P7U_candidate_method",
        "P7U_mask_quality",
        "P7U_review_reason",
        "p7v_valid",
        "p7v_invalid_reason",
        "P2_eye_x",
        "P2_eye_y",
        "P2_eye_quality",
        "P2_eye_bbox",
        "P2_eye_candidate_count",
        "P2_source_suggestion",
        "P2_model_to_eye_distance_mm",
        "P3_edge_x",
        "P3_edge_y",
        "P3_edge_quality",
        "P3_edge_strength",
        "P3_edge_method",
        "P3_model_to_edge_distance_mm",
        "P5_transition_x",
        "P5_transition_y",
        "P5_transition_quality",
        "P5_transition_method",
        "P5_model_to_transition_distance_mm",
        "tail_width_profile",
        "tail_width_transition_score",
        "local_structure_qc_pass",
        "local_structure_review_reason",
        "axis_centerline_quality",
        "USE_MASK_AXIS_POINTS",
        "C1_mask_x",
        "C1_mask_y",
        "C1_mask_quality",
        "C1_source",
        "C2_mask_x",
        "C2_mask_y",
        "C2_mask_quality",
        "C2_source",
        "C3_mask_x",
        "C3_mask_y",
        "C3_mask_quality",
        "C3_source",
        "C4_mask_x",
        "C4_mask_y",
        "C4_mask_quality",
        "C4_source",
        "body_depth_refined_mm",
        "body_depth_original_mm",
        "body_depth_refined_minus_original_mm",
        "body_depth_geometric_mm",
        "body_depth_geometric_qc_pass",
        "body_depth_geometric_review_reason",
        "caudal_peduncle_depth_refined_mm",
        "caudal_peduncle_depth_original_mm",
        "caudal_peduncle_depth_refined_minus_original_mm",
        "curvature_qc_level",
        "max_axis_deviation_mm",
        "axis_bend_angle_deg",
        "curvature_high_review_required",
        "measurements_needs_review",
        "curvature_review_points",
    ):
        if key in annotation_metadata:
            payload[key] = annotation_metadata[key]
    json_path = keypoint_dir / f"{base}_keypoints.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    measurement_paths = upsert_corrected_measurement_row(result_row, annotation_root)
    correction_summary_path = upsert_correction_summary(
        annotation_root,
        image_name,
        specimen_id,
        annotation_metadata,
        mm_per_pixel,
    )
    return {
        "keypoints": json_path,
        "annotations": annotated_path,
        "csv": measurement_paths["csv"],
        "xlsx": measurement_paths["xlsx"],
        "correction_summary": correction_summary_path,
    }


def write_real_annotation_summary(project_root: Path) -> Path:
    df = load_manifest(project_root)
    annotation_root = real_annotation_dir(project_root)
    annotation_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        image_name = str(row.get("image_name", ""))
        base = sanitize_filename_stem(Path(image_name).stem)
        json_path = annotation_root / "keypoints" / f"{base}_keypoints.json"
        annotated_path = annotation_root / "annotations" / f"{base}_annotated.png"
        payload = load_keypoints_json(json_path) if json_path.exists() else {}
        keypoints = payload.get("keypoints", {}) if isinstance(payload, Mapping) else {}
        derived = payload.get("derived_points", {}) if isinstance(payload, Mapping) else {}
        rows.append(
            {
                "image_name": image_name,
                "specimen_id": row.get("specimen_id", ""),
                "source_type": "real",
                "has_keypoints_json": json_path.exists(),
                "manual_keypoint_count": len(keypoints) if isinstance(keypoints, Mapping) else 0,
                "has_all_16_keypoints": len(keypoints) == len(KEYPOINT_DEFS),
                "has_p7v": isinstance(derived, Mapping) and "P7V_caudal_fin_posterior_endpoint" in derived,
                "annotated_exists": annotated_path.exists(),
                "needs_review": payload.get("needs_review", "") if isinstance(payload, Mapping) else "",
                "review_reason": payload.get("review_reason", "") if isinstance(payload, Mapping) else "",
            }
        )
    output = annotation_root / REAL_ANNOTATION_SUMMARY_CSV
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    return output


def check_real_annotation_readiness(project_root: Path) -> pd.DataFrame:
    df = load_manifest(project_root)
    annotation_root = real_annotation_dir(project_root)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        image_name = str(row.get("image_name", ""))
        specimen_id = str(row.get("specimen_id", "")).strip()
        base = sanitize_filename_stem(Path(image_name).stem)
        json_path = annotation_root / "keypoints" / f"{base}_keypoints.json"
        payload = load_keypoints_json(json_path) if json_path.exists() else {}
        keypoints = payload.get("keypoints", {}) if isinstance(payload, Mapping) else {}
        derived = payload.get("derived_points", {}) if isinstance(payload, Mapping) else {}
        source_type = str(payload.get("source_type", row.get("source_type", ""))) if isinstance(payload, Mapping) else str(row.get("source_type", ""))
        has_all_16 = isinstance(keypoints, Mapping) and len(keypoints) == len(KEYPOINT_DEFS)
        has_p7v = isinstance(derived, Mapping) and "P7V_caudal_fin_posterior_endpoint" in derived
        warp_success = str(row.get("warp_success", "")).lower() == "true"
        needs_review = str(row.get("needs_review", "")).lower() == "true" or bool(payload.get("needs_review", False))
        review_reason = str(row.get("review_reason", "") or payload.get("review_reason", ""))
        keep_for_training = str(row.get("keep_for_training", "true")).lower() == "true"
        ready = (
            bool(specimen_id)
            and keep_for_training
            and warp_success
            and json_path.exists()
            and has_all_16
            and has_p7v
            and source_type == "real"
            and not needs_review
        )
        rows.append(
            {
                "image_name": image_name,
                "specimen_id": specimen_id,
                "keep_for_training": keep_for_training,
                "has_specimen_id": bool(specimen_id),
                "warp_success": warp_success,
                "has_keypoints_json": json_path.exists(),
                "has_all_16_keypoints": has_all_16,
                "has_p7v": has_p7v,
                "needs_review": needs_review,
                "review_reason": review_reason,
                "ready_for_training": ready,
            }
        )
    report = pd.DataFrame(rows)
    if not report.empty:
        report = report[
            [
                "image_name",
                "specimen_id",
                "keep_for_training",
                "has_specimen_id",
                "warp_success",
                "has_keypoints_json",
                "has_all_16_keypoints",
                "has_p7v",
                "needs_review",
                "review_reason",
                "ready_for_training",
            ]
        ]
    output = annotation_root / READINESS_REPORT_CSV
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False, encoding="utf-8-sig")
    return report


def _load_annotation_payload(project_root: Path, image_name: str) -> dict[str, Any]:
    base = sanitize_filename_stem(Path(image_name).stem)
    path = real_annotation_dir(project_root) / "keypoints" / f"{base}_keypoints.json"
    return load_keypoints_json(path)


def _bbox_from_keypoints(points: list[tuple[float, float]], width: int, height: int, padding: float = 0.05) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x0, x1 = max(0.0, min(xs)), min(float(width), max(xs))
    y0, y1 = max(0.0, min(ys)), min(float(height), max(ys))
    pad_x = (x1 - x0) * padding
    pad_y = (y1 - y0) * padding
    x0 = max(0.0, x0 - pad_x)
    x1 = min(float(width), x1 + pad_x)
    y0 = max(0.0, y0 - pad_y)
    y1 = min(float(height), y1 + pad_y)
    return ((x0 + x1) / 2 / width, (y0 + y1) / 2 / height, (x1 - x0) / width, (y1 - y0) / height)


def _fish_roi_crop_box(
    width: int,
    height: int,
    padding_mm: float = 12.0,
    px_per_mm: float | None = None,
) -> tuple[int, int, int, int]:
    """Return the fixed fish-placement ROI in warped-image pixel coordinates."""
    px_per_mm = px_per_mm or float(A3_V2_CHARUCO_PLUMB_CONFIG["output_px_per_mm"])
    fish_box = A3_V2_CHARUCO_PLUMB_CONFIG["plumb_lines"]["fish_box_mm"]
    x0_mm, y0_mm, x1_mm, y1_mm = [float(value) for value in fish_box]
    x0 = int(round((x0_mm - padding_mm) * px_per_mm))
    y0 = int(round((y0_mm - padding_mm) * px_per_mm))
    x1 = int(round((x1_mm + padding_mm) * px_per_mm))
    y1 = int(round((y1_mm + padding_mm) * px_per_mm))
    return (
        max(0, min(width - 1, x0)),
        max(0, min(height - 1, y0)),
        max(1, min(width, x1)),
        max(1, min(height, y1)),
    )


def _expand_crop_to_include_points(
    crop_box: tuple[int, int, int, int],
    points: list[tuple[float, float]],
    width: int,
    height: int,
    margin_px: int = 20,
) -> tuple[int, int, int, int]:
    """Keep the ROI mostly fixed, but avoid dropping a valid keypoint at the edge."""
    x0, y0, x1, y1 = crop_box
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    return (
        max(0, min(x0, int(math.floor(min_x)) - margin_px)),
        max(0, min(y0, int(math.floor(min_y)) - margin_px)),
        min(width, max(x1, int(math.ceil(max_x)) + margin_px)),
        min(height, max(y1, int(math.ceil(max_y)) + margin_px)),
    )


def export_yolopose_dataset(
    project_root: Path,
    train_count: int = 25,
    val_count: int = 5,
    test_count: int = 5,
    dataset_root: Path | None = None,
    crop_to_fish_roi: bool = False,
    fish_roi_padding_mm: float = 12.0,
) -> dict[str, Any]:
    readiness = check_real_annotation_readiness(project_root)
    ready = readiness[readiness["ready_for_training"] == True].copy()  # noqa: E712
    if ready.empty:
        raise RuntimeError("No ready real annotations found. Fill specimen_id, warp images, and annotate 16 keypoints first.")

    specimen_ids = sorted(ready["specimen_id"].unique().tolist())
    random.Random(RANDOM_SEED).shuffle(specimen_ids)
    train_ids = set(specimen_ids[:train_count])
    val_ids = set(specimen_ids[train_count : train_count + val_count])
    test_ids = set(specimen_ids[train_count + val_count : train_count + val_count + test_count])
    remaining = set(specimen_ids) - train_ids - val_ids - test_ids
    train_ids.update(remaining)

    dataset_root = dataset_root or yolo_dataset_dir(project_root)
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root
    if dataset_root.exists():
        backup_root = dataset_root.with_name(f"{dataset_root.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        shutil.move(str(dataset_root), str(backup_root))
    for split in ("train", "val", "test"):
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    split_rows: list[dict[str, Any]] = []
    key_order = [f"{definition.code}_{definition.name}" for definition in KEYPOINT_DEFS]
    manifest = load_manifest(project_root)
    manifest_by_image = {str(row["image_name"]): row for _, row in manifest.iterrows()}

    for _, row in ready.iterrows():
        image_name = str(row["image_name"])
        specimen_id = str(row["specimen_id"])
        split = "train" if specimen_id in train_ids else "val" if specimen_id in val_ids else "test"
        manifest_row = manifest_by_image[image_name]
        image_path = project_root / str(manifest_row["warped_image_path"])
        payload = _load_annotation_payload(project_root, image_name)
        keypoints = payload["keypoints"]
        with Image.open(image_path) as image:
            width, height = image.size
            rgb_image = image.convert("RGB")
        points = [(float(keypoints[key][0]), float(keypoints[key][1])) for key in key_order]
        crop_box = (0, 0, width, height)
        if crop_to_fish_roi:
            crop_box = _fish_roi_crop_box(width, height, padding_mm=fish_roi_padding_mm)
            crop_box = _expand_crop_to_include_points(crop_box, points, width, height)
        crop_x0, crop_y0, crop_x1, crop_y1 = crop_box
        crop_width = crop_x1 - crop_x0
        crop_height = crop_y1 - crop_y0
        if crop_width <= 0 or crop_height <= 0:
            raise RuntimeError(f"Invalid crop box for {image_name}: {crop_box}")
        crop_points = [(x - crop_x0, y - crop_y0) for x, y in points]
        if any(x < 0 or y < 0 or x > crop_width or y > crop_height for x, y in crop_points):
            raise RuntimeError(f"Keypoints fall outside crop box for {image_name}: {crop_box}")

        bbox = _bbox_from_keypoints(crop_points, crop_width, crop_height)
        yolo_values = [str(CLASS_ID), *(f"{value:.8f}" for value in bbox)]
        for x, y in crop_points:
            yolo_values.extend([f"{x / crop_width:.8f}", f"{y / crop_height:.8f}", "2"])
        out_name = image_path.name
        if crop_to_fish_roi:
            out_name = f"{image_path.stem}_fish_roi.png"
        out_image = dataset_root / "images" / split / out_name
        out_label = dataset_root / "labels" / split / f"{Path(out_name).stem}.txt"
        if crop_to_fish_roi:
            rgb_image.crop(crop_box).save(out_image)
        else:
            shutil.copy2(image_path, out_image)
        out_label.write_text(" ".join(yolo_values) + "\n", encoding="utf-8")
        split_rows.append(
            {
                "image_name": image_name,
                "specimen_id": specimen_id,
                "split": split,
                "source_type": "real",
                "warped_image_path": relpath(out_image, project_root),
                "label_path": relpath(out_label, project_root),
                "crop_to_fish_roi": crop_to_fish_roi,
                "crop_x0": crop_x0,
                "crop_y0": crop_y0,
                "crop_x1": crop_x1,
                "crop_y1": crop_y1,
            }
        )

    split_df = pd.DataFrame(split_rows)
    split_df.to_csv(dataset_root / "split_manifest.csv", index=False, encoding="utf-8-sig")
    summary_rows = []
    for split in ("train", "val", "test"):
        group = split_df[split_df["split"] == split]
        ids = sorted(group["specimen_id"].unique().tolist())
        summary_rows.append({"split": split, "num_specimens": len(ids), "num_images": len(group), "specimen_ids": ";".join(ids)})
    pd.DataFrame(summary_rows).to_csv(dataset_root / "dataset_summary.csv", index=False, encoding="utf-8-sig")

    schema = {
        "class_id": CLASS_ID,
        "class_name": CLASS_NAME,
        "kpt_shape": [len(KEYPOINT_DEFS), 3],
        "keypoints": [{"index": index, "code": definition.code, "name": definition.name} for index, definition in enumerate(KEYPOINT_DEFS)],
        "excluded_derived_points": [f"{definition.code}_{definition.name}" for definition in DERIVED_POINT_DEFS],
        "crop_to_fish_roi": crop_to_fish_roi,
        "fish_roi_padding_mm": fish_roi_padding_mm if crop_to_fish_roi else None,
    }
    (dataset_root / "keypoint_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    data_yaml = f"""path: {relpath(dataset_root, project_root)}
train: images/train
val: images/val
test: images/test

names:
  0: {CLASS_NAME}

kpt_shape: [{len(KEYPOINT_DEFS)}, 3]

flip_idx:
{chr(10).join(f"  - {i}" for i in range(len(KEYPOINT_DEFS)))}
"""
    (dataset_root / "data.yaml").write_text(data_yaml, encoding="utf-8")
    return {
        "dataset_root": dataset_root,
        "num_images": len(split_df),
        "num_specimens": len(specimen_ids),
        "crop_to_fish_roi": crop_to_fish_roi,
    }
