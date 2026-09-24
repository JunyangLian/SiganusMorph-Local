"""Utilities for the v0.4.1 real-world review workflow."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .config import KEYPOINT_DEFS
from .image_utils import ensure_rgb, load_image_file
from .measurements import calculate_measurements
from .dual_axis_measurement import (
    MEASUREMENT_AXIS_MODEL,
    compute_measurement_axis_metadata,
)
from .local_normal_measurement import estimate_peduncle_depth_by_axis_normals
from .segmentation import save_mask_png
from .visualization import draw_enhanced_preannotation_overlay, draw_keypoints_and_measurements


REVIEW_VERSION = "v0.4.1_realworld_review_workflow"
REVIEW_DIRNAME = "realworld_review_v0.4.1"
REVIEW_STATUS_VALUES = (
    "pending_review",
    "in_progress",
    "corrected_and_confirmed",
    "excluded",
    "needs_retake",
    "needs_later_review",
)

KEYPOINT_KEYS = tuple(f"{definition.code}_{definition.name}" for definition in KEYPOINT_DEFS)
FULL_TO_SHORT = {f"{definition.code}_{definition.name}": definition.name for definition in KEYPOINT_DEFS}
SHORT_TO_FULL = {definition.name: f"{definition.code}_{definition.name}" for definition in KEYPOINT_DEFS}


def review_root(project_root: Path, review_dirname: str = REVIEW_DIRNAME) -> Path:
    return project_root / "results" / review_dirname


def preannotation_dir(project_root: Path, review_dirname: str = REVIEW_DIRNAME) -> Path:
    return review_root(project_root, review_dirname) / "preannotations"


def corrected_dir(project_root: Path, review_dirname: str = REVIEW_DIRNAME) -> Path:
    return review_root(project_root, review_dirname) / "corrected_keypoints"


def corrected_preview_dir(project_root: Path, review_dirname: str = REVIEW_DIRNAME) -> Path:
    return review_root(project_root, review_dirname) / "corrected_previews"


def manifest_path(project_root: Path, review_dirname: str = REVIEW_DIRNAME) -> Path:
    return review_root(project_root, review_dirname) / "review_manifest.csv"


def sample_manifest_path(project_root: Path, review_dirname: str = REVIEW_DIRNAME) -> Path:
    return review_root(project_root, review_dirname) / "review_sample_manifest.csv"


def time_log_path(project_root: Path, review_dirname: str = REVIEW_DIRNAME) -> Path:
    return review_root(project_root, review_dirname) / "review_time_log.csv"


def correction_summary_path(project_root: Path, review_dirname: str = REVIEW_DIRNAME) -> Path:
    return review_root(project_root, review_dirname) / "correction_summary.csv"


def base_stem(image_name: str) -> str:
    stem = Path(image_name).stem
    if stem.endswith("_warped"):
        stem = stem[: -len("_warped")]
    return stem


def warped_name_from_image_name(image_name: str) -> str:
    stem = base_stem(image_name)
    return f"{stem}_warped.png"


def preannotation_json_path(project_root: Path, image_name: str, review_dirname: str = REVIEW_DIRNAME) -> Path:
    return preannotation_dir(project_root, review_dirname) / "json" / f"{base_stem(image_name)}_preannotation.json"


def corrected_json_path(project_root: Path, image_name: str, review_dirname: str = REVIEW_DIRNAME) -> Path:
    return corrected_dir(project_root, review_dirname) / f"{base_stem(image_name)}_corrected.json"


def corrected_preview_path(project_root: Path, image_name: str, review_dirname: str = REVIEW_DIRNAME) -> Path:
    return corrected_preview_dir(project_root, review_dirname) / f"{base_stem(image_name)}_corrected.png"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def full_to_short_keypoints(points: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for full, short in FULL_TO_SHORT.items():
        point = points.get(full) or points.get(short)
        xy = xy_from_payload(point)
        if xy is not None:
            out[short] = {"x": xy[0], "y": xy[1]}
    return out


def short_to_full_keypoints(points: Mapping[str, Any]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for short, full in SHORT_TO_FULL.items():
        point = points.get(short) or points.get(full)
        xy = xy_from_payload(point)
        if xy is not None:
            out[full] = [float(xy[0]), float(xy[1])]
    return out


def xy_from_payload(point: Any) -> tuple[float, float] | None:
    if isinstance(point, Mapping):
        if "x" in point and "y" in point:
            return float(point["x"]), float(point["y"])
        return None
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        return float(point[0]), float(point[1])
    return None


def path_from_project(project_root: Path, value: str | Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else project_root / path


def load_review_sample(project_root: Path, review_dirname: str = REVIEW_DIRNAME) -> pd.DataFrame:
    path = sample_manifest_path(project_root, review_dirname)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def update_sample_status(
    project_root: Path,
    image_name: str,
    status: str,
    *,
    notes: str = "",
    exclude_reason: str = "",
    specimen_id: str | None = None,
    specimen_id_status: str | None = None,
    review_dirname: str = REVIEW_DIRNAME,
) -> None:
    path = sample_manifest_path(project_root, review_dirname)
    if not path.exists() or status not in REVIEW_STATUS_VALUES:
        return
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "review_status" not in df.columns:
        df["review_status"] = "pending_review"
    mask = df["image_name"].astype(str) == str(image_name)
    df.loc[mask, "review_status"] = status
    if exclude_reason:
        if "exclude_reason" not in df.columns:
            df["exclude_reason"] = ""
        df.loc[mask, "exclude_reason"] = exclude_reason
    if notes:
        if "notes" not in df.columns:
            df["notes"] = ""
        df.loc[mask, "notes"] = notes
    if specimen_id is not None:
        if "specimen_id" not in df.columns:
            df["specimen_id"] = ""
        df.loc[mask, "specimen_id"] = specimen_id
    if specimen_id_status is not None:
        if "specimen_id_status" not in df.columns:
            df["specimen_id_status"] = ""
        df.loc[mask, "specimen_id_status"] = specimen_id_status
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_preannotation_artifacts(
    project_root: Path,
    record: Mapping[str, Any],
    warped_image: np.ndarray,
    result: Mapping[str, Any],
    *,
    mm_per_pixel: float = 0.1,
    review_dirname: str = REVIEW_DIRNAME,
) -> dict[str, Path]:
    """Save an unverified v0.4.1 preannotation bundle for one image."""
    image_name = str(record["image_name"])
    root = preannotation_dir(project_root, review_dirname)
    for folder in ("json", "previews", "debug_json", "masks", "crops"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    metadata = dict(result.get("metadata", {}) or {})
    metadata["annotation_status"] = "auto_preannotation_unverified"
    metadata["review_workflow_version"] = REVIEW_VERSION
    metadata.setdefault("preannotation_mode", "v0.4_hybrid_heatmap_geometry")
    metadata["show_heatmap_points"] = True
    metadata["show_v034_points"] = True
    metadata["show_mask_suggestions"] = True
    metadata["show_qc_warnings"] = True
    metadata["show_local_normal_measurement_suggestions"] = True
    metadata["show_width_profiles"] = True
    metadata["show_curvature_qc"] = True

    fish_mask = result.get("fish_mask")
    if fish_mask is not None:
        save_mask_png(np.asarray(fish_mask), root / "masks" / f"{base_stem(image_name)}_fish_mask.png")

    crop_path = root / "crops" / f"{base_stem(image_name)}_crop.png"
    crop_box = (
        (metadata.get("heatmap_debug_info") or {})
        .get("crop_box")
        if isinstance(metadata.get("heatmap_debug_info"), Mapping)
        else None
    )
    if isinstance(crop_box, (list, tuple)) and len(crop_box) == 4:
        x1, y1, x2, y2 = [int(round(float(v))) for v in crop_box]
        crop = ensure_rgb(warped_image)[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size:
            Image.fromarray(crop).save(crop_path)

    overlay = draw_enhanced_preannotation_overlay(warped_image, metadata)
    preview_path = root / "previews" / f"{base_stem(image_name)}_preannotation.png"
    Image.fromarray(ensure_rgb(overlay)).save(preview_path)

    qc_summary = {
        "segmentation_success": metadata.get("segmentation_success", ""),
        "segmentation_quality": metadata.get("segmentation_quality", ""),
        "p7v_valid": metadata.get("p7v_valid", ""),
        "hybrid_qc_results": metadata.get("hybrid_qc_results", {}),
        "curvature_qc": metadata.get("curvature_qc", {}),
        "review_reason": metadata.get("review_reason", ""),
    }

    payload = {
        "review_workflow_version": REVIEW_VERSION,
        "annotation_status": "auto_preannotation_unverified",
        "image_name": image_name,
        "original_file_name": record.get("original_file_name", ""),
        "warped_image_path": record.get("warped_image_path", ""),
        "specimen_id": record.get("specimen_id", ""),
        "view_id": record.get("view_id", ""),
        "source_status": record.get("source_status", ""),
        "used_in_training_dataset": record.get("used_in_training_dataset", ""),
        "mm_per_pixel": mm_per_pixel,
        "heatmap_keypoints": metadata.get("heatmap_keypoints", {}),
        "v034_keypoints": metadata.get("v034_keypoints", {}),
        "mask_suggestions": metadata.get("mask_suggestions", {}),
        "geometric_suggestions": metadata.get("geometric_suggestions", {}),
        "local_normal_suggestions": metadata.get("local_normal_suggestions", {}),
        "hybrid_keypoints": metadata.get("hybrid_keypoints", result.get("keypoints", {})),
        "local_normal_measurements": metadata.get("local_normal_measurements", {}),
        "curvature_qc": metadata.get("curvature_qc", {}),
        "tail_qc": metadata.get("tail_qc", {}),
        "body_depth_geometry": metadata.get("body_depth_geometry", {}),
        "peduncle_depth_geometry": metadata.get("peduncle_depth_geometry", {}),
        "operculum_qc": metadata.get("operculum_qc", {}),
        "derived_points": metadata.get("derived_points", {}),
        "point_sources": metadata.get("point_sources", {}),
        "hybrid_qc_results": metadata.get("hybrid_qc_results", {}),
        "review_reason": metadata.get("review_reason", ""),
        "preannotation_metadata": metadata,
        "qc_summary": qc_summary,
        "paths": {
            "preview": str(preview_path),
            "mask": str(root / "masks" / f"{base_stem(image_name)}_fish_mask.png"),
            "crop": str(crop_path),
        },
    }
    json_path = preannotation_json_path(project_root, image_name, review_dirname)
    write_json(json_path, payload)
    write_json(root / "debug_json" / f"{base_stem(image_name)}_debug.json", payload)
    return {"json": json_path, "preview": preview_path, "crop": crop_path}


def upsert_review_time_log(
    project_root: Path,
    row: Mapping[str, Any],
    *,
    review_dirname: str = REVIEW_DIRNAME,
) -> Path:
    path = time_log_path(project_root, review_dirname)
    columns = [
        "image_name",
        "specimen_id",
        "review_start_time",
        "review_end_time",
        "duration_sec",
        "preannotation_mode",
        "num_points_manually_moved",
        "moved_keypoints",
        "num_qc_warnings",
        "qc_warning_types",
        "reviewer_notes",
        "saved_as_corrected",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() else pd.DataFrame(columns=columns)
    for column in columns:
        if column not in existing.columns:
            existing[column] = ""
    existing = existing.loc[existing["image_name"].astype(str) != str(row.get("image_name", "")), columns]
    combined = pd.concat([existing, pd.DataFrame([{column: row.get(column, "") for column in columns}])], ignore_index=True)
    combined.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def upsert_correction_summary(
    project_root: Path,
    *,
    image_name: str,
    specimen_id: str,
    pre_points: Mapping[str, Any],
    corrected_points: Mapping[str, Any],
    point_sources: Mapping[str, Any],
    review_reason: str,
    mm_per_pixel: float,
    review_dirname: str = REVIEW_DIRNAME,
) -> Path:
    path = correction_summary_path(project_root, review_dirname)
    columns = [
        "image_name",
        "specimen_id",
        "keypoint_name",
        "pre_x",
        "pre_y",
        "corrected_x",
        "corrected_y",
        "displacement_px",
        "displacement_mm",
        "was_moved",
        "point_source_before",
        "review_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() else pd.DataFrame(columns=columns)
    for column in columns:
        if column not in existing.columns:
            existing[column] = ""
    existing = existing.loc[existing["image_name"].astype(str) != image_name, columns]
    rows = []
    for key in KEYPOINT_KEYS:
        pre = xy_from_payload(pre_points.get(key))
        cor = xy_from_payload(corrected_points.get(key))
        if pre is None or cor is None:
            continue
        displacement_px = math.hypot(cor[0] - pre[0], cor[1] - pre[1])
        rows.append(
            {
                "image_name": image_name,
                "specimen_id": specimen_id,
                "keypoint_name": key,
                "pre_x": round(pre[0], 3),
                "pre_y": round(pre[1], 3),
                "corrected_x": round(cor[0], 3),
                "corrected_y": round(cor[1], 3),
                "displacement_px": round(displacement_px, 3),
                "displacement_mm": round(displacement_px * mm_per_pixel, 3),
                "was_moved": bool(displacement_px * mm_per_pixel > 0.5),
                "point_source_before": point_sources.get(key, ""),
                "review_reason": review_reason,
            }
        )
    combined = pd.concat([existing, pd.DataFrame(rows, columns=columns)], ignore_index=True) if rows else existing
    combined.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_corrected_review_annotation(
    project_root: Path,
    *,
    image_name: str,
    warped_image: np.ndarray,
    corrected_full: Mapping[str, Any],
    preannotation_payload: Mapping[str, Any],
    review_start_time: str,
    reviewer_notes: str,
    mm_per_pixel: float = 0.1,
    measurement_axis_mode: str = MEASUREMENT_AXIS_MODEL,
    review_dirname: str = REVIEW_DIRNAME,
) -> dict[str, Path]:
    corrected_dir(project_root, review_dirname).mkdir(parents=True, exist_ok=True)
    corrected_preview_dir(project_root, review_dirname).mkdir(parents=True, exist_ok=True)
    specimen_id = str(preannotation_payload.get("specimen_id", ""))
    pre_points = preannotation_payload.get("hybrid_keypoints", {})
    point_sources = preannotation_payload.get("point_sources", {})
    if not isinstance(pre_points, Mapping):
        pre_points = {}
    if not isinstance(point_sources, Mapping):
        point_sources = {}
    corrected_short = full_to_short_keypoints(corrected_full)
    measurements = calculate_measurements(corrected_short, mm_per_pixel)
    measurement_axis_payload = compute_measurement_axis_metadata(
        warped_image,
        corrected_full,
        mm_per_pixel=mm_per_pixel,
        measurement_axis_mode=measurement_axis_mode,
    )
    measurement_axes = dict(measurement_axis_payload.get("measurement_axes", {}) or {})
    peduncle_depth_geometry = (preannotation_payload.get("preannotation_metadata", {}) or {}).get("peduncle_depth_geometry", {})
    if not isinstance(peduncle_depth_geometry, Mapping) or not peduncle_depth_geometry:
        try:
            fish_mask = (preannotation_payload.get("preannotation_metadata", {}) or {}).get("fish_mask")
            if fish_mask is None:
                fish_mask = preannotation_payload.get("fish_mask")
            if fish_mask is not None:
                peduncle_depth_geometry = estimate_peduncle_depth_by_axis_normals(
                    fish_mask,
                    corrected_full,
                    warped_image.shape,
                    config={"mm_per_pixel": mm_per_pixel},
                )
        except Exception as exc:  # noqa: BLE001
            peduncle_depth_geometry = {
                "peduncle_depth_source": "P4_P5_axis_normal_min_width",
                "peduncle_depth_geometry_version": "v0.6.8_peduncle_axis_normals",
                "peduncle_depth_geometric_qc_pass": False,
                "peduncle_depth_geometric_review_reason": f"peduncle_depth_geometry_failed:{exc}",
            }
    axis_row_fields = dict(measurement_axis_payload.get("measurement_axis_row_fields", {}) or {})
    for key in (
        "TL_curve_selected_mm",
        "SL_curve_selected_mm",
        "curvature_index_selected",
        "selected_measurement_axis",
        "TL_curve_axis_diff_mm",
        "SL_curve_axis_diff_mm",
        "curvature_index_axis_diff",
        "dual_axis_disagreement",
    ):
        if key in axis_row_fields:
            measurement_axes[key] = axis_row_fields[key]
    for key, value in axis_row_fields.items():
        measurements[key] = value
    selected_axis = str(axis_row_fields.get("selected_measurement_axis", ""))
    if selected_axis in {"model_axis", "body_midline_axis"}:
        for source_key, compat_key in (
            ("TL_curve_selected_mm", "TL_curve_mm"),
            ("SL_curve_selected_mm", "SL_curve_mm"),
            ("curvature_index_selected", "curvature_index"),
        ):
            if axis_row_fields.get(source_key) is not None:
                measurements[compat_key] = axis_row_fields.get(source_key)
    overlay = draw_keypoints_and_measurements(
        warped_image,
        corrected_short,
        measurements,
        image_name=image_name,
        specimen_id=specimen_id,
        axis_mode_selected=str(measurements.get("axis_mode_selected", "")),
    )
    overlay = draw_enhanced_preannotation_overlay(
        overlay,
        {
            "corrected_keypoints": corrected_full,
            "measurement_axes": measurement_axes,
            "show_model_axis": True,
            "show_body_midline_axis": True,
            "show_geometric_c_points": True,
            "show_body_midline_contour": True,
            "show_selected_measurement_axis_only": True,
            "show_dual_axis_comparison": True,
        },
    )
    preview_path = corrected_preview_path(project_root, image_name, review_dirname)
    Image.fromarray(ensure_rgb(overlay)).save(preview_path)

    end_time = datetime.now().isoformat(timespec="seconds")
    start = datetime.fromisoformat(review_start_time) if review_start_time else datetime.now()
    end = datetime.fromisoformat(end_time)
    duration = max(0.0, (end - start).total_seconds())
    moved = []
    displacements = []
    for key in KEYPOINT_KEYS:
        pre = xy_from_payload(pre_points.get(key))
        cor = xy_from_payload(corrected_full.get(key))
        if pre is None or cor is None:
            continue
        displacement_mm = math.hypot(cor[0] - pre[0], cor[1] - pre[1]) * mm_per_pixel
        if displacement_mm > 0.5:
            moved.append(key)
            displacements.append(displacement_mm)

    review_reason = str(preannotation_payload.get("review_reason", ""))
    qc_warning_types = [item for item in review_reason.replace(",", ";").split(";") if item.strip()]
    upsert_review_time_log(
        project_root,
        {
            "image_name": image_name,
            "specimen_id": specimen_id,
            "review_start_time": review_start_time,
            "review_end_time": end_time,
            "duration_sec": round(duration, 2),
            "preannotation_mode": preannotation_payload.get("preannotation_metadata", {}).get("preannotation_mode", "v0.4_hybrid_heatmap_geometry"),
            "num_points_manually_moved": len(moved),
            "moved_keypoints": ";".join(moved),
            "num_qc_warnings": len(qc_warning_types),
            "qc_warning_types": ";".join(qc_warning_types),
            "reviewer_notes": reviewer_notes,
            "saved_as_corrected": True,
        },
        review_dirname=review_dirname,
    )
    upsert_correction_summary(
        project_root,
        image_name=image_name,
        specimen_id=specimen_id,
        pre_points=pre_points,
        corrected_points=corrected_full,
        point_sources=point_sources,
        review_reason=review_reason,
        mm_per_pixel=mm_per_pixel,
        review_dirname=review_dirname,
    )
    update_sample_status(
        project_root,
        image_name,
        "corrected_and_confirmed",
        specimen_id=specimen_id,
        specimen_id_status="confirmed" if specimen_id and specimen_id != "unknown" else "needs_manual_input",
        review_dirname=review_dirname,
    )

    payload = {
        "review_workflow_version": REVIEW_VERSION,
        "annotation_status": "corrected_and_confirmed",
        "annotation_mode": "auto_assisted_hybrid_v0.4.1_realworld_review",
        "image_name": image_name,
        "specimen_id": specimen_id,
        "created_at": end_time,
        "mm_per_pixel": mm_per_pixel,
        "corrected_keypoints": {key: list(map(float, xy_from_payload(value))) for key, value in corrected_full.items() if xy_from_payload(value) is not None},
        "preannotation": preannotation_payload,
        "tail_qc": (preannotation_payload.get("preannotation_metadata", {}) or {}).get("tail_qc", {}),
        "body_depth_geometry": (preannotation_payload.get("preannotation_metadata", {}) or {}).get("body_depth_geometry", {}),
        "peduncle_depth_geometry": peduncle_depth_geometry,
        "operculum_qc": (preannotation_payload.get("preannotation_metadata", {}) or {}).get("operculum_qc", {}),
        "measurement_axis_mode_request": measurement_axis_mode,
        "measurement_axes": measurement_axes,
        "measurements": measurements,
        "reviewer_notes": reviewer_notes,
        "paths": {"corrected_preview": str(preview_path)},
    }
    json_path = corrected_json_path(project_root, image_name, review_dirname)
    write_json(json_path, payload)
    return {
        "json": json_path,
        "preview": preview_path,
        "time_log": time_log_path(project_root, review_dirname),
        "correction_summary": correction_summary_path(project_root, review_dirname),
    }


def draw_review_comparison(
    image: np.ndarray,
    pre_points: Mapping[str, Any],
    corrected_points: Mapping[str, Any],
    output_path: Path,
    *,
    title: str = "",
    mm_per_pixel: float = 0.1,
) -> Path:
    pil = Image.fromarray(ensure_rgb(image))
    scale = min(1.0, 1800 / max(1, pil.width))
    if scale < 1.0:
        pil = pil.resize((int(pil.width * scale), int(pil.height * scale)), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(pil)
    radius = max(5, int(round(min(pil.size) / 260)))
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 18)
    except OSError:
        font = ImageFont.load_default()
    large = []
    for key in KEYPOINT_KEYS:
        pre = xy_from_payload(pre_points.get(key))
        cor = xy_from_payload(corrected_points.get(key))
        if pre is None or cor is None:
            continue
        ps = (pre[0] * scale, pre[1] * scale)
        cs = (cor[0] * scale, cor[1] * scale)
        displacement_mm = math.hypot(cor[0] - pre[0], cor[1] - pre[1]) * mm_per_pixel
        draw.line((ps[0], ps[1], cs[0], cs[1]), fill=(255, 255, 255), width=1)
        draw.ellipse((ps[0] - radius, ps[1] - radius, ps[0] + radius, ps[1] + radius), outline=(245, 70, 50), width=3)
        draw.ellipse((cs[0] - radius, cs[1] - radius, cs[0] + radius, cs[1] + radius), fill=(60, 220, 90), outline=(10, 80, 20), width=2)
        if displacement_mm > 5:
            large.append(key)
            draw.ellipse((cs[0] - radius - 8, cs[1] - radius - 8, cs[0] + radius + 8, cs[1] + radius + 8), outline=(255, 230, 0), width=4)
    label = title or output_path.stem
    label += f" | large moved: {';'.join(large) if large else 'none'}"
    draw.rectangle((8, 8, min(pil.width - 8, 24 + len(label) * 10), 42), fill=(0, 0, 0))
    draw.text((16, 12), label, font=font, fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(output_path, optimize=True)
    return output_path
