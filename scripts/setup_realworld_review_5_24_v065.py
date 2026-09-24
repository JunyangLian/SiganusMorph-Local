"""Import, preprocess, and preannotate the 5.24 new-fish review batch.

This script is intentionally batch-scoped.  It writes only under
``results/realworld_review_5_24_v0.6.5/`` and does not modify any existing
corrected labels or training datasets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph.aruco_utils import (  # noqa: E402
    detect_aruco_markers,
    select_board_config_for_markers,
    warp_board_by_aruco,
)
from siganusmorph.dual_axis_measurement import (  # noqa: E402
    MEASUREMENT_AXIS_AUTO_QC,
    compute_measurement_axis_metadata,
)
from siganusmorph.image_utils import ensure_rgb, load_image_file  # noqa: E402
from siganusmorph.realworld_review import base_stem, write_json  # noqa: E402
from siganusmorph.segmentation import padded_bbox, save_mask_png, segment_fish_from_blue_board  # noqa: E402
from siganusmorph.v06_preannotation import preannotate_warped_image_v06_keypointwise  # noqa: E402
from siganusmorph.visualization import draw_enhanced_preannotation_overlay  # noqa: E402


SOURCE_DIR = Path(r"E:\1_yanjiusheng\caiyang\5.24")
REVIEW_DIRNAME = "realworld_review_5_24_v0.6.5"
REVIEW_ROOT = PROJECT_ROOT / "results" / REVIEW_DIRNAME
SOURCE_BATCH = "5_24_new_fish"
VERSION_NAME = "realworld_review_5_24_v0.6.5"
SUPPORTED_EXTS = {".heic", ".heif", ".jpg", ".jpeg", ".png"}


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _ensure_dirs() -> None:
    for folder in (
        "converted_images",
        "warped_images",
        "maskcrop_images",
        "fish_masks",
        "preannotations/json",
        "preannotations/debug_json",
        "preannotations/masks",
        "preannotations/crops",
        "preannotation_previews",
        "corrected_keypoints",
        "corrected_previews",
    ):
        (REVIEW_ROOT / folder).mkdir(parents=True, exist_ok=True)


def _source_images() -> list[Path]:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Input directory does not exist: {SOURCE_DIR}")
    return sorted(path for path in SOURCE_DIR.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS)


def _converted_name(index: int) -> str:
    return f"real_5_24_{index:03d}.png"


def import_images() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, original_path in enumerate(_source_images(), start=1):
        image_name = _converted_name(index)
        converted_path = REVIEW_ROOT / "converted_images" / image_name
        include = True
        exclude_reason = ""
        notes = ""
        try:
            image = load_image_file(original_path)
            Image.fromarray(ensure_rgb(image)).save(converted_path)
        except Exception as exc:  # noqa: BLE001 - keep row for manual follow-up.
            include = False
            exclude_reason = "conversion_failed"
            notes = str(exc)
            converted_path = Path("")
        rows.append(
            {
                "original_filename": original_path.name,
                "original_path": str(original_path),
                "converted_image_path": _rel(converted_path) if converted_path else "",
                "image_name": image_name,
                "specimen_id": "unknown",
                "specimen_id_status": "needs_manual_input",
                "view_or_angle": "",
                "source_batch": SOURCE_BATCH,
                "include_for_review": include,
                "exclude_reason": exclude_reason,
                "notes": notes,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(REVIEW_ROOT / "import_manifest.csv", index=False, encoding="utf-8-sig")
    return df


def preprocess_images(import_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    qc_rows: list[dict[str, Any]] = []
    warped_images: dict[str, np.ndarray] = {}
    for _, row in import_df.iterrows():
        image_name = str(row["image_name"])
        converted_path = PROJECT_ROOT / str(row.get("converted_image_path", ""))
        qc: dict[str, Any] = {
            "image_name": image_name,
            "warp_success": False,
            "aruco_detected_count": 0,
            "charuco_success": False,
            "scale_mm_per_pixel": "",
            "segmentation_success": False,
            "fish_mask_area": 0,
            "fish_bbox": "",
            "preprocess_warning": "",
            "recommended_action": "needs_manual_check",
            "warped_image_path": "",
            "crop_image_path": "",
            "fish_mask_path": "",
        }
        if not converted_path.exists():
            qc["preprocess_warning"] = "missing_converted_image"
            qc_rows.append(qc)
            continue
        try:
            image = load_image_file(converted_path)
            detection = detect_aruco_markers(image)
            markers = detection.get("markers", {}) or {}
            qc["aruco_detected_count"] = len(markers)
            board_config = select_board_config_for_markers(markers)
            warped, _transform, info = warp_board_by_aruco(image, markers, board_config)
            qc["warp_success"] = True
            qc["charuco_success"] = "location_marker_squares_mm" in board_config
            qc["scale_mm_per_pixel"] = float(info.get("mm_per_pixel", 0.1))
            warped_path = REVIEW_ROOT / "warped_images" / f"{base_stem(image_name)}_warped.png"
            Image.fromarray(ensure_rgb(warped)).save(warped_path)
            qc["warped_image_path"] = _rel(warped_path)
            warped_images[image_name] = warped
        except Exception as exc:  # noqa: BLE001
            qc["preprocess_warning"] = f"warp_failed:{exc}"
            qc_rows.append(qc)
            continue
        try:
            fish_mask, fish_bbox, quality = segment_fish_from_blue_board(warped)
            seg_success = bool(quality.get("segmentation_success", False))
            qc["segmentation_success"] = seg_success
            qc["fish_mask_area"] = int(quality.get("mask_area_px", 0) or 0)
            qc["fish_bbox"] = json.dumps(list(fish_bbox))
            mask_path = REVIEW_ROOT / "fish_masks" / f"{base_stem(image_name)}_fish_mask.png"
            save_mask_png(fish_mask, mask_path)
            qc["fish_mask_path"] = _rel(mask_path)
            if seg_success:
                crop_box = padded_bbox(fish_bbox, warped.shape, padding_ratio=0.08)
                x1, y1, x2, y2 = crop_box
                crop = ensure_rgb(warped)[y1:y2, x1:x2]
                crop_path = REVIEW_ROOT / "maskcrop_images" / f"{base_stem(image_name)}_maskcrop.png"
                Image.fromarray(crop).save(crop_path)
                qc["crop_image_path"] = _rel(crop_path)
                qc["recommended_action"] = "review_preannotation"
            else:
                qc["preprocess_warning"] = str(quality.get("segmentation_quality", "segmentation_failed"))
        except Exception as exc:  # noqa: BLE001
            qc["preprocess_warning"] = f"segmentation_failed:{exc}"
        qc_rows.append(qc)
    qc_df = pd.DataFrame(qc_rows)
    qc_df.to_csv(REVIEW_ROOT / "preprocess_qc.csv", index=False, encoding="utf-8-sig")
    return qc_df, warped_images


def _save_preannotation_bundle(
    record: Mapping[str, Any],
    warped_image: np.ndarray,
    result: Mapping[str, Any],
    *,
    mm_per_pixel: float,
    preprocess_row: Mapping[str, Any],
) -> dict[str, str]:
    image_name = str(record["image_name"])
    metadata = dict(result.get("metadata", {}) or {})
    try:
        axis_payload = compute_measurement_axis_metadata(
            warped_image,
            metadata.get("hybrid_keypoints", result.get("keypoints", {})),
            mm_per_pixel=mm_per_pixel,
            measurement_axis_mode=MEASUREMENT_AXIS_AUTO_QC,
        )
        measurement_axes = dict(axis_payload.get("measurement_axes", {}) or {})
        row_fields = dict(axis_payload.get("measurement_axis_row_fields", {}) or {})
        for key, value in row_fields.items():
            measurement_axes[key] = value
        metadata["measurement_axes"] = measurement_axes
        metadata["measurement_axis_mode_request"] = MEASUREMENT_AXIS_AUTO_QC
        metadata["selected_measurement_axis"] = measurement_axes.get("selected_measurement_axis", "")
    except Exception as exc:  # noqa: BLE001
        metadata.setdefault("measurement_axes", {})
        metadata["measurement_axis_error"] = str(exc)

    metadata.update(
        {
            "annotation_status": "auto_preannotation_unverified",
            "review_workflow_version": VERSION_NAME,
            "source_batch": SOURCE_BATCH,
            "show_qc_warnings": True,
            "show_model_axis": True,
            "show_body_midline_axis": True,
            "show_geometric_c_points": True,
            "show_dual_axis_comparison": True,
            "show_selected_measurement_axis_only": False,
        }
    )
    fish_mask = result.get("fish_mask")
    mask_path = REVIEW_ROOT / "preannotations" / "masks" / f"{base_stem(image_name)}_fish_mask.png"
    if fish_mask is not None:
        save_mask_png(np.asarray(fish_mask), mask_path)
    crop_src = str(preprocess_row.get("crop_image_path", "") or "")
    crop_dest = REVIEW_ROOT / "preannotations" / "crops" / f"{base_stem(image_name)}_crop.png"
    if crop_src:
        crop_path = PROJECT_ROOT / crop_src
        if crop_path.exists():
            Image.open(crop_path).save(crop_dest)

    preview = draw_enhanced_preannotation_overlay(warped_image, metadata)
    preview_path = REVIEW_ROOT / "preannotation_previews" / f"{base_stem(image_name)}_preannotation.png"
    Image.fromarray(ensure_rgb(preview)).save(preview_path)
    metadata = _json_safe(metadata)
    payload = {
        "review_workflow_version": VERSION_NAME,
        "annotation_status": "auto_preannotation_unverified",
        "image_name": image_name,
        "original_file_name": record.get("original_filename", ""),
        "original_path": record.get("original_path", ""),
        "warped_image_path": preprocess_row.get("warped_image_path", ""),
        "crop_image_path": preprocess_row.get("crop_image_path", ""),
        "specimen_id": record.get("specimen_id", "unknown"),
        "specimen_id_status": record.get("specimen_id_status", "needs_manual_input"),
        "view_id": record.get("view_or_angle", ""),
        "source_batch": SOURCE_BATCH,
        "mm_per_pixel": mm_per_pixel,
        "preannotation_mode": "v0.6_keypointwise_hybrid_experimental",
        "measurement_axis_mode": MEASUREMENT_AXIS_AUTO_QC,
        "heatmap_keypoints": metadata.get("heatmap_keypoints", {}),
        "v01_heatmap_keypoints": metadata.get("v01_heatmap_keypoints", {}),
        "v05_heatmap_keypoints": metadata.get("v05_heatmap_keypoints", {}),
        "v034_keypoints": metadata.get("v034_keypoints", {}),
        "v041_hybrid_keypoints": metadata.get("v041_hybrid_keypoints", {}),
        "v06_keypoints": metadata.get("v06_keypoints", {}),
        "v06_point_sources": metadata.get("v06_point_sources", {}),
        "v06_qc_results": metadata.get("v06_qc_results", {}),
        "model_keypoints_raw": metadata.get("model_keypoints_raw", {}),
        "mask_suggestions": metadata.get("mask_suggestions", {}),
        "geometric_suggestions": metadata.get("geometric_suggestions", {}),
        "hybrid_keypoints": metadata.get("hybrid_keypoints", result.get("keypoints", {})),
        "corrected_keypoints": metadata.get("hybrid_keypoints", result.get("keypoints", {})),
        "local_normal_measurements": metadata.get("local_normal_measurements", {}),
        "body_depth_geometry": metadata.get("body_depth_geometry", {}),
        "peduncle_depth_geometry": metadata.get("peduncle_depth_geometry", {}),
        "operculum_qc": metadata.get("operculum_qc", {}),
        "curvature_qc": metadata.get("curvature_qc", {}),
        "tail_qc": metadata.get("tail_qc", {}),
        "P6_gap_derivation": metadata.get("P6_gap_derivation", {}),
        "derived_points": metadata.get("derived_points", {}),
        "point_sources": metadata.get("point_sources", {}),
        "hybrid_qc_results": metadata.get("hybrid_qc_results", {}),
        "measurement_axes": metadata.get("measurement_axes", {}),
        "review_reason": metadata.get("review_reason", ""),
        "preannotation_metadata": metadata,
        "keypoint_edit_log": metadata.get("keypoint_edit_log", []),
        "paths": {
            "preview": _rel(preview_path),
            "mask": _rel(mask_path),
            "crop": _rel(crop_dest) if crop_dest.exists() else crop_src,
        },
    }
    json_path = REVIEW_ROOT / "preannotations" / "json" / f"{base_stem(image_name)}_preannotation.json"
    debug_path = REVIEW_ROOT / "preannotations" / "debug_json" / f"{base_stem(image_name)}_debug.json"
    write_json(json_path, _json_safe(payload))
    write_json(debug_path, _json_safe(payload))
    return {
        "preannotation_json_path": _rel(json_path),
        "preannotation_preview_path": _rel(preview_path),
        "preannotation_debug_path": _rel(debug_path),
        "selected_measurement_axis": (metadata.get("measurement_axes", {}) or {}).get("selected_measurement_axis", ""),
        "dual_axis_disagreement": ((metadata.get("measurement_axes", {}) or {}).get("dual_axis_comparison", {}) or {}).get("dual_axis_disagreement", ""),
        "measurements_needs_review": (metadata.get("measurement_axes", {}) or {}).get("measurements_needs_review", ""),
    }


def run_preannotations(import_df: pd.DataFrame, preprocess_df: pd.DataFrame, warped_images: dict[str, np.ndarray]) -> pd.DataFrame:
    preprocess_by_name = {str(row["image_name"]): row for _, row in preprocess_df.iterrows()}
    import_by_name = {str(row["image_name"]): row for _, row in import_df.iterrows()}
    qc_rows: list[dict[str, Any]] = []
    for image_name, pre_row in preprocess_by_name.items():
        record = import_by_name.get(image_name, {})
        preannotation_success = False
        notes = ""
        num_keypoints = 0
        fields: dict[str, Any] = {}
        if str(pre_row.get("warp_success", "")).lower() != "true" or str(pre_row.get("segmentation_success", "")).lower() != "true":
            notes = "skipped_preannotation_due_to_preprocess_failure"
        else:
            try:
                warped = warped_images.get(image_name)
                if warped is None:
                    warped = load_image_file(PROJECT_ROOT / str(pre_row.get("warped_image_path", "")))
                mm_per_pixel = float(pre_row.get("scale_mm_per_pixel") or 0.1)
                result = preannotate_warped_image_v06_keypointwise(
                    warped,
                    image_name,
                    PROJECT_ROOT,
                    mm_per_pixel=mm_per_pixel,
                )
                paths = _save_preannotation_bundle(record, warped, result, mm_per_pixel=mm_per_pixel, preprocess_row=pre_row)
                metadata = dict(result.get("metadata", {}) or {})
                keypoints = dict(metadata.get("hybrid_keypoints", result.get("keypoints", {})) or {})
                num_keypoints = len(keypoints)
                tail_qc = metadata.get("tail_qc", {}) if isinstance(metadata.get("tail_qc", {}), Mapping) else {}
                hybrid_qc = metadata.get("hybrid_qc_results", {}) if isinstance(metadata.get("hybrid_qc_results", {}), Mapping) else {}
                preannotation_success = True
                fields.update(paths)
                fields.update(
                    {
                        "p7v_valid": hybrid_qc.get("p7v_valid", metadata.get("p7v_valid", "")),
                        "P6_geometry_qc_pass": tail_qc.get("P6_geometry_qc_pass", ""),
                        "P6_needs_review": tail_qc.get("P6_needs_review", ""),
                        "preannotation_reliability_level": "needs_review" if metadata.get("needs_review", False) else "ok",
                        "measurement_axis_mode": MEASUREMENT_AXIS_AUTO_QC,
                        "selected_measurement_axis": paths.get("selected_measurement_axis", ""),
                        "dual_axis_disagreement": paths.get("dual_axis_disagreement", ""),
                        "measurements_needs_review": paths.get("measurements_needs_review", ""),
                        "recommended_action": "manual_review_confirm",
                        "review_points": ";".join(tail_qc.get("review_points", []) if isinstance(tail_qc.get("review_points", []), list) else []),
                    }
                )
                notes = str(metadata.get("review_reason", ""))
            except Exception as exc:  # noqa: BLE001
                notes = f"preannotation_failed:{exc}"
        qc_rows.append(
            {
                "image_name": image_name,
                "specimen_id": record.get("specimen_id", "unknown") if isinstance(record, Mapping) else "unknown",
                "preannotation_success": preannotation_success,
                "num_keypoints": num_keypoints,
                "p7v_valid": fields.get("p7v_valid", ""),
                "P6_geometry_qc_pass": fields.get("P6_geometry_qc_pass", ""),
                "P6_needs_review": fields.get("P6_needs_review", ""),
                "preannotation_reliability_level": fields.get("preannotation_reliability_level", "failed" if not preannotation_success else ""),
                "measurement_axis_mode": fields.get("measurement_axis_mode", MEASUREMENT_AXIS_AUTO_QC if preannotation_success else ""),
                "selected_measurement_axis": fields.get("selected_measurement_axis", ""),
                "dual_axis_disagreement": fields.get("dual_axis_disagreement", ""),
                "measurements_needs_review": fields.get("measurements_needs_review", ""),
                "recommended_action": fields.get("recommended_action", "needs_manual_check"),
                "review_points": fields.get("review_points", ""),
                "notes": notes,
                **{key: fields.get(key, "") for key in ("preannotation_json_path", "preannotation_preview_path", "preannotation_debug_path")},
            }
        )
    qc_df = pd.DataFrame(qc_rows)
    qc_df.to_csv(REVIEW_ROOT / "preannotation_qc.csv", index=False, encoding="utf-8-sig")
    return qc_df


def write_review_manifests(import_df: pd.DataFrame, preprocess_df: pd.DataFrame, pre_qc_df: pd.DataFrame) -> pd.DataFrame:
    merged = import_df.merge(preprocess_df, on="image_name", how="left", suffixes=("", "_preprocess"))
    merged = merged.merge(pre_qc_df, on=["image_name", "specimen_id"], how="left", suffixes=("", "_preannotation"))
    if "review_status" not in merged.columns:
        merged["review_status"] = "pending_review"
    merged["include_for_review"] = merged["include_for_review"].astype(str).str.lower().isin({"true", "1", "yes"})
    merged.loc[merged["preannotation_success"].astype(str).str.lower() != "true", "review_status"] = "needs_later_review"
    merged["source_status"] = "unseen_real_image"
    merged["used_in_training_dataset"] = False
    desired_cols = [
        "image_name",
        "original_filename",
        "original_path",
        "converted_image_path",
        "warped_image_path",
        "crop_image_path",
        "specimen_id",
        "specimen_id_status",
        "view_or_angle",
        "source_batch",
        "source_status",
        "used_in_training_dataset",
        "warp_success",
        "segmentation_success",
        "preannotation_success",
        "preannotation_json_path",
        "preannotation_preview_path",
        "review_status",
        "include_for_review",
        "exclude_reason",
        "notes",
    ]
    for col in desired_cols:
        if col not in merged.columns:
            merged[col] = ""
    sample = merged[desired_cols].copy()
    sample.to_csv(REVIEW_ROOT / "review_manifest.csv", index=False, encoding="utf-8-sig")
    sample.to_csv(REVIEW_ROOT / "review_sample_manifest.csv", index=False, encoding="utf-8-sig")
    return sample


def write_readme(import_df: pd.DataFrame, preprocess_df: pd.DataFrame, pre_qc_df: pd.DataFrame, sample_df: pd.DataFrame) -> None:
    input_count = len(import_df)
    converted_count = int(import_df["converted_image_path"].astype(str).ne("").sum()) if not import_df.empty else 0
    warp_success = int(preprocess_df["warp_success"].astype(str).str.lower().eq("true").sum()) if not preprocess_df.empty else 0
    seg_success = int(preprocess_df["segmentation_success"].astype(str).str.lower().eq("true").sum()) if not preprocess_df.empty else 0
    pre_success = int(pre_qc_df["preannotation_success"].astype(str).str.lower().eq("true").sum()) if not pre_qc_df.empty else 0
    pending = int(sample_df["review_status"].astype(str).isin(["pending_review", "in_progress", "needs_later_review"]).sum()) if not sample_df.empty else 0
    text = f"""# 5.24 New Fish Review Workflow

Workflow version: `{VERSION_NAME}`

This batch imports 5.24 real fish photos, runs v0.6.5 recommended AI-assisted preannotation, and prepares a dedicated Review Queue.  It does not train a model and does not write unverified points as corrected labels.

## Summary

- Input images: {input_count}
- Converted images: {converted_count}
- Warp success: {warp_success}
- Segmentation success: {seg_success}
- Preannotation success: {pre_success}
- Pending review: {pending}

## Review Queue

Open Streamlit and select review batch:

`realworld_review_5_24_v0.6.5`

Specimen IDs are initially `unknown` and should be filled or corrected during review. Corrected labels for this batch are saved only under:

- `results/realworld_review_5_24_v0.6.5/corrected_keypoints/`
- `results/realworld_review_5_24_v0.6.5/corrected_previews/`

## Follow-up

After manual confirmation, run:

```bash
python scripts/evaluate_realworld_review_5_24_v065.py
```

Then prepare the v0.7 dataset plan. Do not train v0.7 until specimen IDs and confirmed labels are checked.
"""
    (REVIEW_ROOT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    _ensure_dirs()
    import_df = import_images()
    preprocess_df, warped_images = preprocess_images(import_df)
    pre_qc_df = run_preannotations(import_df, preprocess_df, warped_images)
    sample_df = write_review_manifests(import_df, preprocess_df, pre_qc_df)
    write_readme(import_df, preprocess_df, pre_qc_df, sample_df)
    input_count = len(import_df)
    converted_count = int(import_df["converted_image_path"].astype(str).ne("").sum()) if not import_df.empty else 0
    warp_success = int(preprocess_df["warp_success"].astype(str).str.lower().eq("true").sum()) if not preprocess_df.empty else 0
    seg_success = int(preprocess_df["segmentation_success"].astype(str).str.lower().eq("true").sum()) if not preprocess_df.empty else 0
    pre_success = int(pre_qc_df["preannotation_success"].astype(str).str.lower().eq("true").sum()) if not pre_qc_df.empty else 0
    pending = int(sample_df["review_status"].astype(str).isin(["pending_review", "in_progress", "needs_later_review"]).sum()) if not sample_df.empty else 0
    print("Finished 5.24 new-fish preannotation and review setup.")
    print(f"Input images: {input_count}")
    print(f"Converted images: {converted_count}")
    print(f"Warp success: {warp_success}")
    print(f"Segmentation success: {seg_success}")
    print(f"Preannotation success: {pre_success}")
    print(f"Pending review: {pending}")
    print("")
    print("Review queue batch:")
    print(REVIEW_DIRNAME)
    print("")
    print("Next step:")
    print("Manually review and confirm 5.24 preannotations in Review Queue.")
    print("After confirmation, run post-review evaluation and v0.7 dataset planning.")


if __name__ == "__main__":
    main()
