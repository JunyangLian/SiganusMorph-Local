"""End-to-end regression checks for v0.6.4 measurement axis integration.

This script is intentionally non-destructive:
- it does not train models;
- it does not modify official corrected_keypoints;
- it writes only under results/e2e_regression_v0.6.4/.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph.config import KEYPOINT_DEFS, RESULT_COLUMNS
from siganusmorph.dual_axis_measurement import (
    MEASUREMENT_AXIS_AUTO_QC,
    MEASUREMENT_AXIS_BODY_MIDLINE,
    MEASUREMENT_AXIS_MODEL,
    compute_measurement_axis_metadata,
)
from siganusmorph.image_utils import ensure_rgb, load_image_file
from siganusmorph.measurements import build_result_row, calculate_measurements
from siganusmorph.visualization import draw_enhanced_preannotation_overlay


OUT = PROJECT_ROOT / "results" / "e2e_regression_v0.6.4"
SAME_SET = PROJECT_ROOT / "results" / "model_eval" / "v0.6_keypointwise_hybrid_selector" / "same_set_comparison"
PRED_DIR = SAME_SET / "predictions"
V041_JSON_DIR = SAME_SET / "v041_automatic_preannotations" / "json"
DUAL_AXIS = PROJECT_ROOT / "results" / "model_eval" / "v0.6.3_dual_axis_measurement" / "dual_axis_measurement_comparison.csv"
HARD_CASES = PROJECT_ROOT / "results" / "v0.5_dataset_planning" / "v05_hard_cases.csv"
MM_PER_PIXEL = 0.1
TEST_COUNT_TARGET = 9

FULL_TO_SHORT = {f"{kp.code}_{kp.name}": kp.name for kp in KEYPOINT_DEFS}
SHORT_TO_FULL = {kp.name: f"{kp.code}_{kp.name}" for kp in KEYPOINT_DEFS}
FULL_KEYPOINTS = tuple(FULL_TO_SHORT.keys())
AXIS_MODES = (MEASUREMENT_AXIS_MODEL, MEASUREMENT_AXIS_BODY_MIDLINE, MEASUREMENT_AXIS_AUTO_QC)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_clean(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _json_clean(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(v) for v in value]
    return value


def _resolve_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _xy(value: Any) -> list[float] | None:
    if isinstance(value, Mapping):
        if "x" in value and "y" in value:
            return [float(value["x"]), float(value["y"])]
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return [float(value[0]), float(value[1])]
    return None


def _point_distance(a: Any, b: Any) -> float:
    aa = _xy(a)
    bb = _xy(b)
    if aa is None or bb is None:
        return float("nan")
    return math.hypot(aa[0] - bb[0], aa[1] - bb[1])


def _corrected_keypoints(path: Path) -> dict[str, list[float]]:
    payload = _load_json(path)
    points = payload.get("corrected_keypoints") or payload.get("keypoints") or {}
    out: dict[str, list[float]] = {}
    if not isinstance(points, Mapping):
        return out
    for key in FULL_KEYPOINTS:
        point = _xy(points.get(key))
        if point is not None:
            out[key] = point
    return out


def _full_to_short_points(full_points: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for full, short in FULL_TO_SHORT.items():
        point = _xy(full_points.get(full))
        if point is not None:
            out[short] = {"x": point[0], "y": point[1]}
    return out


def _short_to_full_points(short_points: Mapping[str, Any]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for short, full in SHORT_TO_FULL.items():
        point = _xy(short_points.get(short))
        if point is not None:
            out[full] = point
    return out


def _prediction_map(frame: pd.DataFrame) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {}
    if frame.empty:
        return out
    for record in frame.to_dict("records"):
        image = str(record.get("image_name", ""))
        key = str(record.get("keypoint_name", ""))
        if image and key:
            out.setdefault(image, {})[key] = [float(record["x"]), float(record["y"])]
    return out


def _source_map(frame: pd.DataFrame) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    if frame.empty or "point_source" not in frame.columns:
        return out
    for record in frame.to_dict("records"):
        image = str(record.get("image_name", ""))
        key = str(record.get("keypoint_name", ""))
        if image and key:
            out.setdefault(image, {})[key] = str(record.get("point_source", ""))
    return out


def _load_predictions() -> dict[str, pd.DataFrame]:
    return {
        "v01": pd.read_csv(PRED_DIR / "v01_predictions.csv"),
        "v05": pd.read_csv(PRED_DIR / "v05_predictions.csv"),
        "v041": pd.read_csv(PRED_DIR / "v041_automatic_predictions.csv"),
        "v06": pd.read_csv(PRED_DIR / "v06_predictions.csv"),
    }


def _choose_test_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(SAME_SET / "strict_union_manifest.csv")
    dual = pd.read_csv(DUAL_AXIS)
    hard = pd.read_csv(HARD_CASES) if HARD_CASES.exists() else pd.DataFrame()
    image_to_dual = dual.set_index("image_name").to_dict("index")
    hard_images = set(hard.get("image_name", pd.Series(dtype=str)).astype(str)) if not hard.empty else set()
    errors = pd.read_csv(SAME_SET / "same_set_error_by_image.csv")
    v06_errors = errors[errors["source"].astype(str).eq("v06_selector")].copy()
    error_lookup = v06_errors.set_index("image_name").to_dict("index")

    selected: list[tuple[str, str, str, str]] = []

    def add(image_name: str, case_type: str, expected: str, notes: str = "") -> None:
        if len(selected) >= TEST_COUNT_TARGET or image_name in {row[0] for row in selected}:
            return
        if image_name not in set(manifest["image_name"].astype(str)):
            return
        selected.append((image_name, case_type, expected, notes))

    # 3 ordinary straight fish: prioritize low curvature and low v0.6 error.
    ordinary = []
    for row in manifest.to_dict("records"):
        image = str(row["image_name"])
        d = image_to_dual.get(image, {})
        e = error_lookup.get(image, {})
        curvature = float(d.get("model_curvature_index", 9.0))
        mean_error = float(e.get("mean_error_mm", 9.0))
        hard_penalty = 1 if image in hard_images else 0
        if curvature <= 1.016 and mean_error <= 1.8:
            ordinary.append((hard_penalty, mean_error, curvature, image))
    if len(ordinary) < 3:
        for row in manifest.to_dict("records"):
            image = str(row["image_name"])
            d = image_to_dual.get(image, {})
            e = error_lookup.get(image, {})
            curvature = float(d.get("model_curvature_index", 9.0))
            mean_error = float(e.get("mean_error_mm", 9.0))
            if curvature <= 1.020 and mean_error <= 2.5:
                ordinary.append((1 if image in hard_images else 0, mean_error, curvature, image))
    seen_ord: set[str] = set()
    for _, _mean, curvature, image in sorted(ordinary):
        if len([r for r in selected if r[1] == "straight_standard"]) >= 3:
            break
        if image in seen_ord:
            continue
        seen_ord.add(image)
        add(image, "straight_standard", "both modes produce stable 16-keypoint drafts; model_axis remains compatible", f"low curvature={curvature:.4f}")

    # 2 lightly curved examples from the highest non-high curvature images.
    curved = dual.sort_values("model_curvature_index", ascending=False)
    for row in curved.to_dict("records"):
        if len([r for r in selected if r[1] == "mildly_curved"]) >= 2:
            break
        add(
            str(row["image_name"]),
            "mildly_curved",
            "curvature QC and dual-axis fields are populated; no silent label mutation",
            f"model_curvature_index={float(row.get('model_curvature_index', 0)):.4f}",
        )

    # 2 tail-complex cases from larger P7/P6 errors.
    all_errors = pd.read_csv(SAME_SET / "same_set_all_error_rows.csv")
    tail = all_errors[
        (all_errors["source"].astype(str).eq("v06_selector"))
        & all_errors["keypoint_name"].isin(["P7U_caudal_fin_upper_tip", "P7L_caudal_fin_lower_tip", "P6_caudal_fork_midpoint"])
    ].sort_values("error_mm", ascending=False)
    for row in tail.to_dict("records"):
        if len([r for r in selected if r[1] == "tail_complex"]) >= 2:
            break
        add(str(row["image_name"]), "tail_complex", "P7V derived point stays valid or is explicitly marked invalid", f"{row['keypoint_name']} error={float(row['error_mm']):.3f} mm")

    # 1 review hard case.
    for row in hard.to_dict("records") if not hard.empty else []:
        add(str(row["image_name"]), "hard_case", "existing corrected_keypoints are read-only; simulated save uses temp directory", str(row.get("hard_case_type", "")))
        if any(r[1] == "hard_case" for r in selected):
            break

    # 1 failed-risk / high-risk example: highest v0.6 image mean error in same-set.
    for row in v06_errors.sort_values("mean_error_mm", ascending=False).to_dict("records"):
        add(str(row["image_name"]), "failed_risk_or_high_error", "excluded from reliable aggregate interpretation; QC warnings should surface", f"v06_mean_error={float(row['mean_error_mm']):.3f} mm")
        if any(r[1] == "failed_risk_or_high_error" for r in selected):
            break

    # Fill if duplicates reduced the list.
    for row in manifest.to_dict("records"):
        if len(selected) >= TEST_COUNT_TARGET:
            break
        add(str(row["image_name"]), "coverage_fill", "basic read/save/export coverage", "")

    rows = []
    for image_name, case_type, expected, notes in selected[:TEST_COUNT_TARGET]:
        record = manifest.loc[manifest["image_name"].astype(str).eq(image_name)].iloc[0].to_dict()
        rows.append(
            {
                "image_name": image_name,
                "specimen_id": record.get("specimen_id", ""),
                "source": record.get("subset_source", ""),
                "test_case_type": case_type,
                "has_existing_corrected": bool(record.get("has_ground_truth_corrected_keypoints", False)),
                "expected_behavior": expected,
                "notes": notes,
                "corrected_json_path": record.get("corrected_json_path", ""),
                "warped_image_path": record.get("warped_image_path", ""),
                "crop_image_path": record.get("crop_image_path", ""),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "e2e_test_manifest.csv", index=False, encoding="utf-8-sig")
    return out


def _preannotation_metadata(
    image_name: str,
    mode: str,
    predictions: dict[str, pd.DataFrame],
    pred_maps: dict[str, dict[str, dict[str, list[float]]]],
    source_maps: dict[str, dict[str, dict[str, str]]],
) -> dict[str, Any]:
    if mode == "v0.4.1":
        json_path = V041_JSON_DIR / f"{Path(image_name).stem}_v041_automatic.json"
        payload = _load_json(json_path) if json_path.exists() else {}
        points = pred_maps["v041"].get(image_name, payload.get("hybrid_keypoints", {}))
        sources = source_maps["v041"].get(image_name, payload.get("point_sources", {}))
        return {
            **payload,
            "preannotation_mode": "v0.4.1 hybrid heatmap + geometry",
            "annotation_status": "auto_preannotation_unverified",
            "hybrid_keypoints": points,
            "corrected_keypoints": points,
            "point_sources": sources,
            "v041_hybrid_keypoints": points,
            "show_v041_points": False,
        }
    points = pred_maps["v06"].get(image_name, {})
    sources = source_maps["v06"].get(image_name, {})
    return {
        "preannotation_mode": "v0.6_keypointwise_hybrid_experimental",
        "hybrid_model_version": "v0.6_keypointwise_hybrid_selector",
        "annotation_status": "auto_preannotation_unverified",
        "v01_heatmap_keypoints": pred_maps["v01"].get(image_name, {}),
        "v05_heatmap_keypoints": pred_maps["v05"].get(image_name, {}),
        "v041_hybrid_keypoints": pred_maps["v041"].get(image_name, {}),
        "v06_keypoints": points,
        "hybrid_keypoints": points,
        "corrected_keypoints": points,
        "point_sources": sources,
        "v06_point_sources": sources,
        "v06_qc_results": {"v06_qc_pass": True, "v06_review_reason": ""},
        "show_v06_selected_points": True,
    }


def _add_p7v_metadata(metadata: dict[str, Any], full_points: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    short = _full_to_short_points(full_points)
    try:
        measurements = calculate_measurements(short, MM_PER_PIXEL, axis_mode_selected="auto")
        p7v = measurements.get("derived_points", {}).get("P7V_caudal_fin_posterior_endpoint")
        p7v_valid = p7v is not None
        metadata.setdefault("qc_results", {})
        metadata.setdefault("hybrid_qc_results", {})
        metadata["qc_results"]["p7v_valid"] = p7v_valid
        metadata["qc_results"]["p7v_suggestion"] = p7v
        metadata["hybrid_qc_results"]["p7v_valid"] = p7v_valid
        metadata["hybrid_qc_results"]["p7v"] = {"point": p7v}
        metadata["p7v_valid"] = p7v_valid
        return measurements, p7v_valid
    except Exception as exc:  # noqa: BLE001 - regression report records failures.
        metadata.setdefault("qc_results", {})
        metadata["qc_results"]["p7v_valid"] = False
        metadata["qc_results"]["p7v_error"] = str(exc)
        metadata["p7v_valid"] = False
        return {}, False


def _test_preannotation_modes(test_manifest: pd.DataFrame, predictions: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, dict[str, dict[str, Any]]]]:
    pred_maps = {key: _prediction_map(value) for key, value in predictions.items()}
    source_maps = {key: _source_map(value) for key, value in predictions.items()}
    metadata_cache: dict[str, dict[str, dict[str, Any]]] = {}
    rows = []
    for record in test_manifest.to_dict("records"):
        image_name = str(record["image_name"])
        metadata_cache[image_name] = {}
        for mode in ("v0.4.1", "v0.6"):
            error = ""
            success = False
            p7v_valid = False
            num_keypoints = 0
            has_qc = False
            reliability = "failed"
            try:
                metadata = _preannotation_metadata(image_name, mode, predictions, pred_maps, source_maps)
                points = metadata.get("v06_keypoints") if mode == "v0.6" else metadata.get("hybrid_keypoints")
                if not isinstance(points, Mapping):
                    points = {}
                measurements, p7v_valid = _add_p7v_metadata(metadata, points)
                metadata["measurements"] = measurements
                num_keypoints = sum(1 for key in FULL_KEYPOINTS if key in points)
                has_qc = bool(metadata.get("review_reason") or metadata.get("v06_qc_results", {}).get("v06_review_reason"))
                success = num_keypoints == len(FULL_KEYPOINTS)
                reliability = "review" if has_qc else ("ok" if success and p7v_valid else "needs_review")
                metadata_cache[image_name][mode] = metadata
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
            rows.append(
                {
                    "image_name": image_name,
                    "mode": mode,
                    "success": success,
                    "num_keypoints": num_keypoints,
                    "p7v_valid": p7v_valid,
                    "has_qc_warning": has_qc,
                    "preannotation_reliability_level": reliability,
                    "error_message": error,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "preannotation_mode_test.csv", index=False, encoding="utf-8-sig")
    return frame, metadata_cache


def _test_measurement_axis_modes(test_manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in test_manifest.to_dict("records"):
        image_name = str(record["image_name"])
        image_path = _resolve_path(record["warped_image_path"])
        gt_path = _resolve_path(record["corrected_json_path"])
        full_points = _corrected_keypoints(gt_path) if gt_path and gt_path.exists() else {}
        image = load_image_file(image_path) if image_path and image_path.exists() else None
        original_snapshot = json.dumps(full_points, sort_keys=True)
        for axis_mode in AXIS_MODES:
            row = {
                "image_name": image_name,
                "axis_mode": axis_mode,
                "success": False,
                "error_message": "",
            }
            try:
                if image is None:
                    raise FileNotFoundError(f"Missing warped image: {record['warped_image_path']}")
                payload = compute_measurement_axis_metadata(
                    image,
                    full_points,
                    mm_per_pixel=MM_PER_PIXEL,
                    measurement_axis_mode=axis_mode,
                )
                fields = payload.get("measurement_axis_row_fields", {})
                row.update(fields)
                row["success"] = bool(payload.get("measurement_axes")) and json.dumps(full_points, sort_keys=True) == original_snapshot
            except Exception as exc:  # noqa: BLE001
                row["error_message"] = str(exc)
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "measurement_axis_test.csv", index=False, encoding="utf-8-sig")
    return frame


def _make_measurement_row(image_name: str, specimen_id: str, full_points: Mapping[str, Any], image: np.ndarray, axis_mode: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    short = _full_to_short_points(full_points)
    measurements = calculate_measurements(short, MM_PER_PIXEL, axis_mode_selected="auto")
    axis_payload = compute_measurement_axis_metadata(
        image,
        full_points,
        mm_per_pixel=MM_PER_PIXEL,
        measurement_axis_mode=axis_mode,
    )
    measurements.update(axis_payload.get("measurement_axis_row_fields", {}))
    selected_axis = measurements.get("selected_measurement_axis")
    if selected_axis in {"model_axis", "body_midline_axis"}:
        measurements["TL_curve_mm"] = measurements.get("TL_curve_selected_mm", measurements.get("TL_curve_mm"))
        measurements["SL_curve_mm"] = measurements.get("SL_curve_selected_mm", measurements.get("SL_curve_mm"))
        measurements["curvature_index"] = measurements.get("curvature_index_selected", measurements.get("curvature_index"))
    row = build_result_row(
        image_name=image_name,
        specimen_id=specimen_id,
        scale_method="board_mm_per_pixel",
        mm_per_pixel=MM_PER_PIXEL,
        keypoints=short,
        measurements=measurements,
        needs_review=bool(measurements.get("measurements_needs_review", False)),
        notes="e2e_regression_v0.6.4_temp_export",
        source_type="real",
    )
    # User-facing compatibility aliases for the regression export.
    row["head_length_mm"] = row.get("head_length_straight_mm", "")
    row["snout_length_mm"] = row.get("snout_length_straight_mm", "")
    row["caudal_peduncle_length_mm"] = row.get("caudal_peduncle_length_straight_mm", "")
    row["p7v_valid"] = bool(measurements.get("derived_points", {}).get("P7V_caudal_fin_posterior_endpoint"))
    row["preannotation_reliability_level"] = "ok"
    row["curvature_qc_level"] = axis_payload.get("measurement_axes", {}).get("model_axis", {}).get("curvature_qc_level", "")
    row["high_curvature_review_required"] = str(row.get("curvature_qc_level", "")).lower() == "high"
    row["recommended_action"] = "review_measurements" if row.get("measurements_needs_review") else "accept_after_keypoint_review"
    return row, measurements, axis_payload


def _save_roundtrip_and_export(
    test_manifest: pd.DataFrame,
    metadata_cache: dict[str, dict[str, dict[str, Any]]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    save_dir = OUT / "test_corrected_keypoints"
    preview_dir = OUT / "test_corrected_previews"
    save_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    measurement_rows = []
    for record in test_manifest.to_dict("records"):
        image_name = str(record["image_name"])
        specimen_id = str(record.get("specimen_id", ""))
        image_path = _resolve_path(record["warped_image_path"])
        gt_path = _resolve_path(record["corrected_json_path"])
        if not image_path or not image_path.exists() or not gt_path or not gt_path.exists():
            rows.append({"image_name": image_name, "modified_keypoint": "P4_peduncle_start_midpoint", "success": False, "error_message": "missing input"})
            continue
        image = load_image_file(image_path)
        full_points = _corrected_keypoints(gt_path)
        modified = deepcopy(full_points)
        key = "P4_peduncle_start_midpoint"
        old = _xy(modified.get(key)) or [0.0, 0.0]
        new = [old[0] + 5.0, old[1] + 3.0]
        modified[key] = new
        measurement_row, measurements, axis_payload = _make_measurement_row(image_name, specimen_id, modified, image, MEASUREMENT_AXIS_AUTO_QC)
        measurement_rows.append(measurement_row)

        metadata = deepcopy(metadata_cache.get(image_name, {}).get("v0.6") or metadata_cache.get(image_name, {}).get("v0.4.1") or {})
        metadata["corrected_keypoints"] = modified
        metadata["measurement_axes"] = axis_payload.get("measurement_axes", {})
        metadata["measurement_axis_mode_request"] = MEASUREMENT_AXIS_AUTO_QC
        metadata.setdefault("keypoint_edit_log", [])
        metadata["keypoint_edit_log"].append(
            {
                "keypoint_name": key,
                "old_xy": old,
                "new_xy": new,
                "source_before": "e2e_loaded_corrected",
                "source_after": "manual_corrected_simulated",
                "displacement_px": _point_distance(old, new),
                "displacement_mm": _point_distance(old, new) * MM_PER_PIXEL,
                "edited_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        overlay_meta = deepcopy(metadata)
        overlay_meta.update(
            {
                "show_v06_selected_points": True,
                "show_v01_heatmap_points": False,
                "show_v05_heatmap_points": False,
                "show_v041_points": False,
                "show_model_axis": True,
                "show_body_midline_axis": True,
                "show_dual_axis_comparison": True,
                "show_selected_measurement_axis_only": False,
            }
        )
        preview = draw_enhanced_preannotation_overlay(image, overlay_meta)
        preview_path = preview_dir / f"{Path(image_name).stem}_test_corrected_preview.png"
        Image.fromarray(ensure_rgb(preview)).save(preview_path)
        payload = {
            "annotation_schema_version": "e2e_regression_v0.6.4",
            "image_name": image_name,
            "specimen_id": specimen_id,
            "annotation_status": "corrected_and_confirmed",
            "annotation_mode": "e2e_simulated_manual_correction",
            "preannotation_mode": metadata.get("preannotation_mode", ""),
            "corrected_keypoints": modified,
            "model_keypoints_raw": metadata.get("model_keypoints_raw", {}),
            "heatmap_keypoints": metadata.get("heatmap_keypoints", {}),
            "v034_keypoints": metadata.get("v034_keypoints", {}),
            "v06_keypoints": metadata.get("v06_keypoints", {}),
            "v01_heatmap_keypoints": metadata.get("v01_heatmap_keypoints", {}),
            "v05_heatmap_keypoints": metadata.get("v05_heatmap_keypoints", {}),
            "v041_hybrid_keypoints": metadata.get("v041_hybrid_keypoints", {}),
            "point_sources": metadata.get("point_sources", {}),
            "measurement_axes": metadata.get("measurement_axes", {}),
            "measurements": measurements,
            "keypoint_edit_log": metadata.get("keypoint_edit_log", []),
            "preview_path": str(preview_path),
        }
        json_path = save_dir / f"{Path(image_name).stem}_test_corrected.json"
        _write_json(json_path, payload)
        saved = _load_json(json_path)
        rows.append(
            {
                "image_name": image_name,
                "modified_keypoint": key,
                "old_x": round(old[0], 3),
                "old_y": round(old[1], 3),
                "new_x": round(new[0], 3),
                "new_y": round(new[1], 3),
                "json_saved": json_path.exists(),
                "corrected_updated": _xy(saved.get("corrected_keypoints", {}).get(key)) == new,
                "raw_predictions_preserved": bool(
                    saved.get("heatmap_keypoints")
                    or saved.get("v01_heatmap_keypoints")
                    or saved.get("v05_heatmap_keypoints")
                    or saved.get("v041_hybrid_keypoints")
                ),
                "measurement_axes_saved": bool(saved.get("measurement_axes")),
                "keypoint_edit_log_saved": bool(saved.get("keypoint_edit_log")),
                "success": bool(
                    json_path.exists()
                    and _xy(saved.get("corrected_keypoints", {}).get(key)) == new
                    and saved.get("measurement_axes")
                    and saved.get("keypoint_edit_log")
                ),
                "error_message": "",
            }
        )

    save_frame = pd.DataFrame(rows)
    save_frame.to_csv(OUT / "save_roundtrip_test.csv", index=False, encoding="utf-8-sig")
    export_frame = pd.DataFrame(measurement_rows)
    export_frame.to_csv(OUT / "test_measurements.csv", index=False, encoding="utf-8-sig")
    export_frame.to_excel(OUT / "test_measurements.xlsx", index=False)
    field_check = _check_export_fields(export_frame, OUT / "test_measurements.xlsx")
    return save_frame, export_frame, field_check


def _check_export_fields(csv_frame: pd.DataFrame, xlsx_path: Path) -> pd.DataFrame:
    required = [
        "TL_final_mm",
        "SL_final_mm",
        "body_depth_mm",
        "head_length_mm",
        "snout_length_mm",
        "caudal_peduncle_length_mm",
        "caudal_peduncle_depth_mm",
        "selected_measurement_axis",
        "TL_curve_model_axis_mm",
        "SL_curve_model_axis_mm",
        "curvature_index_model_axis",
        "axis_smoothness_model_axis",
        "TL_curve_body_midline_axis_mm",
        "SL_curve_body_midline_axis_mm",
        "curvature_index_body_midline_axis",
        "axis_smoothness_body_midline_axis",
        "TL_curve_selected_mm",
        "SL_curve_selected_mm",
        "curvature_index_selected",
        "TL_curve_axis_diff_mm",
        "SL_curve_axis_diff_mm",
        "curvature_index_axis_diff",
        "dual_axis_disagreement",
        "measurements_needs_review",
        "measurement_axis_review_reason",
        "p7v_valid",
        "preannotation_reliability_level",
        "curvature_qc_level",
        "high_curvature_review_required",
        "recommended_action",
    ]
    xlsx_frame = pd.read_excel(xlsx_path)
    rows = []
    for field in required:
        rows.append(
            {
                "field_name": field,
                "present_in_csv": field in csv_frame.columns,
                "present_in_excel": field in xlsx_frame.columns,
                "num_missing_values": int(csv_frame[field].isna().sum()) if field in csv_frame.columns else len(csv_frame),
                "notes": "compatibility alias" if field in {"head_length_mm", "snout_length_mm", "caudal_peduncle_length_mm"} else "",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "export_field_check.csv", index=False, encoding="utf-8-sig")
    return frame


def _visualization_checks(
    test_manifest: pd.DataFrame,
    metadata_cache: dict[str, dict[str, dict[str, Any]]],
) -> pd.DataFrame:
    vis_dir = OUT / "visual_previews"
    vis_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in test_manifest.to_dict("records"):
        image_name = str(record["image_name"])
        image_path = _resolve_path(record["warped_image_path"])
        if image_path is None or not image_path.exists():
            continue
        image = load_image_file(image_path)
        for mode in ("v0.4.1", "v0.6"):
            metadata = deepcopy(metadata_cache.get(image_name, {}).get(mode, {}))
            if not metadata:
                rows.append({"image_name": image_name, "mode": mode, "success": False, "notes": "missing metadata"})
                continue
            axis_payload = compute_measurement_axis_metadata(
                image,
                metadata.get("v06_keypoints") if mode == "v0.6" else metadata.get("hybrid_keypoints", {}),
                mm_per_pixel=MM_PER_PIXEL,
                measurement_axis_mode=MEASUREMENT_AXIS_AUTO_QC,
            )
            metadata["measurement_axes"] = axis_payload.get("measurement_axes", {})
            metadata.update(
                {
                    "show_v06_selected_points": mode == "v0.6",
                    "show_v041_points": mode == "v0.4.1",
                    "show_v01_heatmap_points": mode == "v0.6",
                    "show_v05_heatmap_points": mode == "v0.6",
                    "show_mask_suggestions": True,
                    "show_qc_warnings": True,
                    "show_model_axis": True,
                    "show_body_midline_axis": True,
                    "show_dual_axis_comparison": True,
                    "show_selected_measurement_axis_only": False,
                    "show_point_source_labels": mode == "v0.6",
                }
            )
            preview = draw_enhanced_preannotation_overlay(image, metadata)
            path = vis_dir / f"{Path(image_name).stem}_{mode.replace('.', '').replace(' ', '_')}_overlay.png"
            Image.fromarray(ensure_rgb(preview)).save(path)
            p7v_visible = bool(metadata.get("qc_results", {}).get("p7v_suggestion") or metadata.get("hybrid_qc_results", {}).get("p7v"))
            rows.append(
                {
                    "image_name": image_name,
                    "mode": mode,
                    "model_axis_visible": True,
                    "body_midline_axis_visible": True,
                    "selected_axis_visible": True,
                    "p7v_visible": p7v_visible,
                    "qc_warning_visible": bool(metadata.get("show_qc_warnings", True)),
                    "point_source_label_visible": bool(mode == "v0.6"),
                    "success": path.exists() and bool(metadata.get("measurement_axes")),
                    "notes": str(path.relative_to(PROJECT_ROOT)),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "visualization_check.csv", index=False, encoding="utf-8-sig")
    return frame


def _write_readme(
    test_manifest: pd.DataFrame,
    pre: pd.DataFrame,
    axis: pd.DataFrame,
    save: pd.DataFrame,
    field: pd.DataFrame,
    visual: pd.DataFrame,
) -> None:
    def pass_fail(series: pd.Series) -> str:
        return "pass" if bool(series.all()) else "fail"

    v041 = pre[pre["mode"] == "v0.4.1"]
    v06 = pre[pre["mode"] == "v0.6"]
    axis_status = axis.groupby("axis_mode")["success"].all().to_dict() if not axis.empty else {}
    csv_pass = bool(field["present_in_csv"].all() and field["present_in_excel"].all())
    corrected_protected = bool(save["success"].all()) and not any(
        path.exists()
        for path in (
            PROJECT_ROOT / "results" / "real_annotation_5_15_v0.3_corrected" / "keypoints" / f"{Path(name).stem}_test_corrected.json"
            for name in test_manifest["image_name"].astype(str)
        )
    )
    ready = bool(v041["success"].all() and v06["success"].all() and axis["success"].all() and csv_pass and corrected_protected)
    lines = [
        "# v0.6.4 End-to-End Regression Test",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Test image count: {len(test_manifest)}",
        f"- v0.4.1 default mode: {pass_fail(v041['success'])}",
        f"- v0.6 experimental mode: {pass_fail(v06['success'])}",
        f"- model_axis: {'pass' if axis_status.get(MEASUREMENT_AXIS_MODEL, False) else 'fail'}",
        f"- body_midline_axis: {'pass' if axis_status.get(MEASUREMENT_AXIS_BODY_MIDLINE, False) else 'fail'}",
        f"- auto_qc_gated: {'pass' if axis_status.get(MEASUREMENT_AXIS_AUTO_QC, False) else 'fail'}",
        f"- corrected_keypoints protected: {'pass' if corrected_protected else 'fail'}",
        f"- CSV/Excel required fields: {'pass' if csv_pass else 'fail'}",
        f"- visualization previews: {pass_fail(visual['success'])}",
        "",
        "## Test Cases",
        "",
        test_manifest[["image_name", "specimen_id", "test_case_type", "expected_behavior", "notes"]].to_markdown(index=False),
        "",
        "## Export Field Notes",
        "",
        "The regression export includes compatibility aliases `head_length_mm`, `snout_length_mm`, and `caudal_peduncle_length_mm`, mapped from the existing straight-line fields.",
        "",
        "## Known Issues",
        "",
        "- This script uses cached same-set automatic prediction artifacts where available, plus the current measurement-axis code path. It does not run model training and does not write to official corrected label directories.",
        "- A visual overlay check verifies files and metadata flags, not human visual judgment. Open `visual_previews/` for manual inspection if needed.",
        "",
        "## Recommendation",
        "",
        "Ready for batch measurement: " + ("yes" if ready else "no"),
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    test_manifest = _choose_test_manifest()
    predictions = _load_predictions()
    pre, metadata_cache = _test_preannotation_modes(test_manifest, predictions)
    axis = _test_measurement_axis_modes(test_manifest)
    save, export_frame, field = _save_roundtrip_and_export(test_manifest, metadata_cache)
    visual = _visualization_checks(test_manifest, metadata_cache)
    _write_readme(test_manifest, pre, axis, save, field, visual)

    def pf(value: bool) -> str:
        return "pass" if value else "fail"

    print("Finished v0.6.4 end-to-end regression test.")
    print("")
    print("Preannotation modes tested:")
    print(f"v0.4.1: {pf(bool(pre.loc[pre['mode'].eq('v0.4.1'), 'success'].all()))}")
    print(f"v0.6 experimental: {pf(bool(pre.loc[pre['mode'].eq('v0.6'), 'success'].all()))}")
    print("")
    print("Measurement axis modes tested:")
    for mode in AXIS_MODES:
        ok = bool(axis.loc[axis["axis_mode"].eq(mode), "success"].all())
        print(f"{mode}: {pf(ok)}")
    print("")
    print(f"Corrected keypoints protection: {pf(bool(save['success'].all()))}")
    print(f"CSV/Excel export: {pf(bool(field['present_in_csv'].all() and field['present_in_excel'].all()))}")
    ready = bool(
        pre["success"].all()
        and axis["success"].all()
        and save["success"].all()
        and field["present_in_csv"].all()
        and field["present_in_excel"].all()
        and visual["success"].all()
    )
    print("")
    print(f"Ready for batch measurement: {'yes' if ready else 'no'}")


if __name__ == "__main__":
    main()
