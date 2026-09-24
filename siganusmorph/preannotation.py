"""Automatic preannotation helpers for the local Streamlit app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import cv2

from .caudal_base_geometry import estimate_caudal_base_transition
from .config import KEYPOINT_DEFS
from .centerline_geometry import estimate_axis_points_from_mask
from .eye_geometry import estimate_eye_front_from_head_roi
from .geometric_rules import generate_mask_suggestions, run_preannotation_qc
from .image_utils import ensure_rgb
from .mask_keypoint_correction import correct_keypoints_with_mask
from .operculum_geometry import estimate_operculum_posterior_edge
from .segmentation import padded_bbox, save_mask_png, save_segmentation_preview, segment_fish_from_blue_board
from .tail_geometry import apply_tail_geometry_rules

_MODEL_CACHE: dict[str, Any] = {}


def clear_preannotation_model_cache(model_path: str | Path | None = None) -> None:
    """Release cached YOLO models without affecting materialized predictions."""
    if model_path is None:
        _MODEL_CACHE.clear()
        return
    _MODEL_CACHE.pop(str(Path(model_path).resolve()), None)


def default_preannotation_model(project_root: Path) -> Path:
    """Return the preferred local preannotation model path."""
    candidates = [
        project_root
        / "models"
        / "siganusmorph_yolopose_v0.3_corrected_real_5_15"
        / "preannotation_candidate.pt",
        project_root
        / "models"
        / "siganusmorph_yolopose_v0.2_maskcrop_real_5_15"
        / "preannotation_candidate.pt",
        project_root
        / "models"
        / "siganusmorph_yolopose_v0.1_real_5_15"
        / "weights"
        / "preannotation_candidate.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _quality_text(quality: dict[str, Any]) -> str:
    if not quality:
        return "failed"
    status = str(quality.get("status", "") or quality.get("quality", "") or "")
    if status:
        return status
    if quality.get("success") or quality.get("segmentation_success"):
        return "ok"
    return "failed"


def _clip_bbox(bbox: tuple[int, int, int, int] | None, shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    if bbox is None:
        return 0, 0, width, height
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def _predict_crop_keypoints(crop: np.ndarray, model_path: Path, imgsz: int = 1024) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - optional dependency UI path
        raise RuntimeError("ultralytics is required for automatic preannotation") from exc

    cache_key = str(Path(model_path).resolve())
    model = _MODEL_CACHE.get(cache_key)
    if model is None:
        model = YOLO(str(model_path))
        _MODEL_CACHE[cache_key] = model
    # Ultralytics treats numpy image inputs as OpenCV-style BGR arrays.
    bgr_crop = cv2.cvtColor(ensure_rgb(crop), cv2.COLOR_RGB2BGR)
    results = model.predict(
        bgr_crop,
        imgsz=imgsz,
        conf=0.05,
        max_det=1,
        verbose=False,
    )
    if not results:
        raise RuntimeError("model returned no prediction result")
    result = results[0]
    if result.keypoints is None or len(result.keypoints) == 0:
        raise RuntimeError("model did not detect keypoints")

    if result.boxes is not None and len(result.boxes) > 0 and hasattr(result.boxes, "conf"):
        det_index = int(np.argmax(result.boxes.conf.cpu().numpy()))
    else:
        det_index = 0

    xy = result.keypoints.xy.cpu().numpy()[det_index]
    if getattr(result.keypoints, "conf", None) is not None:
        conf_array = result.keypoints.conf.cpu().numpy()[det_index]
    else:
        conf_array = np.ones((len(KEYPOINT_DEFS),), dtype=float)

    keypoints: dict[str, tuple[float, float]] = {}
    confidences: dict[str, float] = {}
    for index, definition in enumerate(KEYPOINT_DEFS):
        if index >= len(xy):
            continue
        key = f"{definition.code}_{definition.name}"
        keypoints[key] = (float(xy[index][0]), float(xy[index][1]))
        confidences[key] = float(conf_array[index]) if index < len(conf_array) else 1.0
    if len(keypoints) != len(KEYPOINT_DEFS):
        raise RuntimeError(f"expected {len(KEYPOINT_DEFS)} keypoints, got {len(keypoints)}")
    return keypoints, confidences


def _merge_reason_text(*values: Any) -> str:
    reasons: list[str] = []
    for value in values:
        if not value:
            continue
        for part in str(value).split(";"):
            part = part.strip()
            if part:
                reasons.append(part)
    return ";".join(dict.fromkeys(reasons))


def _save_tail_detail_preview(image: np.ndarray, mask: np.ndarray | None, output_path: Path) -> Path | None:
    if mask is None:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = ensure_rgb(image)
    overlay = rgb.copy()
    mask_bool = mask > 0
    overlay[mask_bool] = (0.55 * overlay[mask_bool] + 0.45 * np.array([255, 80, 220])).astype(np.uint8)
    cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return output_path


def _xy(value: Any) -> tuple[float, float] | None:
    if isinstance(value, dict) and "x" in value and "y" in value:
        return float(value["x"]), float(value["y"])
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return None


def _dist(a: Any, b: Any) -> float | None:
    pa = _xy(a)
    pb = _xy(b)
    if pa is None or pb is None:
        return None
    return float(((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5)


def _local_structure_qc(
    *,
    corrected: dict[str, Any],
    local_suggestions: dict[str, Any],
    fish_bbox: tuple[int, int, int, int],
) -> dict[str, Any]:
    width = max(1, fish_bbox[2] - fish_bbox[0])
    thresholds = {
        "P2_eye_front": max(30.0, 0.025 * width),
        "P3_operculum_posterior": max(35.0, 0.030 * width),
        "P5_caudal_base_midpoint": max(45.0, 0.040 * width),
    }
    reasons: list[str] = []
    distances: dict[str, float] = {}
    for key, suggestion in local_suggestions.items():
        distance = _dist(corrected.get(key), suggestion)
        if distance is None:
            continue
        distances[key] = distance
        if distance > thresholds.get(key, 999999.0):
            if key == "P2_eye_front":
                reasons.append("P2_model_differs_from_eye_detection")
            elif key == "P3_operculum_posterior":
                reasons.append("P3_model_differs_from_operculum_edge")
            elif key == "P5_caudal_base_midpoint":
                reasons.append("P5_model_differs_from_tail_transition")
    return {
        "local_structure_qc_pass": not reasons,
        "local_structure_review_reason": ";".join(dict.fromkeys(reasons)),
        "local_structure_distances_px": distances,
    }


def preannotate_warped_image(
    warped_image: np.ndarray,
    image_name: str,
    project_root: Path,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Segment fish, predict keypoints on mask crop, and correct P1 with mask."""
    rgb = ensure_rgb(warped_image)
    selected_model = Path(model_path) if model_path else default_preannotation_model(project_root)
    if not selected_model.exists():
        raise FileNotFoundError(f"preannotation model not found: {selected_model}")

    fish_mask, fish_bbox, quality = segment_fish_from_blue_board(rgb)
    segmentation_success = bool(quality.get("segmentation_success", quality.get("success", False))) and fish_bbox is not None
    bbox = _clip_bbox(padded_bbox(fish_bbox, rgb.shape, padding_ratio=0.08) if fish_bbox else None, rgb.shape)
    if not segmentation_success:
        bbox = _clip_bbox(None, rgb.shape)

    x1, y1, x2, y2 = bbox
    crop = rgb[y1:y2, x1:x2]
    crop_keypoints, confidences = _predict_crop_keypoints(crop, selected_model)
    warped_keypoints = {
        key: (float(point[0]) + x1, float(point[1]) + y1)
        for key, point in crop_keypoints.items()
    }

    if segmentation_success:
        corrected, correction_log = correct_keypoints_with_mask(
            warped_keypoints,
            fish_mask,
            bbox,
            confidences,
            force_p1_mask=True,
        )
        mask_suggestions = generate_mask_suggestions(fish_mask, bbox, warped_keypoints)
    else:
        corrected = dict(warped_keypoints)
        correction_log = {
            "p1_source": "model",
            "p1_correction_applied": False,
            "review_reason": "segmentation_failed",
        }
        mask_suggestions = {}

    tail_result = apply_tail_geometry_rules(
        warped_image=rgb,
        fish_mask=fish_mask,
        fish_bbox=bbox,
        corrected_keypoints=corrected,
        model_keypoints_raw=warped_keypoints,
        confidences=confidences,
        segmentation_success=segmentation_success,
    )
    tail_detail_mask = tail_result.pop("tail_detail_mask", None)
    corrected = dict(tail_result.get("corrected_keypoints", corrected))
    tail_geometry_qc = dict(tail_result.get("tail_geometry_qc", {}) or {})
    eye_result = estimate_eye_front_from_head_roi(rgb, fish_mask, corrected, warped_keypoints)
    operculum_result = estimate_operculum_posterior_edge(rgb, fish_mask, corrected, warped_keypoints)
    tail_axis_vector = tail_geometry_qc.get("tail_axis_vector")
    caudal_base_result = estimate_caudal_base_transition(fish_mask, tail_detail_mask, corrected, tail_axis_vector)
    local_structure_suggestions: dict[str, Any] = {}
    if eye_result.get("P2_eye_suggestion") is not None:
        local_structure_suggestions["P2_eye_front"] = eye_result["P2_eye_suggestion"]
    if operculum_result.get("P3_edge_suggestion") is not None:
        local_structure_suggestions["P3_operculum_posterior"] = operculum_result["P3_edge_suggestion"]
    if caudal_base_result.get("P5_transition_suggestion") is not None:
        local_structure_suggestions["P5_caudal_base_midpoint"] = caudal_base_result["P5_transition_suggestion"]
    local_structure_qc = _local_structure_qc(
        corrected=corrected,
        local_suggestions=local_structure_suggestions,
        fish_bbox=bbox,
    )
    centerline_result = estimate_axis_points_from_mask(fish_mask, corrected)
    if centerline_result.get("point_sources"):
        for key, source in dict(centerline_result.get("point_sources", {}) or {}).items():
            if key in centerline_result.get("mask_suggestions", {}):
                corrected[key] = centerline_result["mask_suggestions"][key]
    merged_suggestions = dict(mask_suggestions)
    merged_suggestions.update(tail_result.get("mask_suggestions", {}) or {})
    merged_suggestions.update(local_structure_suggestions)
    merged_suggestions.update(centerline_result.get("mask_suggestions", {}) or {})
    mask_suggestions = merged_suggestions
    qc_results = run_preannotation_qc(corrected, mask_suggestions, bbox, segmentation_success)
    qc_flagged = qc_results.get("flagged_keypoints", {})
    if not isinstance(qc_flagged, dict):
        qc_flagged = {}
    qc_flagged.update(tail_result.get("flagged_keypoints", {}) or {})
    qc_results["flagged_keypoints"] = qc_flagged
    body_depth_qc = dict(tail_result.get("body_depth_qc", {}) or {})
    peduncle_depth_qc = dict(tail_result.get("peduncle_depth_qc", {}) or {})
    axis_qc = dict(tail_result.get("axis_qc", {}) or {})
    centerline_axis_qc = dict(centerline_result.get("axis_qc", {}) or {})
    if centerline_axis_qc:
        axis_qc.update(centerline_axis_qc)
    local_reason = local_structure_qc.get("local_structure_review_reason", "")
    if local_reason:
        qc_results["review_reason"] = _merge_reason_text(qc_results.get("review_reason", ""), local_reason)
    qc_results.update(
        {
            "tail_qc_pass": tail_geometry_qc.get("tail_qc_pass", ""),
            "P5_qc_pass": tail_geometry_qc.get("P5_qc_pass", ""),
            "P6_qc_pass": tail_geometry_qc.get("P6_qc_pass", ""),
            "p7v_valid": tail_geometry_qc.get("p7v_valid", qc_results.get("p7v_valid", True)),
            "p7v_suggestion": tail_geometry_qc.get("p7v_suggestion", qc_results.get("p7v_suggestion")),
            "body_depth_qc_pass": body_depth_qc.get("body_depth_qc_pass", ""),
            "peduncle_depth_qc_pass": peduncle_depth_qc.get("peduncle_depth_qc_pass", ""),
            "axis_qc_pass": axis_qc.get("axis_qc_pass", ""),
            "tail_axis_source": tail_geometry_qc.get("tail_axis_source", ""),
            "tail_mask_quality": tail_geometry_qc.get("tail_mask_quality", ""),
            "local_structure_qc_pass": local_structure_qc.get("local_structure_qc_pass", ""),
        }
    )
    merged_review_reason = _merge_reason_text(qc_results.get("review_reason", ""), tail_result.get("review_reason", ""), local_reason)
    qc_results["review_reason"] = merged_review_reason
    qc_results["needs_review"] = bool(merged_review_reason)

    output_root = project_root / "results"
    mask_dir = output_root / "segmentation_masks"
    preview_dir = output_root / "segmentation_previews"
    tail_detail_mask_dir = output_root / "tail_detail_masks"
    tail_detail_preview_dir = output_root / "tail_detail_previews"
    stem = Path(image_name).stem
    mask_path = save_mask_png(fish_mask, mask_dir / f"{stem}_fish_mask.png")
    preview_path = save_segmentation_preview(
        rgb,
        fish_mask,
        bbox if segmentation_success else None,
        quality,
        preview_dir / f"{stem}_segmentation_preview.png",
    )
    tail_detail_mask_path = save_mask_png(tail_detail_mask, tail_detail_mask_dir / f"{stem}_tail_detail_mask.png") if tail_detail_mask is not None else None
    tail_detail_preview_path = _save_tail_detail_preview(
        rgb,
        tail_detail_mask,
        tail_detail_preview_dir / f"{stem}_tail_detail_preview.png",
    )

    p1_source = "mask_left_boundary" if bool(correction_log.get("p1_correction_applied", False)) else correction_log.get("p1_source", "model")
    point_sources = dict(tail_result.get("point_sources", {}) or {})
    point_sources.update(centerline_result.get("point_sources", {}) or {})
    if p1_source == "mask_left_boundary":
        point_sources["P1_snout_tip"] = "mask_left_boundary"

    metadata = {
        "enhanced_preannotation_version": "v0.3.4_local_structure_refinement",
        "segmentation_success": segmentation_success,
        "segmentation_quality": _quality_text(quality),
        "fish_bbox_x1": x1,
        "fish_bbox_y1": y1,
        "fish_bbox_x2": x2,
        "fish_bbox_y2": y2,
        "mask_area_px": int(np.count_nonzero(fish_mask)),
        "p1_source": p1_source,
        "p1_model_x": correction_log.get("p1_model_x", ""),
        "p1_model_y": correction_log.get("p1_model_y", ""),
        "p1_mask_x": correction_log.get("p1_mask_x", ""),
        "p1_mask_y": correction_log.get("p1_mask_y", ""),
        "p1_correction_applied": bool(correction_log.get("p1_correction_applied", False)),
        "keypoint_correction_log": json.dumps(
            {
                "p1": correction_log,
                "tail_geometry": tail_result,
                "eye_detection": eye_result,
                "operculum_edge_detection": operculum_result,
                "caudal_base_transition": caudal_base_result,
                "local_structure_qc": local_structure_qc,
                "qc": qc_results,
            },
            ensure_ascii=False,
        ),
        "model_keypoints_raw": dict(warped_keypoints),
        "mask_suggestions": dict(mask_suggestions),
        "geometric_suggestions": dict(tail_result.get("geometric_suggestions", {}) or {}),
        "local_structure_suggestions": dict(local_structure_suggestions),
        "eye_detection": eye_result,
        "operculum_edge_detection": operculum_result,
        "caudal_base_transition": caudal_base_result,
        "local_structure_qc": local_structure_qc,
        "centerline_suggestions": dict(centerline_result.get("mask_suggestions", {}) or {}),
        "point_sources": point_sources,
        "tail_geometry_qc": tail_geometry_qc,
        "body_depth_qc": body_depth_qc,
        "peduncle_depth_qc": peduncle_depth_qc,
        "axis_qc": axis_qc,
        "corrected_keypoints": dict(corrected),
        "auto_preannotation_initial_corrected_keypoints": dict(corrected),
        "annotation_status": "auto_preannotation_unverified",
        "keypoint_edit_log": [
            {
                "event": "enhanced_preannotation_created",
                "version": "v0.3.4_local_structure_refinement",
                "model_path": str(selected_model),
            }
        ],
        "qc_results": qc_results,
        "model_path": str(selected_model),
        "mask_path": str(mask_path),
        "segmentation_preview_path": str(preview_path),
        "tail_detail_mask_path": str(tail_detail_mask_path) if tail_detail_mask_path else "",
        "tail_detail_preview_path": str(tail_detail_preview_path) if tail_detail_preview_path else "",
        "needs_review": bool(qc_results.get("needs_review", False)),
        "review_reason": str(qc_results.get("review_reason", "")),
        "P7U_source": tail_geometry_qc.get("P7U_source", ""),
        "P7L_source": tail_geometry_qc.get("P7L_source", ""),
        "P6_source": tail_geometry_qc.get("P6_source", ""),
        "P6_mask_x": (tail_geometry_qc.get("P6_mask") or ["", ""])[0] if isinstance(tail_geometry_qc.get("P6_mask"), list) else "",
        "P6_mask_y": (tail_geometry_qc.get("P6_mask") or ["", ""])[1] if isinstance(tail_geometry_qc.get("P6_mask"), list) else "",
        "P6_mask_quality": tail_geometry_qc.get("P6_mask_quality", ""),
        "P6_gap_x": (tail_geometry_qc.get("P6_gap") or ["", ""])[0] if isinstance(tail_geometry_qc.get("P6_gap"), list) else "",
        "P6_gap_y": (tail_geometry_qc.get("P6_gap") or ["", ""])[1] if isinstance(tail_geometry_qc.get("P6_gap"), list) else "",
        "P6_gap_quality": tail_geometry_qc.get("P6_gap_quality", ""),
        "P6_concavity_x": (tail_geometry_qc.get("P6_concavity") or ["", ""])[0] if isinstance(tail_geometry_qc.get("P6_concavity"), list) else "",
        "P6_concavity_y": (tail_geometry_qc.get("P6_concavity") or ["", ""])[1] if isinstance(tail_geometry_qc.get("P6_concavity"), list) else "",
        "P6_concavity_quality": tail_geometry_qc.get("P6_concavity_quality", ""),
        "P6_final_x": (tail_geometry_qc.get("P6_final") or ["", ""])[0] if isinstance(tail_geometry_qc.get("P6_final"), list) else "",
        "P6_final_y": (tail_geometry_qc.get("P6_final") or ["", ""])[1] if isinstance(tail_geometry_qc.get("P6_final"), list) else "",
        "P6_review_reason": tail_geometry_qc.get("P6_review_reason", ""),
        "tail_detail_mask_quality": tail_geometry_qc.get("tail_detail_mask_quality", ""),
        "tail_mask_gap_visible": tail_geometry_qc.get("tail_mask_gap_visible", ""),
        "tail_mask_gap_filled": tail_geometry_qc.get("tail_mask_gap_filled", ""),
        "P7U_mask_x": (tail_geometry_qc.get("P7U_mask") or ["", ""])[0] if isinstance(tail_geometry_qc.get("P7U_mask"), list) else "",
        "P7U_mask_y": (tail_geometry_qc.get("P7U_mask") or ["", ""])[1] if isinstance(tail_geometry_qc.get("P7U_mask"), list) else "",
        "P7L_mask_x": (tail_geometry_qc.get("P7L_mask") or ["", ""])[0] if isinstance(tail_geometry_qc.get("P7L_mask"), list) else "",
        "P7L_mask_y": (tail_geometry_qc.get("P7L_mask") or ["", ""])[1] if isinstance(tail_geometry_qc.get("P7L_mask"), list) else "",
        "tail_mask_quality": tail_geometry_qc.get("tail_mask_quality", ""),
        "tail_axis_source": tail_geometry_qc.get("tail_axis_source", ""),
        "tail_axis_used_without_P6": tail_geometry_qc.get("tail_axis_used_without_P6", ""),
        "tail_axis_used_without_model_P6": tail_geometry_qc.get("tail_axis_used_without_model_P6", ""),
        "tail_axis_dx": tail_geometry_qc.get("tail_axis_dx", ""),
        "tail_axis_dy": tail_geometry_qc.get("tail_axis_dy", ""),
        "tail_region_bbox": tail_geometry_qc.get("tail_region_bbox", ""),
        "P7U_candidate_count": tail_geometry_qc.get("P7U_candidate_count", ""),
        "P7U_candidate_method": tail_geometry_qc.get("P7U_candidate_method", ""),
        "P7U_mask_quality": tail_geometry_qc.get("P7U_mask_quality", ""),
        "P7U_review_reason": tail_geometry_qc.get("P7U_review_reason", ""),
        "p7v_valid": tail_geometry_qc.get("p7v_valid", ""),
        "p7v_invalid_reason": tail_geometry_qc.get("p7v_review_reason", ""),
        "P2_eye_x": (eye_result.get("P2_eye_suggestion") or ["", ""])[0] if isinstance(eye_result.get("P2_eye_suggestion"), (list, tuple)) else "",
        "P2_eye_y": (eye_result.get("P2_eye_suggestion") or ["", ""])[1] if isinstance(eye_result.get("P2_eye_suggestion"), (list, tuple)) else "",
        "P2_eye_quality": eye_result.get("P2_eye_quality", ""),
        "P2_eye_bbox": eye_result.get("P2_eye_bbox", ""),
        "P2_eye_candidate_count": eye_result.get("P2_eye_candidate_count", ""),
        "P2_source_suggestion": eye_result.get("P2_source_suggestion", ""),
        "P2_model_to_eye_distance_mm": "",
        "P3_edge_x": (operculum_result.get("P3_edge_suggestion") or ["", ""])[0] if isinstance(operculum_result.get("P3_edge_suggestion"), (list, tuple)) else "",
        "P3_edge_y": (operculum_result.get("P3_edge_suggestion") or ["", ""])[1] if isinstance(operculum_result.get("P3_edge_suggestion"), (list, tuple)) else "",
        "P3_edge_quality": operculum_result.get("P3_edge_quality", ""),
        "P3_edge_strength": operculum_result.get("P3_edge_strength", ""),
        "P3_edge_method": operculum_result.get("P3_edge_method", ""),
        "P3_model_to_edge_distance_mm": "",
        "P5_transition_x": (caudal_base_result.get("P5_transition_suggestion") or ["", ""])[0] if isinstance(caudal_base_result.get("P5_transition_suggestion"), (list, tuple)) else "",
        "P5_transition_y": (caudal_base_result.get("P5_transition_suggestion") or ["", ""])[1] if isinstance(caudal_base_result.get("P5_transition_suggestion"), (list, tuple)) else "",
        "P5_transition_quality": caudal_base_result.get("P5_transition_quality", ""),
        "P5_transition_method": caudal_base_result.get("P5_transition_method", ""),
        "P5_model_to_transition_distance_mm": "",
        "tail_width_profile": caudal_base_result.get("tail_width_profile", ""),
        "tail_width_transition_score": caudal_base_result.get("tail_width_transition_score", ""),
        "local_structure_qc_pass": local_structure_qc.get("local_structure_qc_pass", ""),
        "local_structure_review_reason": local_structure_qc.get("local_structure_review_reason", ""),
        "axis_centerline_quality": axis_qc.get("axis_centerline_quality", ""),
        "axis_qc_reason": axis_qc.get("axis_qc_reason", ""),
        "USE_MASK_AXIS_POINTS": axis_qc.get("USE_MASK_AXIS_POINTS", False),
    }
    for code in ("C1", "C2", "C3", "C4"):
        point = axis_qc.get(f"{code}_mask")
        metadata[f"{code}_mask_x"] = point[0] if isinstance(point, list) and len(point) >= 2 else ""
        metadata[f"{code}_mask_y"] = point[1] if isinstance(point, list) and len(point) >= 2 else ""
        metadata[f"{code}_mask_quality"] = axis_qc.get(f"{code}_mask_quality", "")
        metadata[f"{code}_source"] = axis_qc.get(f"{code}_source", "model")
    return {
        "keypoints": corrected,
        "confidences": confidences,
        "fish_mask": fish_mask,
        "fish_bbox": bbox,
        "segmentation_quality": quality,
        "correction_log": correction_log,
        "metadata": metadata,
    }
