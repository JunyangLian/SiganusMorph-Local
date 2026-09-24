"""v0.4 hybrid heatmap + geometry preannotation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .heatmap_preannotation import clear_heatmap_model_cache, default_heatmap_model, predict_warped_keypoints_heatmap_unet
from .hybrid_point_selector import select_hybrid_keypoints
from .local_normal_measurement import (
    compute_curvature_qc,
    estimate_body_depth_by_local_normals,
    estimate_peduncle_depth_by_local_normals,
)
from .preannotation import clear_preannotation_model_cache, default_preannotation_model, preannotate_warped_image
from .runtime_resources import collect_runtime_memory, low_resource_mode


def preannotate_warped_image_hybrid_v04(
    warped_image: np.ndarray,
    image_name: str,
    project_root: Path,
    *,
    v034_model_path: str | Path | None = None,
    heatmap_model_path: str | Path | None = None,
    mm_per_pixel: float = 0.1,
) -> dict[str, Any]:
    """Run v0.4 hybrid preannotation without saving or modifying labels."""
    v034_model = Path(v034_model_path) if v034_model_path else default_preannotation_model(project_root)
    heatmap_model = Path(heatmap_model_path) if heatmap_model_path else default_heatmap_model(project_root)
    v034_result = preannotate_warped_image(warped_image, image_name, project_root, v034_model)
    v034_metadata = dict(v034_result.get("metadata", {}) or {})
    v034_keypoints = dict(v034_metadata.get("corrected_keypoints", {}) or v034_result.get("keypoints", {}) or {})
    v034_model_raw = dict(v034_metadata.get("model_keypoints_raw", {}) or {})
    mask_suggestions = dict(v034_metadata.get("mask_suggestions", {}) or {})
    geometric_suggestions = dict(v034_metadata.get("geometric_suggestions", {}) or {})
    local_suggestions = dict(v034_metadata.get("local_structure_suggestions", {}) or {})
    v034_qc = dict(v034_metadata.get("qc_results", {}) or {})
    tail_qc = dict(v034_metadata.get("tail_geometry_qc", {}) or {})

    if low_resource_mode():
        clear_preannotation_model_cache(v034_model)
        collect_runtime_memory()

    heatmap_result = predict_warped_keypoints_heatmap_unet(
        warped_image,
        image_name,
        project_root,
        model_path=heatmap_model,
    )
    heatmap_keypoints = dict(heatmap_result.get("heatmap_keypoints_warped", {}) or {})
    heatmap_confidences = dict(heatmap_result.get("heatmap_confidences", {}) or {})
    if low_resource_mode():
        clear_heatmap_model_cache(heatmap_model)
        collect_runtime_memory()
    fish_bbox = tuple(v034_result.get("fish_bbox")) if v034_result.get("fish_bbox") is not None else None
    hybrid = select_hybrid_keypoints(
        heatmap_keypoints,
        heatmap_confidences,
        v034_model_raw,
        mask_suggestions,
        {**geometric_suggestions, **local_suggestions},
        {**v034_qc, **tail_qc},
        fish_mask=v034_result.get("fish_mask"),
        fish_bbox=fish_bbox,
        mm_per_pixel=mm_per_pixel,
        v034_keypoints=v034_keypoints,
    )
    hybrid_keypoints = dict(hybrid.get("hybrid_keypoints", {}) or {})
    point_sources = dict(hybrid.get("point_sources", {}) or {})
    hybrid_qc = dict(hybrid.get("hybrid_qc_results", {}) or {})
    merged_suggestions = dict(mask_suggestions)
    merged_suggestions.update(geometric_suggestions)
    merged_suggestions.update(local_suggestions)

    local_config = {"mm_per_pixel": mm_per_pixel}
    body_local = estimate_body_depth_by_local_normals(
        v034_result.get("fish_mask"),
        None,
        hybrid_keypoints,
        warped_image.shape,
        local_config,
    )
    axis_curve = body_local.get("axis_curve") if isinstance(body_local, dict) else None
    peduncle_local = estimate_peduncle_depth_by_local_normals(
        v034_result.get("fish_mask"),
        axis_curve if isinstance(axis_curve, dict) else None,
        hybrid_keypoints,
        warped_image.shape,
        local_config,
    )
    curvature_qc = compute_curvature_qc(
        axis_curve if isinstance(axis_curve, dict) else None,
        hybrid_keypoints,
        local_config,
    )
    local_normal_suggestions = {}
    if isinstance(body_local.get("P8_refined"), list) and isinstance(body_local.get("P9_refined"), list):
        local_normal_suggestions["P8_body_depth_dorsal"] = body_local["P8_refined"]
        local_normal_suggestions["P9_body_depth_ventral"] = body_local["P9_refined"]
    if isinstance(peduncle_local.get("P10_refined"), list) and isinstance(peduncle_local.get("P11_refined"), list):
        local_normal_suggestions["P10_peduncle_depth_dorsal"] = peduncle_local["P10_refined"]
        local_normal_suggestions["P11_peduncle_depth_ventral"] = peduncle_local["P11_refined"]

    review_reason = str(hybrid_qc.get("hybrid_review_reason", "") or "")
    local_review_parts = []
    if body_local.get("qc_pass") is False and body_local.get("review_reason"):
        local_review_parts.append(f"body_depth:{body_local.get('review_reason')}")
    if peduncle_local.get("qc_pass") is False and peduncle_local.get("review_reason"):
        local_review_parts.append(f"peduncle_depth:{peduncle_local.get('review_reason')}")
    if curvature_qc.get("review_reason"):
        local_review_parts.append(str(curvature_qc.get("review_reason")))
    if local_review_parts:
        review_reason = "; ".join([part for part in (review_reason, *local_review_parts) if part])

    body_original = body_local.get("body_depth_original_mm")
    peduncle_original = peduncle_local.get("caudal_peduncle_depth_original_mm")
    body_refined = body_local.get("width_mm")
    peduncle_refined = peduncle_local.get("width_mm")
    metadata = dict(v034_metadata)
    metadata.update(
        {
            "preannotation_mode": "v0.4_hybrid_heatmap_geometry",
            "hybrid_model_version": "v0.4_hybrid_heatmap_geometry",
            "enhanced_preannotation_version": "v0.4.1_curvature_aware_measurement",
            "measurement_geometry_version": "v0.4.1_curvature_aware_measurement",
            "annotation_mode": "auto_assisted_hybrid_v0.4",
            "annotation_status": "auto_preannotation_unverified",
            "model_keypoints_raw": v034_model_raw,
            "heatmap_keypoints": heatmap_keypoints,
            "heatmap_keypoints_crop": dict(heatmap_result.get("heatmap_keypoints_crop", {}) or {}),
            "heatmap_confidences": heatmap_confidences,
            "heatmap_debug_info": dict(heatmap_result.get("heatmap_debug_info", {}) or {}),
            "v034_keypoints": v034_keypoints,
            "v034_model_keypoints_raw": v034_model_raw,
            "mask_suggestions": mask_suggestions,
            "geometric_suggestions": {**geometric_suggestions, **local_suggestions},
            "local_normal_suggestions": local_normal_suggestions,
            "local_normal_measurements": {
                "body_depth": body_local,
                "peduncle_depth": peduncle_local,
            },
            "curvature_qc": curvature_qc,
            "hybrid_keypoints": hybrid_keypoints,
            "corrected_keypoints": hybrid_keypoints,
            "auto_preannotation_initial_corrected_keypoints": hybrid_keypoints,
            "point_sources": point_sources,
            "hybrid_qc_results": hybrid_qc,
            "qc_results": {
                **v034_qc,
                "hybrid_qc_pass": hybrid_qc.get("hybrid_qc_pass", ""),
                "hybrid_review_reason": review_reason,
                "p7v_valid": hybrid_qc.get("p7v_valid", ""),
                "p7v_suggestion": (hybrid_qc.get("p7v", {}) or {}).get("point") if isinstance(hybrid_qc.get("p7v", {}), dict) else "",
                "body_depth_qc_pass": body_local.get("qc_pass", ""),
                "body_depth_review_reason": body_local.get("review_reason", ""),
                "peduncle_depth_qc_pass": peduncle_local.get("qc_pass", ""),
                "peduncle_depth_review_reason": peduncle_local.get("review_reason", ""),
                "curvature_qc_level": curvature_qc.get("curvature_qc_level", ""),
                "curvature_qc_pass": curvature_qc.get("curvature_qc_pass", ""),
            },
            "body_depth_refined_mm": body_refined,
            "body_depth_original_mm": body_original,
            "body_depth_refined_minus_original_mm": body_local.get("body_depth_refined_minus_original_mm"),
            "body_depth_qc_pass": body_local.get("qc_pass", ""),
            "body_depth_review_reason": body_local.get("review_reason", ""),
            "caudal_peduncle_depth_refined_mm": peduncle_refined,
            "caudal_peduncle_depth_original_mm": peduncle_original,
            "caudal_peduncle_depth_refined_minus_original_mm": peduncle_local.get("caudal_peduncle_depth_refined_minus_original_mm"),
            "peduncle_depth_qc_pass": peduncle_local.get("qc_pass", ""),
            "peduncle_depth_review_reason": peduncle_local.get("review_reason", ""),
            "curvature_index": curvature_qc.get("curvature_index"),
            "curvature_qc_level": curvature_qc.get("curvature_qc_level", ""),
            "max_axis_deviation_mm": curvature_qc.get("max_axis_deviation_mm"),
            "axis_bend_angle_deg": curvature_qc.get("axis_bend_angle_deg"),
            "curvature_high_review_required": curvature_qc.get("curvature_high_review_required", ""),
            "measurements_needs_review": curvature_qc.get("measurements_needs_review", ""),
            "curvature_review_points": ";".join(curvature_qc.get("curvature_review_points", []) or []),
            "p7v_valid": hybrid_qc.get("p7v_valid", ""),
            "needs_review": bool(review_reason),
            "review_reason": review_reason,
            "keypoint_edit_log": [
                {
                    "event": "hybrid_preannotation_created",
                    "version": "v0.4_hybrid_heatmap_geometry",
                    "v034_model_path": str(v034_model),
                    "heatmap_model_path": str(heatmap_model),
                }
            ],
        }
    )
    p7v = hybrid_qc.get("p7v", {}) if isinstance(hybrid_qc.get("p7v", {}), dict) else {}
    if p7v.get("point") is not None:
        metadata["derived_points"] = {"P7V_caudal_fin_posterior_endpoint": p7v.get("point")}
    return {
        "keypoints": hybrid_keypoints,
        "confidences": heatmap_confidences,
        "fish_mask": v034_result.get("fish_mask"),
        "fish_bbox": v034_result.get("fish_bbox"),
        "segmentation_quality": v034_result.get("segmentation_quality"),
        "correction_log": v034_result.get("correction_log", {}),
        "metadata": metadata,
        "v034_result": v034_result,
        "heatmap_result": heatmap_result,
    }
