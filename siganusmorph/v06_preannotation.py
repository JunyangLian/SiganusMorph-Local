"""v0.6 keypoint-wise hybrid preannotation wrapper.

This module keeps the experimental v0.6 workflow separate from the stable
v0.4.1 Streamlit path.  It runs v0.1 heatmap, v0.5 heatmap, v0.4.1 hybrid,
then applies the keypoint-wise selector.  It never writes labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .body_contour_midline import estimate_body_contour_midline_points
from .heatmap_preannotation import (
    DEFAULT_DATASET_ROOT,
    clear_heatmap_model_cache,
    default_heatmap_model,
    predict_warped_keypoints_heatmap_unet,
)
from .hybrid_point_selector import derive_hybrid_p7v
from .hybrid_preannotation import preannotate_warped_image_hybrid_v04
from .keypointwise_hybrid_selector import KEYPOINT_NAMES, select_keypoints_v06
from .local_normal_measurement import (
    estimate_body_depth_by_body_midline_normals,
    estimate_peduncle_depth_by_axis_normals,
)
from .operculum_geometry import validate_p3_operculum_position
from .preannotation import default_preannotation_model
from .runtime_resources import collect_runtime_memory, low_resource_mode
from .tail_geometry import estimate_p6_from_tail_fork_gap, validate_p6_geometry


V05_DATASET_ROOT = Path("datasets") / "siganusmorph_heatmap_unet_v0.5_candidate"
V05_MODEL_PATH = Path("models") / "siganusmorph_heatmap_unet_v0.5" / "preannotation_candidate.pt"
V06_RECOMMENDATION_PATH = (
    Path("results") / "model_eval" / "v0.6_keypointwise_hybrid_selector" / "keypoint_source_recommendation.csv"
)
V06_STRICT_RECOMMENDATION_PATH = (
    Path("results")
    / "model_eval"
    / "v0.6_keypointwise_hybrid_selector"
    / "same_set_comparison"
    / "same_set_source_recommendation.csv"
)


def default_v05_heatmap_model(project_root: Path) -> Path:
    """Return the v0.5 heatmap candidate path."""
    return project_root / V05_MODEL_PATH


def _normalize_source(source: object) -> str:
    text = str(source or "").strip()
    aliases = {
        "v041_automatic": "v041_hybrid",
        "v0.4.1_hybrid": "v041_hybrid",
        "v0.4.1 automatic": "v041_hybrid",
        "v0.4.1": "v041_hybrid",
        "mask_or_geometry": "mask_or_geometry_rule_if_available",
        "mask_rule": "mask_or_geometry_rule_if_available",
    }
    return aliases.get(text, text or "v01_heatmap")


def fallback_source_recommendations() -> dict[str, dict[str, str]]:
    """Built-in recommendation table used only when CSV reports are absent."""
    rows = {
        "P1_snout_tip": ("mask_left_boundary", "v041_hybrid"),
        "P2_eye_front": ("v01_heatmap", "v041_hybrid"),
        "P3_operculum_posterior": ("v041_hybrid", "v01_heatmap"),
        "P4_peduncle_start_midpoint": ("v05_heatmap", "v041_hybrid"),
        "P5_caudal_base_midpoint": ("v01_heatmap", "v041_hybrid"),
        "P6_caudal_fork_midpoint": ("v01_heatmap", "v041_hybrid"),
        "P7U_caudal_fin_upper_tip": ("v05_heatmap_with_mask_QC", "v01_heatmap"),
        "P7L_caudal_fin_lower_tip": ("v01_heatmap", "v041_hybrid"),
        "P8_body_depth_dorsal": ("v041_hybrid_with_QC", "v05_heatmap"),
        "P9_body_depth_ventral": ("v05_heatmap_with_QC", "v041_hybrid"),
        "P10_peduncle_depth_dorsal": ("v01_heatmap", "v041_hybrid"),
        "P11_peduncle_depth_ventral": ("v05_heatmap", "v041_hybrid"),
        "C1_head_axis_point": ("v041_hybrid", "v01_heatmap"),
        "C2_trunk_axis_point": ("v041_hybrid", "v01_heatmap"),
        "C3_posterior_trunk_axis_point": ("v041_hybrid", "v01_heatmap"),
        "C4_peduncle_axis_point": ("v041_hybrid", "v01_heatmap"),
    }
    return {
        key: {"recommended_default_source": source, "second_choice_source": fallback}
        for key, (source, fallback) in rows.items()
    }


def load_v06_source_recommendations(project_root: Path) -> tuple[dict[str, dict[str, Any]], Path | None]:
    """Load the selector recommendation table.

    The user-facing report path is preferred.  If it is missing, the strict
    same-set recommendation table is used, then a conservative built-in table.
    """
    for relative_path in (V06_RECOMMENDATION_PATH, V06_STRICT_RECOMMENDATION_PATH):
        path = project_root / relative_path
        if not path.exists():
            continue
        df = pd.read_csv(path)
        recommendations: dict[str, dict[str, Any]] = {}
        for _, row in df.iterrows():
            key = str(row.get("keypoint_name", "") or "").strip()
            if not key:
                continue
            source = row.get("recommended_default_source", row.get("recommend_default_source", ""))
            second = row.get("second_choice_source", "")
            if pd.isna(source):
                source = ""
            if pd.isna(second):
                second = ""
            recommendations[key] = {
                "recommended_default_source": _normalize_source(source),
                "second_choice_source": _normalize_source(second) if str(second or "").strip() else "",
                "reason": str(row.get("reason", "")),
            }
        if recommendations:
            return recommendations, path
    return fallback_source_recommendations(), None


def _point_map(points: Mapping[str, Any]) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for key, value in points.items():
        if isinstance(value, Mapping) and "x" in value and "y" in value:
            result[str(key)] = [float(value["x"]), float(value["y"])]
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            result[str(key)] = [float(value[0]), float(value[1])]
    return result


def _xy(value: Any) -> tuple[float, float] | None:
    if isinstance(value, Mapping) and "x" in value and "y" in value:
        return float(value["x"]), float(value["y"])
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return None


def _tail_axis_from_lobes(points: Mapping[str, Any]) -> tuple[float, float] | None:
    p5 = _xy(points.get("P5_caudal_base_midpoint"))
    p7u = _xy(points.get("P7U_caudal_fin_upper_tip"))
    p7l = _xy(points.get("P7L_caudal_fin_lower_tip"))
    if p5 is None or p7u is None or p7l is None:
        return None
    mid = ((p7u[0] + p7l[0]) / 2.0, (p7u[1] + p7l[1]) / 2.0)
    dx = mid[0] - p5[0]
    dy = mid[1] - p5[1]
    length = float((dx * dx + dy * dy) ** 0.5)
    if length <= 1e-9:
        return None
    return dx / length, dy / length


def _fallback_p6_from_tail_qc(tail_qc: Mapping[str, Any]) -> list[float] | None:
    for key, quality_key in (("P6_gap", "P6_gap_quality"), ("P6_concavity", "P6_concavity_quality"), ("P6_mask", "P6_mask_quality")):
        quality = str(tail_qc.get(quality_key, "") or "")
        if quality not in {"good", "warning"}:
            continue
        xy = _xy(tail_qc.get(key))
        if xy is not None:
            return [float(xy[0]), float(xy[1])]
    return None


def preannotate_warped_image_v06_keypointwise(
    warped_image: np.ndarray,
    image_name: str,
    project_root: Path,
    *,
    v034_model_path: str | Path | None = None,
    v01_heatmap_model_path: str | Path | None = None,
    v05_heatmap_model_path: str | Path | None = None,
    mm_per_pixel: float = 0.1,
) -> dict[str, Any]:
    """Run v0.6 experimental keypoint-wise hybrid preannotation."""

    v034_model = Path(v034_model_path) if v034_model_path else default_preannotation_model(project_root)
    v01_heatmap_model = Path(v01_heatmap_model_path) if v01_heatmap_model_path else default_heatmap_model(project_root)
    v05_heatmap_model = Path(v05_heatmap_model_path) if v05_heatmap_model_path else default_v05_heatmap_model(project_root)

    v041_result = preannotate_warped_image_hybrid_v04(
        warped_image,
        image_name,
        project_root,
        v034_model_path=v034_model,
        heatmap_model_path=v01_heatmap_model,
        mm_per_pixel=mm_per_pixel,
    )
    v041_metadata = dict(v041_result.get("metadata", {}) or {})
    v01_heatmap_keypoints = _point_map(dict(v041_metadata.get("heatmap_keypoints", {}) or {}))
    v01_heatmap_confidences = dict(v041_metadata.get("heatmap_confidences", {}) or {})
    v041_hybrid_keypoints = _point_map(dict(v041_metadata.get("hybrid_keypoints", {}) or v041_result.get("keypoints", {}) or {}))

    v05_result = predict_warped_keypoints_heatmap_unet(
        warped_image,
        image_name,
        project_root,
        model_path=v05_heatmap_model,
        dataset_root=V05_DATASET_ROOT,
    )
    v05_heatmap_keypoints = _point_map(dict(v05_result.get("heatmap_keypoints_warped", {}) or {}))
    v05_heatmap_confidences = dict(v05_result.get("heatmap_confidences", {}) or {})
    if low_resource_mode():
        clear_heatmap_model_cache(v05_heatmap_model)
        collect_runtime_memory()

    mask_suggestions = _point_map(dict(v041_metadata.get("mask_suggestions", {}) or {}))
    geometric_suggestions = _point_map(dict(v041_metadata.get("geometric_suggestions", {}) or {}))
    local_normal_suggestions = _point_map(dict(v041_metadata.get("local_normal_suggestions", {}) or {}))
    geometric_for_selector = dict(geometric_suggestions)
    geometric_for_selector.update(local_normal_suggestions)

    source_recommendations, recommendation_path = load_v06_source_recommendations(project_root)
    v06_keypoints, v06_point_sources, v06_qc = select_keypoints_v06(
        v01_heatmap_keypoints,
        v01_heatmap_confidences,
        v05_heatmap_keypoints,
        v05_heatmap_confidences,
        v041_hybrid_keypoints,
        mask_suggestions,
        geometric_for_selector,
        v041_metadata.get("qc_results", {}),
        source_recommendations,
    )
    if (
        "P1_snout_tip" in v06_keypoints
        and source_recommendations.get("P1_snout_tip", {}).get("recommended_default_source") == "mask_left_boundary"
        and str(v041_metadata.get("p1_source", "")) == "mask_left_boundary"
    ):
        v06_point_sources["P1_snout_tip"] = "mask_left_boundary"

    missing = [key for key in KEYPOINT_NAMES if key not in v06_keypoints]
    if missing:
        v06_qc.setdefault("review_flags", {})
        for key in missing:
            v06_qc["review_flags"][key] = "missing_v06_candidate"
        v06_qc["v06_qc_pass"] = False

    tail_geometry_qc = dict(v041_metadata.get("tail_geometry_qc", {}) or {})
    p6_fallback = _fallback_p6_from_tail_qc(tail_geometry_qc)
    p6_gap_derivation = estimate_p6_from_tail_fork_gap(
        tail_mask=None,
        fish_mask=v041_result.get("fish_mask"),
        keypoints=v06_keypoints,
        image_shape=warped_image.shape,
        config={"mm_per_pixel": mm_per_pixel, "warped_image": warped_image},
    )
    p6_geometric = _xy(p6_gap_derivation.get("P6_geometric"))
    if p6_gap_derivation.get("P6_gap_qc_pass") and p6_geometric is not None:
        p6_fallback = [float(p6_geometric[0]), float(p6_geometric[1])]
    p6_qc = validate_p6_geometry(
        _xy(v06_keypoints.get("P5_caudal_base_midpoint")),
        _xy(v06_keypoints.get("P6_caudal_fork_midpoint")),
        _xy(v06_keypoints.get("P7U_caudal_fin_upper_tip")),
        _xy(v06_keypoints.get("P7L_caudal_fin_lower_tip")),
        None,
        _tail_axis_from_lobes(v06_keypoints),
        fish_mask=v041_result.get("fish_mask"),
        config={"P6_mask_fork_rule": p6_fallback, "P6_geometric": p6_gap_derivation.get("P6_geometric")},
    )
    p6_review_points: list[str] = []
    if not bool(p6_qc.get("P6_geometry_qc_pass", False)):
        p6_review_points.append("P6_caudal_fork_midpoint")
        v06_qc.setdefault("review_flags", {})
        v06_qc["review_flags"]["P6_caudal_fork_midpoint"] = "P6_geometry_failed"
        v06_qc["v06_qc_pass"] = False
        if p6_qc.get("P6_fallback_used") and p6_fallback is not None:
            v06_keypoints["P6_caudal_fork_midpoint"] = [float(p6_fallback[0]), float(p6_fallback[1])]
            source = "tail_fork_gap_geometric" if p6_gap_derivation.get("P6_gap_qc_pass") else "mask_fork_rule_fallback"
            v06_point_sources["P6_caudal_fork_midpoint"] = source
            p6_qc["P6_source"] = source
            p6_qc["P6_review_reason"] = "model P6 failed geometry QC; tail fork gap geometry used" if source == "tail_fork_gap_geometric" else "P6 geometry QC failed; fallback used"
        else:
            p6_qc["P6_source"] = v06_point_sources.get("P6_caudal_fork_midpoint", "")

    body_depth_geometry: dict[str, Any] = {}
    peduncle_depth_geometry: dict[str, Any] = {}
    try:
        body_midline = estimate_body_contour_midline_points(
            v041_result.get("fish_mask"),
            v06_keypoints,
            warped_image.shape,
            config={"mm_per_pixel": mm_per_pixel},
        )
        body_depth_geometry = estimate_body_depth_by_body_midline_normals(
            v041_result.get("fish_mask"),
            body_midline.get("body_core_mask"),
            body_midline,
            v06_keypoints,
            warped_image.shape,
            config={"mm_per_pixel": mm_per_pixel},
        )
        peduncle_depth_geometry = estimate_peduncle_depth_by_axis_normals(
            v041_result.get("fish_mask"),
            v06_keypoints,
            warped_image.shape,
            config={"mm_per_pixel": mm_per_pixel},
        )
    except Exception as exc:  # noqa: BLE001 - keep as QC metadata.
        body_depth_geometry = {
            "P8_geometric": None,
            "P9_geometric": None,
            "body_depth_geometric_mm": None,
            "body_depth_geometric_qc_pass": False,
            "body_depth_geometric_review_reason": f"body_depth_geometry_failed:{exc}",
            "body_depth_source": "body_midline_normal_trunk_boundary_max_width",
            "body_depth_geometry_version": "v0.6.9_morphometric_definition_fix",
        }
        peduncle_depth_geometry = {
            "P10_geometric": None,
            "P11_geometric": None,
            "peduncle_depth_geometric_mm": None,
            "peduncle_depth_geometric_qc_pass": False,
            "peduncle_depth_geometric_review_reason": f"peduncle_depth_geometry_failed:{exc}",
            "peduncle_depth_source": "P4_P5_axis_normal_min_width",
            "peduncle_depth_geometry_version": "v0.6.8_peduncle_axis_normals",
        }
    try:
        operculum_qc = validate_p3_operculum_position(
            v06_keypoints.get("P1_snout_tip"),
            v06_keypoints.get("P2_eye_front"),
            v06_keypoints.get("P3_operculum_posterior"),
            v06_keypoints.get("C1_head_axis_point"),
            v06_keypoints.get("P4_peduncle_start_midpoint"),
            body_midline if "body_midline" in locals() else None,
            v041_result.get("fish_mask"),
            warped_image,
            config={"mm_per_pixel": mm_per_pixel},
        )
    except Exception as exc:  # noqa: BLE001 - keep as QC metadata.
        operculum_qc = {
            "P3_qc_pass": False,
            "P3_needs_review": True,
            "P3_review_reason": f"P3_qc_failed:{exc}",
            "P3_operculum_edge_suggestion": None,
            "P3_current_to_suggestion_distance_mm": None,
        }
    if bool(operculum_qc.get("P3_needs_review", False)):
        v06_qc.setdefault("review_flags", {})
        v06_qc["review_flags"]["P3_operculum_posterior"] = str(operculum_qc.get("P3_review_reason", "P3_operculum_qc_warning") or "P3_operculum_qc_warning")

    p7v = derive_hybrid_p7v(v06_keypoints)
    p7v_valid = bool(p7v.get("point")) and not bool(p7v.get("p7v_invalid_reason"))
    p7v["p7v_valid"] = p7v_valid
    review_flags = v06_qc.get("review_flags", {}) if isinstance(v06_qc.get("review_flags", {}), Mapping) else {}
    review_reason_parts = [f"{key}:{value}" for key, value in review_flags.items()]
    if not p7v_valid:
        review_reason_parts.append(str(p7v.get("p7v_invalid_reason") or "invalid_p7v"))
    review_reason = "; ".join(part for part in review_reason_parts if part)

    point_source_counts = {
        "num_points_from_v01": sum(1 for source in v06_point_sources.values() if source == "v01_heatmap"),
        "num_points_from_v05": sum(1 for source in v06_point_sources.values() if source == "v05_heatmap"),
        "num_points_from_v041": sum(1 for source in v06_point_sources.values() if source == "v041_hybrid"),
        "num_points_from_mask_rule": sum(1 for source in v06_point_sources.values() if "mask" in source or "rule" in source),
    }

    metadata = dict(v041_metadata)
    metadata.update(
        {
            "preannotation_mode": "v0.6_keypointwise_hybrid_experimental",
            "runtime_low_resource_mode": low_resource_mode(),
            "runtime_model_cache_policy": "release_between_stages" if low_resource_mode() else "persistent",
            "hybrid_model_version": "v0.6_keypointwise_hybrid_selector",
            "annotation_mode": "auto_assisted_hybrid_v0.6",
            "annotation_status": "auto_preannotation_unverified",
            "v06_keypoints": v06_keypoints,
            "v06_point_sources": v06_point_sources,
            "v06_qc_results": {
                **v06_qc,
                "tail_qc": p6_qc,
                "P6_gap_derivation": p6_gap_derivation,
                "review_points": p6_review_points,
                "p7v": p7v,
                "p7v_valid": p7v_valid,
                "p7v_invalid_reason": p7v.get("p7v_invalid_reason", ""),
                "v06_review_reason": review_reason,
            },
            "v01_heatmap_keypoints": v01_heatmap_keypoints,
            "v01_heatmap_confidences": v01_heatmap_confidences,
            "v05_heatmap_keypoints": v05_heatmap_keypoints,
            "v05_heatmap_confidences": v05_heatmap_confidences,
            "v05_heatmap_debug_info": dict(v05_result.get("heatmap_debug_info", {}) or {}),
            "v041_hybrid_keypoints": v041_hybrid_keypoints,
            "v041_hybrid_metadata": v041_metadata,
            "tail_qc": {
                **p6_qc,
                "P6_fallback": p6_fallback,
                "P6_geometric": p6_gap_derivation.get("P6_geometric"),
                "P6_gap_qc_pass": p6_gap_derivation.get("P6_gap_qc_pass", ""),
                "P6_geometric_available": p6_gap_derivation.get("P6_geometric_available", False),
                "P6_current_to_geometric_distance_mm": p6_gap_derivation.get("P6_current_to_geometric_distance_mm", ""),
                "P6_needs_review": not bool(p6_qc.get("P6_geometry_qc_pass", False)),
                "review_points": p6_review_points,
            },
            "P6_gap_derivation": p6_gap_derivation,
            "body_depth_geometry": body_depth_geometry,
            "peduncle_depth_geometry": peduncle_depth_geometry,
            "operculum_qc": operculum_qc,
            "v06_source_recommendation_path": "" if recommendation_path is None else str(recommendation_path),
            "v06_source_recommendations": source_recommendations,
            "heatmap_keypoints": v01_heatmap_keypoints,
            "heatmap_confidences": v01_heatmap_confidences,
            "hybrid_keypoints": v06_keypoints,
            "corrected_keypoints": v06_keypoints,
            "auto_preannotation_initial_corrected_keypoints": v06_keypoints,
            "point_sources": v06_point_sources,
            "hybrid_qc_results": {
                **dict(v041_metadata.get("hybrid_qc_results", {}) or {}),
                "v06_qc_pass": bool(v06_qc.get("v06_qc_pass", True)) and p7v_valid,
                "v06_review_reason": review_reason,
                "tail_qc": p6_qc,
                "P6_gap_derivation": p6_gap_derivation,
                "p7v": p7v,
                "p7v_valid": p7v_valid,
                "p7v_invalid_reason": p7v.get("p7v_invalid_reason", ""),
            },
            "qc_results": {
                **dict(v041_metadata.get("qc_results", {}) or {}),
                "v06_qc_pass": bool(v06_qc.get("v06_qc_pass", True)) and p7v_valid,
                "v06_review_reason": review_reason,
                "p7v_valid": p7v_valid,
                "p7v_suggestion": p7v.get("point"),
                "P6_geometry_qc_pass": p6_qc.get("P6_geometry_qc_pass", ""),
                "P6_needs_review": not bool(p6_qc.get("P6_geometry_qc_pass", False)),
                "P6_fallback_used": p6_qc.get("P6_fallback_used", False),
                "P6_review_reason": p6_qc.get("P6_review_reason", ""),
                "P6_gap_qc_pass": p6_gap_derivation.get("P6_gap_qc_pass", ""),
                "P6_geometric_available": p6_gap_derivation.get("P6_geometric_available", False),
                "P6_current_to_geometric_distance_mm": p6_gap_derivation.get("P6_current_to_geometric_distance_mm", ""),
                "body_depth_geometric_qc_pass": body_depth_geometry.get("body_depth_geometric_qc_pass", ""),
                "peduncle_depth_geometric_qc_pass": peduncle_depth_geometry.get("peduncle_depth_geometric_qc_pass", ""),
                "P3_qc_pass": operculum_qc.get("P3_qc_pass", ""),
                "P3_needs_review": operculum_qc.get("P3_needs_review", ""),
                "P3_review_reason": operculum_qc.get("P3_review_reason", ""),
            },
            "derived_points": {"P7V_caudal_fin_posterior_endpoint": p7v.get("point")} if p7v.get("point") else {},
            "needs_review": bool(review_reason) or bool(v041_metadata.get("needs_review", False)),
            "review_reason": "; ".join(part for part in (str(v041_metadata.get("review_reason", "") or ""), review_reason) if part),
            "large_error_risk_flags": review_flags,
            "fallback_used": [key for key, source in v06_point_sources.items() if source != source_recommendations.get(key, {}).get("recommended_default_source")],
            **point_source_counts,
            "keypoint_edit_log": [
                {
                    "event": "v06_keypointwise_preannotation_created",
                    "version": "v0.6_keypointwise_hybrid_selector",
                    "v034_model_path": str(v034_model),
                    "v01_heatmap_model_path": str(v01_heatmap_model),
                    "v05_heatmap_model_path": str(v05_heatmap_model),
                    "source_recommendation_path": "" if recommendation_path is None else str(recommendation_path),
                }
            ],
        }
    )
    return {
        "keypoints": v06_keypoints,
        "confidences": v05_heatmap_confidences,
        "fish_mask": v041_result.get("fish_mask"),
        "fish_bbox": v041_result.get("fish_bbox"),
        "segmentation_quality": v041_result.get("segmentation_quality"),
        "metadata": metadata,
        "v041_result": v041_result,
        "v05_heatmap_result": v05_result,
    }
