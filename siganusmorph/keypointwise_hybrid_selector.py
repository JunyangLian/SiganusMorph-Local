"""Keypoint-wise hybrid source selection for v0.6 evaluation.

This module is intentionally small and side-effect free.  It does not run
models and does not write labels.  The evaluation script supplies candidate
keypoints from v0.1 heatmap, v0.5 heatmap, v0.4.1 hybrid, and optional
geometry rules, then this selector picks a per-keypoint source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


KEYPOINT_NAMES = [
    "P1_snout_tip",
    "P2_eye_front",
    "P3_operculum_posterior",
    "P4_peduncle_start_midpoint",
    "P5_caudal_base_midpoint",
    "P6_caudal_fork_midpoint",
    "P7U_caudal_fin_upper_tip",
    "P7L_caudal_fin_lower_tip",
    "P8_body_depth_dorsal",
    "P9_body_depth_ventral",
    "P10_peduncle_depth_dorsal",
    "P11_peduncle_depth_ventral",
    "C1_head_axis_point",
    "C2_trunk_axis_point",
    "C3_posterior_trunk_axis_point",
    "C4_peduncle_axis_point",
]


SOURCE_ALIASES = {
    "mask_left_boundary": ["mask_or_geometry_rule_if_available", "v041_hybrid", "v05_heatmap", "v01_heatmap"],
    "v01_heatmap": ["v01_heatmap", "v05_heatmap", "v041_hybrid"],
    "v05_heatmap": ["v05_heatmap", "v01_heatmap", "v041_hybrid"],
    "v041_hybrid": ["v041_hybrid", "v05_heatmap", "v01_heatmap"],
    "v05_heatmap_with_QC": ["v05_heatmap", "mask_or_geometry_rule_if_available", "v01_heatmap", "v041_hybrid"],
    "v05_heatmap_with_mask_QC": ["v05_heatmap", "mask_or_geometry_rule_if_available", "v01_heatmap", "v041_hybrid"],
    "v041_hybrid_with_QC": ["v041_hybrid", "mask_or_geometry_rule_if_available", "v01_heatmap", "v05_heatmap"],
}


@dataclass(frozen=True)
class CandidatePoint:
    x: float
    y: float
    confidence: float | None = None
    source_model: str | None = None


def _as_point(value: object) -> CandidatePoint | None:
    if value is None:
        return None
    if isinstance(value, CandidatePoint):
        return value
    if isinstance(value, Mapping):
        try:
            return CandidatePoint(
                x=float(value["x"]),
                y=float(value["y"]),
                confidence=None if value.get("confidence") is None else float(value.get("confidence")),
                source_model=None if value.get("source_model") is None else str(value.get("source_model")),
            )
        except Exception:
            return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return CandidatePoint(float(value[0]), float(value[1]))
        except Exception:
            return None
    return None


def _source_order(recommended_source: str, second_choice_source: str | None = None) -> list[str]:
    order = list(SOURCE_ALIASES.get(recommended_source, [recommended_source]))
    if second_choice_source:
        for source in SOURCE_ALIASES.get(second_choice_source, [second_choice_source]):
            if source not in order:
                order.append(source)
    for source in ["v01_heatmap", "v05_heatmap", "v041_hybrid", "mask_or_geometry_rule_if_available"]:
        if source not in order:
            order.append(source)
    return order


def select_keypoints_v06(
    v01_heatmap_keypoints: Mapping[str, object],
    v01_confidences: Mapping[str, float] | None,
    v05_heatmap_keypoints: Mapping[str, object],
    v05_confidences: Mapping[str, float] | None,
    v041_hybrid_keypoints: Mapping[str, object],
    mask_suggestions: Mapping[str, object],
    geometric_suggestions: Mapping[str, object],
    qc_results: Mapping[str, object] | None,
    source_recommendations: Mapping[str, Mapping[str, object]],
    confidence_threshold: float = 0.05,
) -> tuple[dict[str, list[float]], dict[str, str], dict[str, object]]:
    """Select one point per keypoint from available candidates.

    The selector is deliberately conservative: it follows the recommendation
    table, falls back when a candidate is missing or confidence is extremely
    low, and records review flags rather than pretending low-confidence points
    are reliable.
    """

    candidates_by_source: dict[str, Mapping[str, object]] = {
        "v01_heatmap": v01_heatmap_keypoints,
        "v05_heatmap": v05_heatmap_keypoints,
        "v041_hybrid": v041_hybrid_keypoints,
        "mask_or_geometry_rule_if_available": {**geometric_suggestions, **mask_suggestions},
    }
    confidence_maps = {
        "v01_heatmap": v01_confidences or {},
        "v05_heatmap": v05_confidences or {},
    }

    selected: dict[str, list[float]] = {}
    sources: dict[str, str] = {}
    review_flags: dict[str, str] = {}

    for keypoint in KEYPOINT_NAMES:
        rec = source_recommendations.get(keypoint, {})
        recommended = str(rec.get("recommended_default_source", "v01_heatmap"))
        second_choice = rec.get("second_choice_source")
        chosen_point: CandidatePoint | None = None
        chosen_source: str | None = None

        for source in _source_order(recommended, None if second_choice is None else str(second_choice)):
            point = _as_point(candidates_by_source.get(source, {}).get(keypoint))
            if point is None:
                continue
            confidence = point.confidence
            if confidence is None:
                confidence = confidence_maps.get(source, {}).get(keypoint)
            if confidence is not None and confidence < confidence_threshold:
                review_flags[keypoint] = f"{source}_low_confidence"
                continue
            chosen_point = point
            chosen_source = source
            break

        if chosen_point is None:
            review_flags[keypoint] = review_flags.get(keypoint, "no_candidate_available")
            continue
        selected[keypoint] = [chosen_point.x, chosen_point.y]
        sources[keypoint] = chosen_source or "unknown"

    qc = {
        "v06_qc_pass": len(review_flags) == 0,
        "review_flags": review_flags,
        "input_qc_results": dict(qc_results or {}),
    }
    return selected, sources, qc
