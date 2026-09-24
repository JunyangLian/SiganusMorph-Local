"""Mask-derived centerline suggestions for C1-C4 axis points."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


P1 = "P1_snout_tip"
P4 = "P4_peduncle_start_midpoint"
P5 = "P5_caudal_base_midpoint"
C1 = "C1_head_axis_point"
C2 = "C2_trunk_axis_point"
C3 = "C3_posterior_trunk_axis_point"
C4 = "C4_peduncle_axis_point"

AXIS_KEYS = (C1, C2, C3, C4)
USE_MASK_AXIS_POINTS = False


def _xy(value: Any) -> tuple[float, float] | None:
    if isinstance(value, Mapping):
        if "x" in value and "y" in value:
            return float(value["x"]), float(value["y"])
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return None


def _point(points: Mapping[str, Any], key: str) -> tuple[float, float] | None:
    return _xy(points.get(key))


def _unit(start: tuple[float, float] | None, end: tuple[float, float] | None) -> tuple[float, float] | None:
    if start is None or end is None:
        return None
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return None
    return dx / length, dy / length


def _sample_mask_center(
    mask: np.ndarray,
    center: tuple[float, float],
    tangent: tuple[float, float],
    max_radius: int,
) -> tuple[tuple[float, float] | None, str]:
    normal = (-tangent[1], tangent[0])
    height, width = mask.shape[:2]
    hits: list[tuple[float, float, float]] = []
    for direction in (-1.0, 1.0):
        last_inside: tuple[float, float] | None = None
        for step in range(0, max_radius):
            x = center[0] + normal[0] * direction * step
            y = center[1] + normal[1] * direction * step
            ix = int(round(x))
            iy = int(round(y))
            if ix < 0 or iy < 0 or ix >= width or iy >= height:
                break
            if mask[iy, ix] > 0:
                last_inside = (x, y)
            elif last_inside is not None and step > 3:
                break
        if last_inside is None:
            return None, "failed_boundary_missing"
        signed = (last_inside[0] - center[0]) * normal[0] + (last_inside[1] - center[1]) * normal[1]
        hits.append((last_inside[0], last_inside[1], signed))
    if len(hits) != 2:
        return None, "failed_boundary_missing"
    p_a = hits[0]
    p_b = hits[1]
    span = abs(p_a[2] - p_b[2])
    if span < 18:
        return None, "failed_section_too_narrow"
    return ((p_a[0] + p_b[0]) / 2.0, (p_a[1] + p_b[1]) / 2.0), "good"


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def estimate_axis_points_from_mask(
    fish_mask: np.ndarray | None,
    keypoints: Mapping[str, Any],
    body_axis_reference: Any = None,
    *,
    use_mask_axis_points: bool = USE_MASK_AXIS_POINTS,
) -> dict[str, Any]:
    """Estimate C1-C4 as section midpoints from the fish mask.

    Suggestions are returned separately; by default the caller should not use
    them to overwrite model points.
    """
    if fish_mask is None or fish_mask.size == 0 or np.count_nonzero(fish_mask) == 0:
        return {
            "mask_suggestions": {},
            "axis_qc": {
                "axis_centerline_quality": "failed",
                "axis_qc_pass": False,
                "axis_qc_reason": "missing_mask",
                "USE_MASK_AXIS_POINTS": bool(use_mask_axis_points),
            },
            "point_sources": {},
        }
    mask = (fish_mask > 0).astype(np.uint8) * 255
    p1 = _point(keypoints, P1)
    p5 = _point(keypoints, P5)
    if p1 is None or p5 is None:
        return {
            "mask_suggestions": {},
            "axis_qc": {
                "axis_centerline_quality": "failed",
                "axis_qc_pass": False,
                "axis_qc_reason": "missing_P1_or_P5",
                "USE_MASK_AXIS_POINTS": bool(use_mask_axis_points),
            },
            "point_sources": {},
        }
    tangent = _unit(p1, p5)
    if tangent is None:
        return {
            "mask_suggestions": {},
            "axis_qc": {
                "axis_centerline_quality": "failed",
                "axis_qc_pass": False,
                "axis_qc_reason": "P1_P5_overlap",
                "USE_MASK_AXIS_POINTS": bool(use_mask_axis_points),
            },
            "point_sources": {},
        }

    p4 = _point(keypoints, P4)
    axis_len = _distance(p1, p5)
    targets = {
        C1: 0.23,
        C2: 0.48,
        C3: 0.70,
        C4: 0.88,
    }
    if p4 is not None:
        c4_center = ((p4[0] + p5[0]) / 2.0, (p4[1] + p5[1]) / 2.0)
    else:
        c4_center = (p1[0] + tangent[0] * axis_len * targets[C4], p1[1] + tangent[1] * axis_len * targets[C4])

    suggestions: dict[str, list[float]] = {}
    qualities: dict[str, str] = {}
    reasons: list[str] = []
    max_radius = max(60, int(round(axis_len * 0.35)))
    for key, frac in targets.items():
        if key == C4:
            center = c4_center
        else:
            center = (p1[0] + tangent[0] * axis_len * frac, p1[1] + tangent[1] * axis_len * frac)
        suggestion, quality = _sample_mask_center(mask, center, tangent, max_radius=max_radius)
        qualities[key] = quality
        if suggestion is not None:
            suggestions[key] = [float(suggestion[0]), float(suggestion[1])]
        else:
            reasons.append(f"{key}_{quality}")

    # QC against existing model/manual draft C points without forcing replacement.
    projections: list[float] = []
    for key in AXIS_KEYS:
        point = _point(keypoints, key)
        if point is not None:
            projections.append((point[0] - p1[0]) * tangent[0] + (point[1] - p1[1]) * tangent[1])
        if point is not None and key in suggestions:
            if _distance(point, tuple(suggestions[key])) > max(40.0, axis_len * 0.045):
                reasons.append(f"{key}_far_from_mask_centerline_suggestion")
    if len(projections) == 4 and not all(b >= a - 12 for a, b in zip(projections, projections[1:])):
        reasons.append("axis_projection_not_monotonic")

    quality = "good" if len(suggestions) == 4 and not reasons else "warning" if suggestions else "failed"
    point_sources = {key: "mask_centerline_rule" for key in suggestions} if use_mask_axis_points else {}
    qc = {
        "axis_centerline_quality": quality,
        "axis_qc_pass": not reasons,
        "axis_qc_reason": ";".join(dict.fromkeys(reasons)),
        "USE_MASK_AXIS_POINTS": bool(use_mask_axis_points),
    }
    for key in AXIS_KEYS:
        point = suggestions.get(key)
        qc[f"{key.split('_', 1)[0]}_mask_quality"] = qualities.get(key, "failed")
        qc[f"{key.split('_', 1)[0]}_source"] = "mask_centerline_rule" if use_mask_axis_points and point else "model"
        if point:
            qc[f"{key.split('_', 1)[0]}_mask"] = point
    return {"mask_suggestions": suggestions, "axis_qc": qc, "point_sources": point_sources}
