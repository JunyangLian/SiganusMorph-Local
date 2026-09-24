"""Hybrid heatmap + geometry point selection for v0.4 evaluation."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


def xy(value: Any) -> tuple[float, float] | None:
    if isinstance(value, Mapping):
        if "x" in value and "y" in value:
            return float(value["x"]), float(value["y"])
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return None


def distance_px(a: Any, b: Any) -> float | None:
    pa = xy(a)
    pb = xy(b)
    if pa is None or pb is None:
        return None
    return math.hypot(pa[0] - pb[0], pa[1] - pb[1])


def _point_in_mask(point: Any, fish_mask: np.ndarray | None) -> bool:
    p = xy(point)
    if p is None or fish_mask is None or fish_mask.size == 0:
        return False
    x, y = int(round(p[0])), int(round(p[1]))
    if y < 0 or x < 0 or y >= fish_mask.shape[0] or x >= fish_mask.shape[1]:
        return False
    return bool(fish_mask[y, x] > 0)


def _point_in_bbox(point: Any, bbox: tuple[int, int, int, int] | None, margin: float = 0.0) -> bool:
    p = xy(point)
    if p is None or bbox is None:
        return False
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    return (x1 - margin * w) <= p[0] <= (x2 + margin * w) and (y1 - margin * h) <= p[1] <= (y2 + margin * h)


def _is_head_roi(point: Any, bbox: tuple[int, int, int, int] | None) -> bool:
    p = xy(point)
    if p is None or bbox is None:
        return False
    x1, y1, x2, y2 = bbox
    width = max(1.0, float(x2 - x1))
    return x1 <= p[0] <= x1 + 0.35 * width and y1 <= p[1] <= y2


def _source_reason(existing: str, extra: str) -> str:
    parts = [part.strip() for part in str(existing or "").split(";") if part.strip()]
    if extra and extra not in parts:
        parts.append(extra)
    return ";".join(parts)


def _quality_good(value: Any) -> bool:
    text = str(value or "").lower()
    return text in {"good", "ok", "warning"} or text.startswith("good")


def _mask_suggestion_for(key: str, mask_suggestions: Mapping[str, Any], geometric_suggestions: Mapping[str, Any]) -> Any:
    return mask_suggestions.get(key) or geometric_suggestions.get(key)


def _axis_monotonic(points: Mapping[str, Any]) -> bool:
    order = [
        "P1_snout_tip",
        "C1_head_axis_point",
        "C2_trunk_axis_point",
        "C3_posterior_trunk_axis_point",
        "P4_peduncle_start_midpoint",
        "C4_peduncle_axis_point",
        "P5_caudal_base_midpoint",
    ]
    xs = []
    for key in order:
        point = xy(points.get(key))
        if point is None:
            return False
        xs.append(point[0])
    return all(xs[i] <= xs[i + 1] + 25.0 for i in range(len(xs) - 1))


def _p7v_validity(
    hybrid: Mapping[str, Any],
    p7v: dict[str, Any],
    fish_bbox: tuple[int, int, int, int] | None,
) -> tuple[bool, str]:
    p5 = xy(hybrid.get("P5_caudal_base_midpoint"))
    p7u = xy(hybrid.get("P7U_caudal_fin_upper_tip"))
    p7l = xy(hybrid.get("P7L_caudal_fin_lower_tip"))
    p7v_point = xy(p7v.get("point"))
    if p5 is None or p7u is None or p7l is None or p7v_point is None:
        return False, "missing_tail_points_for_p7v"
    if p7u[1] >= p7l[1]:
        return False, "P7U_not_above_P7L"
    extension = float(p7v.get("extension_px", 0.0) or 0.0)
    if fish_bbox is not None:
        width = max(1.0, float(fish_bbox[2] - fish_bbox[0]))
        if extension <= 0.03 * width or extension > 0.45 * width:
            return False, "caudal_extension_axis_out_of_range"
        if not _point_in_bbox(p7v_point, fish_bbox, margin=0.18):
            return False, "P7V_outside_tail_bbox"
    ymin, ymax = sorted([p7u[1], p7l[1]])
    if p7v_point[1] < ymin - 0.25 * (ymax - ymin) or p7v_point[1] > ymax + 0.25 * (ymax - ymin):
        return False, "P7V_y_outside_tip_range"
    return True, ""


def _unit(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float] | None:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return None
    return dx / norm, dy / norm


def derive_hybrid_p7v(hybrid: Mapping[str, Any]) -> dict[str, Any]:
    """Derive P7V from caudal lobe geometry rather than model P6."""
    p5 = xy(hybrid.get("P5_caudal_base_midpoint"))
    p7u = xy(hybrid.get("P7U_caudal_fin_upper_tip"))
    p7l = xy(hybrid.get("P7L_caudal_fin_lower_tip"))
    c4 = xy(hybrid.get("C4_peduncle_axis_point"))
    c3 = xy(hybrid.get("C3_posterior_trunk_axis_point"))
    if p5 is None or p7u is None or p7l is None:
        return {"point": None, "p7v_valid": False, "p7v_invalid_reason": "missing_tail_points"}
    midpoint = ((p7u[0] + p7l[0]) / 2.0, (p7u[1] + p7l[1]) / 2.0)
    axis = _unit(p5, midpoint)
    source = "P5_to_midpoint_P7U_P7L"
    if axis is None:
        axis_u = _unit(p5, p7u)
        axis_l = _unit(p5, p7l)
        if axis_u is not None and axis_l is not None:
            bx = axis_u[0] + axis_l[0]
            by = axis_u[1] + axis_l[1]
            norm = math.hypot(bx, by)
            if norm > 1e-9:
                axis = (bx / norm, by / norm)
                source = "tail_lobe_bisector"
    if axis is None and c4 is not None:
        axis = _unit(c4, p5)
        source = "C4_to_P5_fallback"
    if axis is None and c3 is not None:
        axis = _unit(c3, p5)
        source = "C3_to_P5_fallback"
    if axis is None:
        return {"point": None, "p7v_valid": False, "p7v_invalid_reason": "tail_axis_failed"}
    proj_upper = (p7u[0] - p5[0]) * axis[0] + (p7u[1] - p5[1]) * axis[1]
    proj_lower = (p7l[0] - p5[0]) * axis[0] + (p7l[1] - p5[1]) * axis[1]
    if proj_upper >= proj_lower:
        selected = "upper"
        extension = proj_upper
    else:
        selected = "lower"
        extension = proj_lower
    extension = max(0.0, float(extension))
    point = {"x": p5[0] + extension * axis[0], "y": p5[1] + extension * axis[1]}
    return {
        "point": point,
        "tail_axis": axis,
        "tail_axis_source": source,
        "tail_axis_uses_P6": False,
        "proj_upper_px": proj_upper,
        "proj_lower_px": proj_lower,
        "extension_px": extension,
        "selected": selected,
    }


def select_hybrid_keypoints(
    heatmap_keypoints: Mapping[str, Any],
    heatmap_confidences: Mapping[str, float],
    model_keypoints_raw: Mapping[str, Any],
    mask_suggestions: Mapping[str, Any],
    geometric_suggestions: Mapping[str, Any],
    qc_results: Mapping[str, Any],
    *,
    fish_mask: np.ndarray | None = None,
    fish_bbox: tuple[int, int, int, int] | None = None,
    mm_per_pixel: float = 0.1,
    v034_keypoints: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select v0.4 hybrid points from heatmap, v0.3.4 and rule suggestions."""
    v034 = dict(v034_keypoints or model_keypoints_raw)
    hybrid: dict[str, Any] = dict(v034)
    point_sources: dict[str, str] = {key: "v0.3.4_enhanced" for key in hybrid}
    review_reasons: list[str] = []
    per_point_qc: dict[str, dict[str, Any]] = {}

    def conf_good(key: str, threshold: float = 0.08) -> bool:
        return float(heatmap_confidences.get(key, 0.0) or 0.0) >= threshold

    def set_point(key: str, point: Any, source: str, reason: str = "") -> None:
        if xy(point) is not None:
            hybrid[key] = point
            point_sources[key] = source
            if reason:
                per_point_qc.setdefault(key, {})["review_reason"] = reason

    # P1 remains the stable mask-left-boundary point from v0.3.4.
    set_point("P1_snout_tip", v034.get("P1_snout_tip"), "mask_left_boundary")

    for key in ("P2_eye_front", "P3_operculum_posterior", "P5_caudal_base_midpoint", "P6_caudal_fork_midpoint"):
        heat = heatmap_keypoints.get(key)
        if xy(heat) is not None and conf_good(key):
            set_point(key, heat, "heatmap_unet")
        else:
            review_reasons.append(f"{key}_heatmap_unreliable")
            if key == "P6_caudal_fork_midpoint":
                fallback = _mask_suggestion_for(key, mask_suggestions, geometric_suggestions)
                if xy(fallback) is not None:
                    set_point(key, fallback, "tail_gap_rule_fallback", "P6_heatmap_unreliable")

    if not _is_head_roi(hybrid.get("P2_eye_front"), fish_bbox):
        review_reasons.append("P2_heatmap_unreliable")
    if not _is_head_roi(hybrid.get("P3_operculum_posterior"), fish_bbox):
        review_reasons.append("P3_heatmap_head_region_check")

    # P4 remains a v0.3.4 point for now. In the 30-image comparison the
    # heatmap candidate is often plausible-looking but shifted along the
    # peduncle axis, which is costly for peduncle-length measurements.
    set_point("P4_peduncle_start_midpoint", v034.get("P4_peduncle_start_midpoint"), "v0.3.4_enhanced")

    # P7U/P7L: use heatmap when it agrees with mask-tail rule; otherwise prefer a good mask rule.
    for key in ("P7U_caudal_fin_upper_tip", "P7L_caudal_fin_lower_tip"):
        heat = heatmap_keypoints.get(key)
        mask = _mask_suggestion_for(key, mask_suggestions, geometric_suggestions)
        dist = distance_px(heat, mask)
        dist_mm = dist * mm_per_pixel if dist is not None else None
        per_point_qc.setdefault(key, {})["heatmap_to_mask_distance_mm"] = dist_mm
        quality_key = "P7U_mask_quality" if key.startswith("P7U") else "P7L_mask_quality"
        mask_good = _quality_good(qc_results.get(quality_key, "good")) and xy(mask) is not None
        if xy(heat) is not None and conf_good(key) and (dist_mm is None or dist_mm <= 8.0):
            set_point(key, heat, "heatmap_unet_agrees_with_mask_tail_rule")
        elif mask_good:
            set_point(key, mask, "mask_tail_rule", f"{key}_heatmap_mask_disagreement")
            review_reasons.append(f"{key}_heatmap_mask_disagreement")
        elif xy(heat) is not None:
            set_point(key, heat, "heatmap_unet")
            review_reasons.append(f"{key}_mask_tail_rule_failed")

    # P8 remains v0.3.4 because dorsal fins can confuse heatmaps. P9 uses a
    # mask-gated heatmap candidate because it has a few large outliers. P10/P11
    # use heatmap directly when confidence is usable; they are boundary points
    # and can sit one or two pixels outside the binary mask after thresholding.
    set_point("P8_body_depth_dorsal", v034.get("P8_body_depth_dorsal"), "v0.3.4_enhanced")
    key = "P9_body_depth_ventral"
    heat = heatmap_keypoints.get(key)
    if xy(heat) is not None and conf_good(key, threshold=0.06) and _point_in_mask(heat, fish_mask):
        set_point(key, heat, "heatmap_unet")
    else:
        set_point(key, v034.get(key), "v0.3.4_enhanced_fallback")
        review_reasons.append(f"{key}_heatmap_unreliable")
    for key in ("P10_peduncle_depth_dorsal", "P11_peduncle_depth_ventral"):
        heat = heatmap_keypoints.get(key)
        if xy(heat) is not None and conf_good(key, threshold=0.06):
            set_point(key, heat, "heatmap_unet")
        else:
            set_point(key, v034.get(key), "v0.3.4_enhanced_fallback")
            review_reasons.append(f"{key}_heatmap_unreliable")

    # C1-C4: try heatmap, but fall back when mask/axis checks look wrong.
    for key in (
        "C1_head_axis_point",
        "C2_trunk_axis_point",
        "C3_posterior_trunk_axis_point",
        "C4_peduncle_axis_point",
    ):
        heat = heatmap_keypoints.get(key)
        if xy(heat) is not None and conf_good(key, threshold=0.06) and _point_in_mask(heat, fish_mask):
            hybrid[key] = heat
            point_sources[key] = "heatmap_unet"
        else:
            point_sources[key] = "v0.3.4_enhanced_fallback"
            review_reasons.append(f"{key}_heatmap_axis_unreliable")
    if not _axis_monotonic(hybrid):
        for key in ("C1_head_axis_point", "C2_trunk_axis_point", "C3_posterior_trunk_axis_point", "C4_peduncle_axis_point"):
            hybrid[key] = v034.get(key)
            point_sources[key] = "v0.3.4_enhanced_axis_fallback"
        review_reasons.append("heatmap_axis_qc_failed")

    p7v = derive_hybrid_p7v(hybrid)
    p7v_valid, p7v_reason = _p7v_validity(hybrid, p7v, fish_bbox)
    p7v["p7v_valid"] = p7v_valid
    p7v["p7v_invalid_reason"] = p7v_reason
    if not p7v_valid:
        review_reasons.append("invalid_p7v")
        if p7v_reason:
            review_reasons.append(p7v_reason)

    hybrid_review_reason = ";".join(dict.fromkeys([r for r in review_reasons if r]))
    return {
        "hybrid_keypoints": hybrid,
        "point_sources": point_sources,
        "hybrid_qc_results": {
            "hybrid_qc_pass": not bool(hybrid_review_reason),
            "hybrid_review_reason": hybrid_review_reason,
            "per_point_qc": per_point_qc,
            "p7v": p7v,
            "p7v_valid": p7v_valid,
            "p7v_invalid_reason": p7v_reason,
        },
    }
