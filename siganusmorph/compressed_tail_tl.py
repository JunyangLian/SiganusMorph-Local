"""Compressed-tail virtual total-length helpers.

The historical P7V/TL workflow projects caudal-fin tips onto a tail axis.
This module adds a parallel radius-preserving interpretation for the common
manual practice of gently closing the caudal lobes toward the body axis.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


Point = tuple[float, float]

ALIASES: dict[str, tuple[str, ...]] = {
    "P4": ("P4_peduncle_start_midpoint", "peduncle_start_midpoint"),
    "P5": ("P5_caudal_base_midpoint", "caudal_base_midpoint"),
    "P7U": ("P7U_caudal_fin_upper_tip", "caudal_fin_upper_tip"),
    "P7L": ("P7L_caudal_fin_lower_tip", "caudal_fin_lower_tip"),
}


def xy(value: Any) -> Point | None:
    if isinstance(value, Mapping):
        if "x" in value and "y" in value:
            try:
                return float(value["x"]), float(value["y"])
            except (TypeError, ValueError):
                return None
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None
    return None


def point_from_keypoints(keypoints: Mapping[str, Any], logical_name: str) -> Point | None:
    for key in ALIASES.get(logical_name, (logical_name,)):
        point = xy(keypoints.get(key))
        if point is not None:
            return point
    return None


def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def unit(start: Point, end: Point) -> Point | None:
    length = dist(start, end)
    if length <= 1e-9:
        return None
    return (end[0] - start[0]) / length, (end[1] - start[1]) / length


def dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def angle_between(a: Point, b: Point) -> float | None:
    na = math.hypot(a[0], a[1])
    nb = math.hypot(b[0], b[1])
    if na <= 1e-9 or nb <= 1e-9:
        return None
    value = max(-1.0, min(1.0, dot(a, b) / (na * nb)))
    return math.degrees(math.acos(value))


def _tail_axis_from_inputs(P5: Point, P7U: Point, P7L: Point, P4: Point | None = None) -> tuple[Point | None, str]:
    midpoint = ((P7U[0] + P7L[0]) / 2.0, (P7U[1] + P7L[1]) / 2.0)
    axis = unit(P5, midpoint)
    if axis is not None:
        return axis, "P5_to_midpoint_P7U_P7L"

    upper_axis = unit(P5, P7U)
    lower_axis = unit(P5, P7L)
    if upper_axis is not None and lower_axis is not None:
        bx = upper_axis[0] + lower_axis[0]
        by = upper_axis[1] + lower_axis[1]
        length = math.hypot(bx, by)
        if length > 1e-9:
            return (bx / length, by / length), "tail_lobe_bisector"

    if P4 is not None:
        axis = unit(P4, P5)
        if axis is not None:
            return axis, "P4_to_P5_fallback"
    return None, "tail_axis_failed"


def derive_p7v_open_projection(
    P5: Point,
    P7U: Point,
    P7L: Point,
    tail_axis_unit: Point,
) -> dict[str, Any]:
    """Project open caudal tips onto ``tail_axis_unit`` and choose the farther projection."""
    proj_upper = dot((P7U[0] - P5[0], P7U[1] - P5[1]), tail_axis_unit)
    proj_lower = dot((P7L[0] - P5[0], P7L[1] - P5[1]), tail_axis_unit)
    selected = "upper" if proj_upper >= proj_lower else "lower"
    extension = max(0.0, float(max(proj_upper, proj_lower)))
    point = (P5[0] + extension * tail_axis_unit[0], P5[1] + extension * tail_axis_unit[1])
    return {
        "P7V_open_projection": {"x": point[0], "y": point[1]},
        "open_projection_extension_px": extension,
        "open_projection_selected_lobe": selected,
        "open_projection_proj_upper_px": float(proj_upper),
        "open_projection_proj_lower_px": float(proj_lower),
    }


def derive_p7v_compressed_tail(
    P5: Any,
    P7U: Any,
    P7L: Any,
    tail_axis_unit: Any | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive a virtual P7V by rotating the longest caudal lobe onto the tail axis."""
    cfg = dict(config or {})
    p5 = xy(P5)
    p7u = xy(P7U)
    p7l = xy(P7L)
    p4 = xy(cfg.get("P4"))
    axis = xy(tail_axis_unit)
    source = str(cfg.get("tail_axis_source") or "provided")
    reasons: list[str] = []
    if p5 is None or p7u is None or p7l is None:
        return {
            "P7V_compressed_virtual": None,
            "compressed_tail_tl_valid_qc_pass": False,
            "compressed_tail_tl_qc_pass": False,
            "compressed_tail_tl_review_required": True,
            "compressed_tail_tl_review_reason": "missing_P5_or_tail_tip_points",
        }
    if axis is None:
        axis, source = _tail_axis_from_inputs(p5, p7u, p7l, p4)
    if axis is None:
        return {
            "P7V_compressed_virtual": None,
            "compressed_tail_tl_valid_qc_pass": False,
            "compressed_tail_tl_qc_pass": False,
            "compressed_tail_tl_review_required": True,
            "compressed_tail_tl_review_reason": "tail_axis_unavailable",
        }

    ru = dist(p5, p7u)
    rl = dist(p5, p7l)
    rmax = max(ru, rl)
    p7v = (p5[0] + rmax * axis[0], p5[1] + rmax * axis[1])
    upper_vec = (p7u[0] - p5[0], p7u[1] - p5[1])
    lower_vec = (p7l[0] - p5[0], p7l[1] - p5[1])
    upper_angle = angle_between(upper_vec, axis)
    lower_angle = angle_between(lower_vec, axis)
    tail_open_angle = angle_between(upper_vec, lower_vec)
    asym = abs(ru - rl) / max(ru, rl, 1e-9)

    if ru <= 1e-6 or rl <= 1e-6:
        reasons.append("invalid_lobe_length")
    if asym > float(cfg.get("lobe_asymmetry_review_threshold", 0.20)):
        reasons.append("tail_lobe_length_asymmetry_high")
    valid_qc_pass = not reasons

    return {
        "P7V_compressed_virtual": {"x": p7v[0], "y": p7v[1]},
        "tail_axis_unit": {"x": axis[0], "y": axis[1]},
        "tail_axis_source_compressed": source,
        "tail_axis_uses_P6": False,
        "RU_px": float(ru),
        "RL_px": float(rl),
        "Rmax_px": float(rmax),
        "upper_lobe_angle_to_axis_deg": float(upper_angle) if upper_angle is not None else None,
        "lower_lobe_angle_to_axis_deg": float(lower_angle) if lower_angle is not None else None,
        "inter_lobe_open_angle_deg": float(tail_open_angle) if tail_open_angle is not None else None,
        "upper_lobe_angle_deg": float(upper_angle) if upper_angle is not None else None,
        "lower_lobe_angle_deg": float(lower_angle) if lower_angle is not None else None,
        "tail_open_angle_deg": float(tail_open_angle) if tail_open_angle is not None else None,
        "tail_lobe_length_asymmetry_ratio": float(asym),
        "compressed_tail_tl_valid_qc_pass": valid_qc_pass,
        "compressed_tail_tl_qc_pass": valid_qc_pass,
        "compressed_tail_tl_review_required": not valid_qc_pass,
        "compressed_tail_tl_review_reason": ";".join(reasons),
    }


def derive_tail_length_comparison(
    keypoints: Mapping[str, Any],
    mm_per_pixel: float,
    *,
    sl_mm: float | None = None,
    open_projection_endpoint: Mapping[str, Any] | None = None,
    open_projection_extension_px: float | None = None,
    open_projection_tl_mm: float | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return open-projection and compressed-tail TL fields without mutating keypoints."""
    p4 = point_from_keypoints(keypoints, "P4")
    p5 = point_from_keypoints(keypoints, "P5")
    p7u = point_from_keypoints(keypoints, "P7U")
    p7l = point_from_keypoints(keypoints, "P7L")
    if p5 is None or p7u is None or p7l is None:
        return {
            "compressed_tail_tl_valid_qc_pass": False,
            "compressed_tail_tl_qc_pass": False,
            "compressed_tail_tl_review_required": True,
            "compressed_tail_tl_review_reason": "missing_tail_points",
        }
    axis, axis_source = _tail_axis_from_inputs(p5, p7u, p7l, p4)
    if axis is None:
        return {
            "compressed_tail_tl_valid_qc_pass": False,
            "compressed_tail_tl_qc_pass": False,
            "compressed_tail_tl_review_required": True,
            "compressed_tail_tl_review_reason": "tail_axis_unavailable",
        }
    open_projection = derive_p7v_open_projection(p5, p7u, p7l, axis)
    compressed = derive_p7v_compressed_tail(
        p5,
        p7u,
        p7l,
        axis,
        {"P4": p4, "tail_axis_source": axis_source, **dict(config or {})},
    )

    if open_projection_endpoint:
        legacy_open = xy(open_projection_endpoint)
        if legacy_open is not None:
            open_projection["P7V_open_projection"] = {"x": legacy_open[0], "y": legacy_open[1]}
    if open_projection_extension_px is not None:
        open_projection["open_projection_extension_px"] = float(open_projection_extension_px)

    sl_value = float(sl_mm) if sl_mm not in (None, "") else None
    open_extension_mm = float(open_projection["open_projection_extension_px"]) * mm_per_pixel
    compressed_extension_mm = float(compressed.get("Rmax_px", 0.0) or 0.0) * mm_per_pixel
    tl_open = float(open_projection_tl_mm) if open_projection_tl_mm not in (None, "") else (
        sl_value + open_extension_mm if sl_value is not None else None
    )
    tl_compressed = sl_value + compressed_extension_mm if sl_value is not None else None
    diff = (tl_compressed - tl_open) if tl_compressed is not None and tl_open is not None else None
    diff_pct = (100.0 * diff / tl_open) if diff is not None and tl_open and abs(tl_open) > 1e-9 else None

    valid_qc_reasons = [
        part for part in str(compressed.get("compressed_tail_tl_review_reason", "")).split(";") if part
    ]
    difference_reasons: list[str] = []
    if diff is not None and abs(diff) > float((config or {}).get("tl_difference_review_mm", 5.0)):
        difference_reasons.append("TL_difference_gt_5mm")
    if diff_pct is not None and abs(diff_pct) > float((config or {}).get("tl_difference_review_percent", 3.0)):
        difference_reasons.append("TL_difference_percent_gt_3")
    valid_qc_pass = bool(compressed.get("compressed_tail_tl_valid_qc_pass", compressed.get("compressed_tail_tl_qc_pass", False)))
    difference_flag = bool(difference_reasons)
    review_required = not valid_qc_pass

    return {
        **open_projection,
        **compressed,
        "P7V_open_projection_x": open_projection["P7V_open_projection"]["x"],
        "P7V_open_projection_y": open_projection["P7V_open_projection"]["y"],
        "P7V_compressed_virtual_x": (compressed.get("P7V_compressed_virtual") or {}).get("x"),
        "P7V_compressed_virtual_y": (compressed.get("P7V_compressed_virtual") or {}).get("y"),
        "TL_open_projection_mm": tl_open,
        "TL_compressed_virtual_mm": tl_compressed,
        "TL_difference_mm": diff,
        "TL_difference_percent": diff_pct,
        "RU_mm": compressed.get("RU_px", 0.0) * mm_per_pixel,
        "RL_mm": compressed.get("RL_px", 0.0) * mm_per_pixel,
        "Rmax_mm": compressed.get("Rmax_px", 0.0) * mm_per_pixel,
        "compressed_tail_tl_valid_qc_pass": valid_qc_pass,
        "compressed_tail_tl_qc_pass": valid_qc_pass,
        "compressed_vs_projection_difference_flag": difference_flag,
        "compressed_vs_projection_difference_reason": ";".join(dict.fromkeys(difference_reasons)),
        "compressed_tail_tl_review_required": review_required,
        "compressed_tail_tl_review_reason": ";".join(dict.fromkeys(valid_qc_reasons)),
    }
