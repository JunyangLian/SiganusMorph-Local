"""Local-normal measurement suggestions and curvature-aware QC.

These helpers are deliberately read-only with respect to annotations: they
generate geometric suggestions and QC metadata that can be shown in the UI or
benchmarked against manual labels, but they do not decide training labels.
"""

from __future__ import annotations

from math import acos, degrees, hypot
from typing import Any, Mapping, Sequence

import numpy as np

from .axis_utils import chord_length_cubic_spline_points, point_xy, polyline_length


PointLike = Mapping[str, float] | tuple[float, float] | list[float]

BODY_AXIS_KEYS = (
    "P1_snout_tip",
    "C1_head_axis_point",
    "C2_trunk_axis_point",
    "C3_posterior_trunk_axis_point",
    "P4_peduncle_start_midpoint",
    "C4_peduncle_axis_point",
    "P5_caudal_base_midpoint",
)

SHORT_TO_FULL = {
    "snout_tip": "P1_snout_tip",
    "head_axis_point": "C1_head_axis_point",
    "trunk_axis_point": "C2_trunk_axis_point",
    "posterior_trunk_axis_point": "C3_posterior_trunk_axis_point",
    "peduncle_start_midpoint": "P4_peduncle_start_midpoint",
    "peduncle_axis_point": "C4_peduncle_axis_point",
    "caudal_base_midpoint": "P5_caudal_base_midpoint",
    "body_depth_dorsal": "P8_body_depth_dorsal",
    "body_depth_ventral": "P9_body_depth_ventral",
    "peduncle_depth_dorsal": "P10_peduncle_depth_dorsal",
    "peduncle_depth_ventral": "P11_peduncle_depth_ventral",
}

DEFAULT_CONFIG = {
    "mm_per_pixel": 0.1,
    "axis_sample_count": 220,
    "body_sample_count": 120,
    "peduncle_sample_count": 70,
    "body_s_min": 0.20,
    "body_s_max": 0.75,
    "peduncle_s_min": 0.78,
    "peduncle_s_max": 0.95,
    "normal_max_radius_px": 520,
    "min_section_width_px": 8,
    "median_kernel": 5,
    "body_spike_ratio": 1.18,
    "body_trunk_width_spike_ratio": 1.10,
    "body_trunk_endpoint_spike_min_px": 8,
    "body_trunk_endpoint_spike_width_fraction": 0.10,
    "body_trunk_current_depth_overestimate_ratio": 1.15,
    "peduncle_boundary_margin": 0.08,
    "body_diff_threshold_mm": 5.0,
    "peduncle_diff_threshold_mm": 3.0,
}


def _as_mask(mask: Any) -> np.ndarray | None:
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(bool)


def _xy_payload(point: PointLike | None) -> tuple[float, float] | None:
    if point is None:
        return None
    try:
        return point_xy(point)
    except Exception:
        return None


def _point(keypoints: Mapping[str, Any], key: str) -> tuple[float, float] | None:
    if key in keypoints:
        return _xy_payload(keypoints.get(key))
    short = key.split("_", 1)[1] if "_" in key else key
    if short in keypoints:
        return _xy_payload(keypoints.get(short))
    full = SHORT_TO_FULL.get(short)
    if full and full in keypoints:
        return _xy_payload(keypoints.get(full))
    return None


def _body_axis_control_points(keypoints: Mapping[str, Any]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for key in BODY_AXIS_KEYS:
        point = _point(keypoints, key)
        if point is not None:
            points.append(point)
    return points


def _cumdist(points: Sequence[tuple[float, float]]) -> np.ndarray:
    if len(points) == 0:
        return np.asarray([], dtype=float)
    values = [0.0]
    for a, b in zip(points, points[1:]):
        values.append(values[-1] + hypot(b[0] - a[0], b[1] - a[1]))
    return np.asarray(values, dtype=float)


def _sample_axis(
    keypoints: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    control = _body_axis_control_points(keypoints)
    if len(control) < 2:
        return {
            "axis_curve_type": "failed",
            "axis_curve_quality": "failed",
            "control_points": control,
            "sampled_points": [],
            "s_values": [],
            "length_px": 0.0,
            "review_reason": "not_enough_axis_points",
        }
    sample_count = int(config.get("axis_sample_count", DEFAULT_CONFIG["axis_sample_count"]))
    axis_type = "polyline"
    quality = "good"
    try:
        if len(control) >= 4:
            sampled = chord_length_cubic_spline_points(control, sample_count=sample_count)
            axis_type = "spline"
        else:
            sampled = _resample_polyline(control, sample_count)
    except Exception:
        sampled = _resample_polyline(control, sample_count)
        axis_type = "polyline"
        quality = "warning"
    if len(sampled) < 2:
        sampled = control
        axis_type = "polyline"
        quality = "warning"
    s_values = _cumdist(sampled)
    return {
        "axis_curve_type": axis_type,
        "axis_curve_quality": quality,
        "control_points": control,
        "sampled_points": sampled,
        "s_values": s_values.tolist(),
        "length_px": float(s_values[-1]) if len(s_values) else 0.0,
        "review_reason": "",
    }


def _resample_polyline(points: Sequence[tuple[float, float]], sample_count: int) -> list[tuple[float, float]]:
    if len(points) <= 1:
        return list(points)
    cumulative = _cumdist(points)
    total = float(cumulative[-1])
    if total <= 0:
        return list(points)
    targets = np.linspace(0, total, max(2, int(sample_count)))
    out: list[tuple[float, float]] = []
    segment = 0
    for target in targets:
        while segment < len(cumulative) - 2 and cumulative[segment + 1] < target:
            segment += 1
        start_s = cumulative[segment]
        end_s = cumulative[segment + 1]
        if end_s <= start_s:
            out.append(points[segment])
            continue
        ratio = (target - start_s) / (end_s - start_s)
        x = points[segment][0] + ratio * (points[segment + 1][0] - points[segment][0])
        y = points[segment][1] + ratio * (points[segment + 1][1] - points[segment][1])
        out.append((float(x), float(y)))
    return out


def _merge_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_CONFIG)
    if config:
        merged.update(dict(config))
    return merged


def _nearest_index_for_point(axis: Sequence[tuple[float, float]], target: tuple[float, float] | None) -> int | None:
    if target is None or not axis:
        return None
    arr = np.asarray(axis, dtype=float)
    dist = np.hypot(arr[:, 0] - target[0], arr[:, 1] - target[1])
    return int(np.argmin(dist))


def _index_range_from_s(
    s_values: np.ndarray,
    s_min_ratio: float,
    s_max_ratio: float,
    *,
    start_index: int | None = None,
    end_index: int | None = None,
) -> tuple[int, int]:
    if len(s_values) < 2:
        return 0, 0
    total = float(s_values[-1])
    lo = int(np.searchsorted(s_values, total * float(s_min_ratio), side="left"))
    hi = int(np.searchsorted(s_values, total * float(s_max_ratio), side="right")) - 1
    if start_index is not None and end_index is not None:
        lo = min(start_index, end_index)
        hi = max(start_index, end_index)
    lo = max(1, min(lo, len(s_values) - 2))
    hi = max(lo + 1, min(hi, len(s_values) - 2))
    return lo, hi


def _tangent_at(axis: Sequence[tuple[float, float]], index: int) -> tuple[float, float] | None:
    if len(axis) < 2:
        return None
    i0 = max(0, index - 2)
    i1 = min(len(axis) - 1, index + 2)
    dx = axis[i1][0] - axis[i0][0]
    dy = axis[i1][1] - axis[i0][1]
    length = hypot(dx, dy)
    if length <= 1e-9:
        return None
    return dx / length, dy / length


def _sample_section(
    mask: np.ndarray,
    center: tuple[float, float],
    normal: tuple[float, float],
    max_radius: int,
    min_width_px: float,
) -> dict[str, Any] | None:
    height, width = mask.shape[:2]
    ts = np.arange(-int(max_radius), int(max_radius) + 1, dtype=float)
    xs = np.rint(center[0] + ts * normal[0]).astype(int)
    ys = np.rint(center[1] + ts * normal[1]).astype(int)
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    inside = np.zeros_like(valid, dtype=bool)
    inside[valid] = mask[ys[valid], xs[valid]]
    if not inside.any():
        return None

    zero_idx = int(max_radius)
    true_indices = np.flatnonzero(inside)
    if len(true_indices) == 0:
        return None
    if inside[min(max(zero_idx, 0), len(inside) - 1)]:
        start = zero_idx
        while start > 0 and inside[start - 1]:
            start -= 1
        end = zero_idx
        while end < len(inside) - 1 and inside[end + 1]:
            end += 1
    else:
        # Pick the contiguous mask segment closest to the axis center.
        runs: list[tuple[int, int]] = []
        start = int(true_indices[0])
        prev = start
        for idx in true_indices[1:]:
            idx = int(idx)
            if idx == prev + 1:
                prev = idx
            else:
                runs.append((start, prev))
                start = prev = idx
        runs.append((start, prev))
        start, end = min(runs, key=lambda run: min(abs(run[0] - zero_idx), abs(run[1] - zero_idx)))

    p_a = (float(xs[start]), float(ys[start]))
    p_b = (float(xs[end]), float(ys[end]))
    width_px = hypot(p_b[0] - p_a[0], p_b[1] - p_a[1])
    if width_px < float(min_width_px):
        return None
    upper, lower = (p_a, p_b) if p_a[1] <= p_b[1] else (p_b, p_a)
    section_center = ((upper[0] + lower[0]) / 2, (upper[1] + lower[1]) / 2)
    return {
        "upper": upper,
        "lower": lower,
        "width_px": width_px,
        "section_center": section_center,
        "t_start": float(ts[start]),
        "t_end": float(ts[end]),
    }


def _median_filter(values: Sequence[float], kernel: int = 5) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return arr
    kernel = max(1, int(kernel))
    if kernel % 2 == 0:
        kernel += 1
    pad = kernel // 2
    out = np.empty_like(arr)
    for i in range(len(arr)):
        lo = max(0, i - pad)
        hi = min(len(arr), i + pad + 1)
        window = arr[lo:hi]
        valid = window[np.isfinite(window)]
        out[i] = np.median(valid) if len(valid) else np.nan
    return out


def _scan_sections(
    mask: np.ndarray,
    axis_info: Mapping[str, Any],
    lo: int,
    hi: int,
    sample_count: int,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    axis = list(axis_info.get("sampled_points", []) or [])
    s_values = np.asarray(axis_info.get("s_values", []) or [], dtype=float)
    if len(axis) < 2 or len(s_values) != len(axis):
        return []
    lo = max(1, min(int(lo), len(axis) - 2))
    hi = max(lo, min(int(hi), len(axis) - 2))
    indices = np.linspace(lo, hi, max(2, int(sample_count))).round().astype(int)
    sections: list[dict[str, Any]] = []
    for idx in indices:
        tangent = _tangent_at(axis, int(idx))
        if tangent is None:
            continue
        normal = (-tangent[1], tangent[0])
        section = _sample_section(
            mask,
            axis[int(idx)],
            normal,
            int(config.get("normal_max_radius_px", DEFAULT_CONFIG["normal_max_radius_px"])),
            float(config.get("min_section_width_px", DEFAULT_CONFIG["min_section_width_px"])),
        )
        if section is None:
            continue
        section.update(
            {
                "axis_index": int(idx),
                "s_px": float(s_values[int(idx)]),
                "axis_point": [float(axis[int(idx)][0]), float(axis[int(idx)][1])],
                "section_tangent": [float(tangent[0]), float(tangent[1])],
                "section_normal": [float(normal[0]), float(normal[1])],
            }
        )
        sections.append(section)
    return sections


def _point_distance_mm(a: tuple[float, float] | None, b: tuple[float, float] | None, mm_per_pixel: float) -> float | None:
    if a is None or b is None:
        return None
    return hypot(a[0] - b[0], a[1] - b[1]) * mm_per_pixel


def _section_payload(
    section: Mapping[str, Any] | None,
    mm_per_pixel: float,
    *,
    upper_key: str,
    lower_key: str,
    qc_pass: bool,
    review_reason: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not section:
        return {
            upper_key: None,
            lower_key: None,
            "width_px": None,
            "width_mm": None,
            "section_center": None,
            "section_tangent": None,
            "section_normal": None,
            "qc_pass": False,
            "review_reason": review_reason or "no_valid_section",
        }
    payload = {
        upper_key: [float(section["upper"][0]), float(section["upper"][1])],
        lower_key: [float(section["lower"][0]), float(section["lower"][1])],
        "width_px": float(section["width_px"]),
        "width_mm": float(section["width_px"]) * float(mm_per_pixel),
        "section_center": [float(section["section_center"][0]), float(section["section_center"][1])],
        "section_tangent": list(section.get("section_tangent", [])),
        "section_normal": list(section.get("section_normal", [])),
        "axis_point": list(section.get("axis_point", [])),
        "s_px": section.get("s_px"),
        "qc_pass": bool(qc_pass),
        "review_reason": review_reason,
    }
    if extra:
        payload.update(dict(extra))
    return payload


def estimate_body_depth_by_local_normals(
    fish_mask: Any,
    axis_curve: Mapping[str, Any] | None,
    keypoints: Mapping[str, Any],
    image_shape: Sequence[int] | None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Find the broadest body section by scanning local normals through mask."""
    del image_shape
    cfg = _merge_config(config)
    mm_per_pixel = float(cfg.get("mm_per_pixel", 0.1))
    mask = _as_mask(fish_mask)
    if mask is None or not mask.any():
        return {"qc_pass": False, "review_reason": "fish_mask_unavailable", "axis_curve": axis_curve or {}}
    axis_info = dict(axis_curve or _sample_axis(keypoints, cfg))
    if not axis_info.get("sampled_points"):
        axis_info = _sample_axis(keypoints, cfg)
    s_values = np.asarray(axis_info.get("s_values", []) or [], dtype=float)
    if len(s_values) < 3:
        return {"qc_pass": False, "review_reason": "axis_curve_failed", "axis_curve": axis_info}

    c1_idx = _nearest_index_for_point(axis_info["sampled_points"], _point(keypoints, "C1_head_axis_point"))
    c3_idx = _nearest_index_for_point(axis_info["sampled_points"], _point(keypoints, "C3_posterior_trunk_axis_point"))
    lo, hi = _index_range_from_s(
        s_values,
        float(cfg["body_s_min"]),
        float(cfg["body_s_max"]),
        start_index=c1_idx,
        end_index=c3_idx,
    )
    sections = _scan_sections(mask, axis_info, lo, hi, int(cfg["body_sample_count"]), cfg)
    if not sections:
        return {"qc_pass": False, "review_reason": "no_valid_body_depth_sections", "axis_curve": axis_info}
    widths = np.asarray([float(s["width_px"]) for s in sections], dtype=float)
    smooth = _median_filter(widths, int(cfg["median_kernel"]))
    best_index = int(np.nanargmax(smooth))
    best = sections[best_index]
    raw_best_index = int(np.nanargmax(widths))
    raw_best = sections[raw_best_index]
    review: list[str] = []
    qc_pass = True
    if widths[raw_best_index] > max(1.0, smooth[best_index]) * float(cfg["body_spike_ratio"]):
        qc_pass = False
        review.append("body_depth_width_spike_possible_fin_interference")
    if best_index in {0, len(sections) - 1}:
        qc_pass = False
        review.append("body_depth_section_on_search_boundary")

    original_width = _point_distance_mm(
        _point(keypoints, "P8_body_depth_dorsal"),
        _point(keypoints, "P9_body_depth_ventral"),
        mm_per_pixel,
    )
    refined_width = float(best["width_px"]) * mm_per_pixel
    diff_mm = None if original_width is None else refined_width - original_width
    diff_threshold = float(cfg["body_diff_threshold_mm"])
    if diff_mm is not None and abs(diff_mm) > diff_threshold:
        review.append("body_depth_refined_differs_from_current_points")

    return _section_payload(
        best,
        mm_per_pixel,
        upper_key="P8_refined",
        lower_key="P9_refined",
        qc_pass=qc_pass,
        review_reason=";".join(review),
        extra={
            "axis_curve": axis_info,
            "search_range": [int(lo), int(hi)],
            "width_profile_px": [float(v) for v in widths],
            "width_profile_smooth_px": [float(v) for v in smooth],
            "raw_max_width_px": float(raw_best["width_px"]),
            "raw_max_point": [float(raw_best["section_center"][0]), float(raw_best["section_center"][1])],
            "body_depth_original_mm": original_width,
            "body_depth_refined_minus_original_mm": diff_mm,
        },
    )


def estimate_body_depth_by_body_midline_normals(
    fish_mask: Any,
    body_core_mask: Any,
    body_midline_axis: Mapping[str, Any] | Sequence[tuple[float, float]] | Sequence[list[float]],
    keypoints: Mapping[str, Any],
    image_shape: Sequence[int] | None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate P8/P9 from trunk boundaries using body-midline normals.

    v0.6.9 replaces the earlier geometric body-depth logic in-place.  The
    body_midline_axis still supplies the local tangent/normal, but the selected
    endpoint pair is filtered against fin-like spikes before the maximum stable
    trunk section is chosen.
    """
    trunk = build_body_trunk_mask_for_depth(fish_mask, body_midline_axis, keypoints, image_shape, config)
    return estimate_body_depth_by_trunk_normal_sections(trunk, body_midline_axis, keypoints, image_shape, config)


def build_body_trunk_mask_for_depth(
    fish_mask: Any,
    body_midline_axis: Mapping[str, Any] | Sequence[tuple[float, float]] | Sequence[list[float]],
    keypoints: Mapping[str, Any],
    image_shape: Sequence[int] | None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a trunk-boundary section set for body-depth measurement.

    The returned object is intentionally serializable: it stores sampled outer
    trunk contours and suppression flags rather than a dense ndarray.
    """
    del image_shape
    del keypoints
    cfg = _merge_config(config)
    mask = _as_mask(fish_mask)
    if mask is None or not mask.any():
        return {
            "body_trunk_qc": {"body_trunk_qc_pass": False, "body_trunk_qc_reason": "fish_mask_unavailable"},
            "sections": [],
            "body_depth_source": "body_midline_normal_trunk_boundary_max_width",
            "body_depth_geometry_version": "v0.6.9_morphometric_definition_fix",
        }
    if isinstance(body_midline_axis, Mapping):
        midline_points = body_midline_axis.get("body_midline_points") or body_midline_axis.get("sampled_points") or []
        qc = body_midline_axis.get("body_midline_qc", {}) if isinstance(body_midline_axis.get("body_midline_qc", {}), Mapping) else {}
        midline_qc_pass = bool(qc.get("body_midline_qc_pass", True))
        midline_reason = str(qc.get("body_midline_qc_reason", ""))
    else:
        midline_points = body_midline_axis or []
        midline_qc_pass = True
        midline_reason = ""
    axis_points: list[tuple[float, float]] = []
    for point in midline_points:
        xy = _xy_payload(point)
        if xy is not None:
            axis_points.append(xy)
    if len(axis_points) < 3:
        return {
            "body_trunk_qc": {"body_trunk_qc_pass": False, "body_trunk_qc_reason": "body_midline_axis_unavailable"},
            "sections": [],
            "body_depth_source": "body_midline_normal_trunk_boundary_max_width",
            "body_depth_geometry_version": "v0.6.9_morphometric_definition_fix",
        }

    s_values = _cumdist(axis_points)
    axis_info = {
        "axis_curve_type": "body_midline_polyline",
        "axis_curve_quality": "good" if midline_qc_pass else "warning",
        "control_points": axis_points,
        "sampled_points": axis_points,
        "s_values": s_values.tolist(),
        "length_px": float(s_values[-1]) if len(s_values) else 0.0,
        "review_reason": midline_reason,
    }
    lo, hi = _index_range_from_s(
        s_values,
        float(cfg.get("body_midline_body_s_min", 0.20)),
        float(cfg.get("body_midline_body_s_max", 0.85)),
    )
    sections = _scan_sections(mask, axis_info, lo, hi, int(cfg.get("body_sample_count", DEFAULT_CONFIG["body_sample_count"])), cfg)
    if not sections:
        return {
            "body_trunk_qc": {"body_trunk_qc_pass": False, "body_trunk_qc_reason": "no_valid_body_trunk_sections"},
            "sections": [],
            "axis_curve": axis_info,
            "search_range": [int(lo), int(hi)],
            "body_depth_source": "body_midline_normal_trunk_boundary_max_width",
            "body_depth_geometry_version": "v0.6.9_morphometric_definition_fix",
        }

    widths = np.asarray([float(s["width_px"]) for s in sections], dtype=float)
    smooth_widths = _median_filter(widths, int(cfg.get("median_kernel", DEFAULT_CONFIG["median_kernel"])))
    upper_y = np.asarray([float(s["upper"][1]) for s in sections], dtype=float)
    lower_y = np.asarray([float(s["lower"][1]) for s in sections], dtype=float)
    upper_smooth = _median_filter(upper_y, int(cfg.get("median_kernel", DEFAULT_CONFIG["median_kernel"])))
    lower_smooth = _median_filter(lower_y, int(cfg.get("median_kernel", DEFAULT_CONFIG["median_kernel"])))
    median_width = float(np.nanmedian(widths)) if len(widths) else 0.0
    endpoint_threshold = max(
        float(cfg.get("body_trunk_endpoint_spike_min_px", DEFAULT_CONFIG["body_trunk_endpoint_spike_min_px"])),
        median_width * float(cfg.get("body_trunk_endpoint_spike_width_fraction", DEFAULT_CONFIG["body_trunk_endpoint_spike_width_fraction"])),
    )
    width_ratio = float(cfg.get("body_trunk_width_spike_ratio", DEFAULT_CONFIG["body_trunk_width_spike_ratio"]))
    suppressed: list[int] = []
    suppression_regions: list[dict[str, Any]] = []
    for idx, section in enumerate(sections):
        reasons: list[str] = []
        if np.isfinite(widths[idx]) and np.isfinite(smooth_widths[idx]) and widths[idx] > max(1.0, float(smooth_widths[idx])) * width_ratio:
            reasons.append("width_spike")
        if np.isfinite(upper_smooth[idx]) and upper_y[idx] < upper_smooth[idx] - endpoint_threshold:
            reasons.append("dorsal_fin_spike")
        if np.isfinite(lower_smooth[idx]) and lower_y[idx] > lower_smooth[idx] + endpoint_threshold:
            reasons.append("ventral_fin_spike")
        if reasons:
            suppressed.append(idx)
            suppression_regions.append(
                {
                    "index": int(idx),
                    "axis_index": int(section.get("axis_index", -1)),
                    "upper": [float(section["upper"][0]), float(section["upper"][1])],
                    "lower": [float(section["lower"][0]), float(section["lower"][1])],
                    "reasons": reasons,
                }
            )

    dorsal_contour = [[float(s["upper"][0]), float(s["upper"][1])] for s in sections]
    ventral_contour = [[float(s["lower"][0]), float(s["lower"][1])] for s in sections]
    trunk_qc_reason: list[str] = []
    if not midline_qc_pass:
        trunk_qc_reason.append("body_midline_axis_qc_warning")
    if len(suppressed) > max(3, len(sections) * 0.25):
        trunk_qc_reason.append("many_fin_like_sections_suppressed")
    return {
        "body_trunk_mask": None,
        "body_trunk_qc": {
            "body_trunk_qc_pass": not trunk_qc_reason,
            "body_trunk_qc_reason": ";".join(trunk_qc_reason),
            "suppressed_section_count": int(len(suppressed)),
            "section_count": int(len(sections)),
        },
        "sections": sections,
        "axis_curve": axis_info,
        "search_range": [int(lo), int(hi)],
        "widths": widths.tolist(),
        "widths_smooth": smooth_widths.tolist(),
        "suppressed_indices": suppressed,
        "fin_suppression_regions": suppression_regions,
        "dorsal_trunk_contour": dorsal_contour,
        "ventral_trunk_contour": ventral_contour,
        "dorsal_body_depth_contour": dorsal_contour,
        "ventral_body_depth_contour": ventral_contour,
        "body_depth_source": "body_midline_normal_trunk_boundary_max_width",
        "body_depth_geometry_version": "v0.6.9_morphometric_definition_fix",
    }


def estimate_body_depth_by_trunk_normal_sections(
    body_trunk_mask: Mapping[str, Any],
    body_midline_axis: Mapping[str, Any] | Sequence[tuple[float, float]] | Sequence[list[float]],
    keypoints: Mapping[str, Any],
    image_shape: Sequence[int] | None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Pick the widest stable trunk-normal section for P8/P9."""
    del body_midline_axis
    del image_shape
    cfg = _merge_config(config)
    mm_per_pixel = float(cfg.get("mm_per_pixel", 0.1))
    sections = list(body_trunk_mask.get("sections", []) or [])
    if not sections:
        reason = "No reliable body-depth boundary available"
        qc = body_trunk_mask.get("body_trunk_qc", {}) if isinstance(body_trunk_mask.get("body_trunk_qc", {}), Mapping) else {}
        if qc.get("body_trunk_qc_reason"):
            reason = str(qc.get("body_trunk_qc_reason"))
        return {
            "P8_geometric": None,
            "P9_geometric": None,
            "body_depth_geometric_mm": None,
            "body_depth_geometric_qc_pass": False,
            "body_depth_geometric_review_reason": reason,
            "body_depth_source": "body_midline_normal_trunk_boundary_max_width",
            "body_depth_geometry_version": "v0.6.9_morphometric_definition_fix",
            "body_trunk_qc": body_trunk_mask.get("body_trunk_qc", {}),
        }
    widths = np.asarray(body_trunk_mask.get("widths") or [float(s["width_px"]) for s in sections], dtype=float)
    smooth = np.asarray(body_trunk_mask.get("widths_smooth") or _median_filter(widths, int(cfg.get("median_kernel", DEFAULT_CONFIG["median_kernel"]))), dtype=float)
    suppressed = set(int(i) for i in (body_trunk_mask.get("suppressed_indices", []) or []))
    candidate_indices = [i for i in range(len(sections)) if i not in suppressed and np.isfinite(smooth[i])]
    review: list[str] = []
    qc_pass = True
    if not candidate_indices:
        candidate_indices = [i for i in range(len(sections)) if np.isfinite(smooth[i])]
        qc_pass = False
        review.append("all_candidate_sections_suppressed_by_fin_filter")
    best_index = max(candidate_indices, key=lambda i: float(smooth[i]))
    best = sections[best_index]
    raw_best_index = int(np.nanargmax(widths))
    if best_index in {0, len(sections) - 1}:
        qc_pass = False
        review.append("body_depth_geometric_section_on_search_boundary")
    if raw_best_index in suppressed:
        review.append("raw_max_width_suppressed_as_fin_like")
    trunk_qc = body_trunk_mask.get("body_trunk_qc", {}) if isinstance(body_trunk_mask.get("body_trunk_qc", {}), Mapping) else {}
    if not bool(trunk_qc.get("body_trunk_qc_pass", True)):
        qc_pass = False
        if trunk_qc.get("body_trunk_qc_reason"):
            review.append(str(trunk_qc.get("body_trunk_qc_reason")))

    original_width = _point_distance_mm(
        _point(keypoints, "P8_body_depth_dorsal"),
        _point(keypoints, "P9_body_depth_ventral"),
        mm_per_pixel,
    )
    width_mm = float(best["width_px"]) * mm_per_pixel
    diff_mm = None if original_width is None else width_mm - original_width
    if original_width is not None and original_width > 0 and original_width > width_mm * float(cfg.get("body_trunk_current_depth_overestimate_ratio", DEFAULT_CONFIG["body_trunk_current_depth_overestimate_ratio"])):
        review.append("P8/P9 may be attached to fins")
    if best_index in suppressed:
        qc_pass = False
        review.append("selected_section_is_suppressed_fin_region")

    selected_section = {
        "index": int(best_index),
        "axis_index": int(best.get("axis_index", -1)),
        "s_px": float(best.get("s_px", np.nan)),
        "raw_max_index": int(raw_best_index),
        "upper": [float(best["upper"][0]), float(best["upper"][1])],
        "lower": [float(best["lower"][0]), float(best["lower"][1])],
        "section_center": [float(best["section_center"][0]), float(best["section_center"][1])],
        "section_tangent": list(best.get("section_tangent", [])),
        "section_normal": list(best.get("section_normal", [])),
    }
    review_reason = ";".join(dict.fromkeys(reason for reason in review if reason))
    return _section_payload(
        best,
        mm_per_pixel,
        upper_key="P8_geometric",
        lower_key="P9_geometric",
        qc_pass=qc_pass,
        review_reason=review_reason,
        extra={
            "body_depth_source": "body_midline_normal_trunk_boundary_max_width",
            "body_depth_geometry_version": "v0.6.9_morphometric_definition_fix",
            "body_depth_geometric_mm": width_mm,
            "body_depth_geometric_qc_pass": qc_pass,
            "body_depth_geometric_review_reason": review_reason,
            "body_depth_review_suggested": "P8/P9 may be attached to fins" in review,
            "body_depth_section_tangent": list(best.get("section_tangent", [])),
            "body_depth_section_normal": list(best.get("section_normal", [])),
            "axis_curve": body_trunk_mask.get("axis_curve", {}),
            "search_range": body_trunk_mask.get("search_range", []),
            "body_depth_mask_source": "trunk_outer_boundary_with_fin_suppression",
            "dorsal_trunk_contour": body_trunk_mask.get("dorsal_trunk_contour", []),
            "ventral_trunk_contour": body_trunk_mask.get("ventral_trunk_contour", []),
            "dorsal_body_depth_contour": body_trunk_mask.get("dorsal_body_depth_contour", []),
            "ventral_body_depth_contour": body_trunk_mask.get("ventral_body_depth_contour", []),
            "fin_suppression_regions": body_trunk_mask.get("fin_suppression_regions", []),
            "body_trunk_qc": body_trunk_mask.get("body_trunk_qc", {}),
            "body_depth_boundary_qc": {
                "suppressed_section_count": len(suppressed),
                "raw_max_index": raw_best_index,
                "raw_max_suppressed": raw_best_index in suppressed,
                "selected_smooth_width_px": float(smooth[best_index]),
            },
            "body_depth_width_profile": [float(v) for v in widths],
            "body_depth_width_profile_smooth": [float(v) for v in smooth],
            "body_depth_selected_section": selected_section,
            "body_depth_original_mm": original_width,
            "body_depth_geometric_minus_original_mm": diff_mm,
        },
    )


def estimate_peduncle_depth_by_local_normals(
    fish_mask: Any,
    axis_curve: Mapping[str, Any] | None,
    keypoints: Mapping[str, Any],
    image_shape: Sequence[int] | None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Find the narrowest caudal peduncle section by local-normal scanning."""
    del image_shape
    cfg = _merge_config(config)
    mm_per_pixel = float(cfg.get("mm_per_pixel", 0.1))
    mask = _as_mask(fish_mask)
    if mask is None or not mask.any():
        return {"qc_pass": False, "review_reason": "fish_mask_unavailable", "axis_curve": axis_curve or {}}
    axis_info = dict(axis_curve or _sample_axis(keypoints, cfg))
    if not axis_info.get("sampled_points"):
        axis_info = _sample_axis(keypoints, cfg)
    s_values = np.asarray(axis_info.get("s_values", []) or [], dtype=float)
    if len(s_values) < 3:
        return {"qc_pass": False, "review_reason": "axis_curve_failed", "axis_curve": axis_info}

    p4_idx = _nearest_index_for_point(axis_info["sampled_points"], _point(keypoints, "P4_peduncle_start_midpoint"))
    p5_idx = _nearest_index_for_point(axis_info["sampled_points"], _point(keypoints, "P5_caudal_base_midpoint"))
    lo, hi = _index_range_from_s(
        s_values,
        float(cfg["peduncle_s_min"]),
        float(cfg["peduncle_s_max"]),
        start_index=p4_idx,
        end_index=p5_idx,
    )
    sections = _scan_sections(mask, axis_info, lo, hi, int(cfg["peduncle_sample_count"]), cfg)
    if not sections:
        return {"qc_pass": False, "review_reason": "no_valid_peduncle_depth_sections", "axis_curve": axis_info}
    widths = np.asarray([float(s["width_px"]) for s in sections], dtype=float)
    smooth = _median_filter(widths, int(cfg["median_kernel"]))
    best_index = int(np.nanargmin(smooth))
    best = sections[best_index]
    review: list[str] = []
    qc_pass = True
    margin = max(1, int(round(len(sections) * float(cfg["peduncle_boundary_margin"]))))
    if best_index < margin or best_index >= len(sections) - margin:
        qc_pass = False
        review.append("peduncle_depth_section_on_boundary")

    original_width = _point_distance_mm(
        _point(keypoints, "P10_peduncle_depth_dorsal"),
        _point(keypoints, "P11_peduncle_depth_ventral"),
        mm_per_pixel,
    )
    refined_width = float(best["width_px"]) * mm_per_pixel
    diff_mm = None if original_width is None else refined_width - original_width
    diff_threshold = float(cfg["peduncle_diff_threshold_mm"])
    if diff_mm is not None and abs(diff_mm) > diff_threshold:
        review.append("peduncle_depth_refined_differs_from_current_points")

    return _section_payload(
        best,
        mm_per_pixel,
        upper_key="P10_refined",
        lower_key="P11_refined",
        qc_pass=qc_pass,
        review_reason=";".join(review),
        extra={
            "axis_curve": axis_info,
            "search_range": [int(lo), int(hi)],
            "width_profile_px": [float(v) for v in widths],
            "width_profile_smooth_px": [float(v) for v in smooth],
            "caudal_peduncle_depth_original_mm": original_width,
            "caudal_peduncle_depth_refined_minus_original_mm": diff_mm,
        },
    )


def estimate_peduncle_depth_by_axis_normals(
    fish_mask: Any,
    keypoints: Mapping[str, Any],
    image_shape: Sequence[int] | None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate P10/P11 by finding the narrowest normal section from P4 to P5.

    This v0.6.8 helper uses the caudal peduncle axis only. It generates a
    geometric suggestion for review/QC and never writes labels.
    """
    del image_shape
    cfg = _merge_config(config)
    mm_per_pixel = float(cfg.get("mm_per_pixel", 0.1))
    mask = _as_mask(fish_mask)
    if mask is None or not mask.any():
        return {
            "P10_geometric": None,
            "P11_geometric": None,
            "peduncle_depth_geometric_mm": None,
            "peduncle_depth_geometric_qc_pass": False,
            "peduncle_depth_geometric_review_reason": "fish_mask_unavailable",
            "peduncle_depth_source": "P4_P5_axis_normal_min_width",
            "peduncle_depth_geometry_version": "v0.6.8_peduncle_axis_normals",
        }
    p4 = _point(keypoints, "P4_peduncle_start_midpoint")
    p5 = _point(keypoints, "P5_caudal_base_midpoint")
    if p4 is None or p5 is None:
        return {
            "P10_geometric": None,
            "P11_geometric": None,
            "peduncle_depth_geometric_mm": None,
            "peduncle_depth_geometric_qc_pass": False,
            "peduncle_depth_geometric_review_reason": "P4_or_P5_unavailable",
            "peduncle_depth_source": "P4_P5_axis_normal_min_width",
            "peduncle_depth_geometry_version": "v0.6.8_peduncle_axis_normals",
        }
    axis_points = _resample_polyline([p4, p5], int(cfg.get("axis_sample_count", DEFAULT_CONFIG["axis_sample_count"])))
    s_values = _cumdist(axis_points)
    axis_info = {
        "axis_curve_type": "P4_P5_line",
        "axis_curve_quality": "good",
        "control_points": [p4, p5],
        "sampled_points": axis_points,
        "s_values": s_values.tolist(),
        "length_px": float(s_values[-1]) if len(s_values) else 0.0,
        "review_reason": "",
    }
    if len(axis_points) < 3 or float(axis_info["length_px"]) <= 1e-9:
        return {
            "P10_geometric": None,
            "P11_geometric": None,
            "peduncle_depth_geometric_mm": None,
            "peduncle_depth_geometric_qc_pass": False,
            "peduncle_depth_geometric_review_reason": "P4_P5_axis_too_short",
            "peduncle_depth_source": "P4_P5_axis_normal_min_width",
            "peduncle_depth_geometry_version": "v0.6.8_peduncle_axis_normals",
            "axis_curve": axis_info,
        }
    lo, hi = _index_range_from_s(
        s_values,
        float(cfg.get("peduncle_axis_s_min", 0.10)),
        float(cfg.get("peduncle_axis_s_max", 0.80)),
    )
    sections = _scan_sections(mask, axis_info, lo, hi, int(cfg.get("peduncle_sample_count", DEFAULT_CONFIG["peduncle_sample_count"])), cfg)
    if not sections:
        return {
            "P10_geometric": None,
            "P11_geometric": None,
            "peduncle_depth_geometric_mm": None,
            "peduncle_depth_geometric_qc_pass": False,
            "peduncle_depth_geometric_review_reason": "no_valid_peduncle_axis_normal_sections",
            "peduncle_depth_source": "P4_P5_axis_normal_min_width",
            "peduncle_depth_geometry_version": "v0.6.8_peduncle_axis_normals",
            "axis_curve": axis_info,
        }
    widths = np.asarray([float(s["width_px"]) for s in sections], dtype=float)
    smooth = _median_filter(widths, int(cfg.get("median_kernel", DEFAULT_CONFIG["median_kernel"])))
    best_index = int(np.nanargmin(smooth))
    best = sections[best_index]
    review: list[str] = []
    qc_pass = True
    margin = max(1, int(round(len(sections) * float(cfg.get("peduncle_boundary_margin", DEFAULT_CONFIG["peduncle_boundary_margin"])))))
    if best_index < margin or best_index >= len(sections) - margin:
        qc_pass = False
        review.append("peduncle_depth_section_on_boundary")
    raw_min_index = int(np.nanargmin(widths))
    raw_min = float(widths[raw_min_index])
    smooth_min = float(smooth[best_index])
    if smooth_min > 0 and raw_min < smooth_min * 0.75:
        qc_pass = False
        review.append("peduncle_depth_single_point_narrow_spike")
    original_width = _point_distance_mm(
        _point(keypoints, "P10_peduncle_depth_dorsal"),
        _point(keypoints, "P11_peduncle_depth_ventral"),
        mm_per_pixel,
    )
    width_mm = float(best["width_px"]) * mm_per_pixel
    diff_mm = None if original_width is None else width_mm - original_width
    if diff_mm is not None and abs(diff_mm) > float(cfg.get("peduncle_diff_threshold_mm", DEFAULT_CONFIG["peduncle_diff_threshold_mm"])):
        review.append("peduncle_depth_review_suggested")

    selected_section = {
        "index": int(best_index),
        "axis_index": int(best.get("axis_index", -1)),
        "s_px": float(best.get("s_px", np.nan)),
        "upper": [float(best["upper"][0]), float(best["upper"][1])],
        "lower": [float(best["lower"][0]), float(best["lower"][1])],
        "section_center": [float(best["section_center"][0]), float(best["section_center"][1])],
        "section_tangent": list(best.get("section_tangent", [])),
        "section_normal": list(best.get("section_normal", [])),
    }
    return _section_payload(
        best,
        mm_per_pixel,
        upper_key="P10_geometric",
        lower_key="P11_geometric",
        qc_pass=qc_pass,
        review_reason=";".join(review),
        extra={
            "peduncle_depth_source": "P4_P5_axis_normal_min_width",
            "peduncle_depth_geometry_version": "v0.6.8_peduncle_axis_normals",
            "peduncle_depth_geometric_mm": width_mm,
            "peduncle_depth_geometric_qc_pass": qc_pass,
            "peduncle_depth_geometric_review_reason": ";".join(review),
            "peduncle_depth_review_suggested": "peduncle_depth_review_suggested" in review,
            "peduncle_depth_selected_section": selected_section,
            "peduncle_depth_width_profile": [float(v) for v in widths],
            "peduncle_depth_width_profile_smooth": [float(v) for v in smooth],
            "peduncle_depth_original_mm": original_width,
            "peduncle_depth_geometric_minus_original_mm": diff_mm,
            "axis_curve": axis_info,
            "search_range": [int(lo), int(hi)],
        },
    )


def compute_curvature_qc(
    axis_curve: Mapping[str, Any] | Sequence[tuple[float, float]] | None,
    keypoints: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute curvature metrics and review prompts for curved specimens."""
    cfg = _merge_config(config)
    mm_per_pixel = float(cfg.get("mm_per_pixel", 0.1))
    if isinstance(axis_curve, Mapping):
        points = [tuple(map(float, p)) for p in axis_curve.get("sampled_points", []) or []]
        control = [tuple(map(float, p)) for p in axis_curve.get("control_points", []) or []]
    else:
        points = [tuple(map(float, p)) for p in (axis_curve or [])]
        control = _body_axis_control_points(keypoints)
    if len(points) < 2:
        points = _sample_axis(keypoints, cfg).get("sampled_points", [])
    p1 = _point(keypoints, "P1_snout_tip")
    p5 = _point(keypoints, "P5_caudal_base_midpoint")
    if len(points) < 2 or p1 is None or p5 is None:
        return {
            "SL_straight_mm": None,
            "SL_curve_mm": None,
            "curvature_index": None,
            "max_axis_deviation_mm": None,
            "axis_bend_angle_deg": None,
            "curvature_qc_level": "unknown",
            "curvature_qc_pass": False,
            "curvature_review_points": ["P4", "P8", "P9", "P10", "P11", "P7V", "TL_curve"],
            "measurements_needs_review": True,
            "curvature_high_review_required": False,
            "review_reason": "axis_curve_failed",
        }
    sl_straight_px = hypot(p5[0] - p1[0], p5[1] - p1[1])
    sl_curve_px = polyline_length(points)
    curvature_index = sl_curve_px / sl_straight_px if sl_straight_px > 0 else None
    deviation_px = _max_deviation_from_line(points, p1, p5)
    bend_angle = _max_bend_angle(control or points)

    if curvature_index is None:
        level = "unknown"
    elif curvature_index <= 1.03:
        level = "straight_or_mild"
    elif curvature_index <= 1.08:
        level = "moderate"
    else:
        level = "high"
    review_points: list[str] = []
    measurements_needs_review = False
    qc_pass = True
    review_reason = ""
    if level == "moderate":
        review_points = ["P4", "P8", "P9", "P10", "P11", "P7V", "TL_curve"]
        measurements_needs_review = True
        review_reason = "curvature-aware QC: fish body is curved; please review local-normal measurements"
    elif level == "high":
        review_points = ["P4", "P8", "P9", "P10", "P11", "P7V", "TL_curve"]
        measurements_needs_review = True
        qc_pass = False
        review_reason = "High curvature detected. Please manually review P4, P8/P9, P10/P11, P7V, and TL_curve"

    return {
        "SL_straight_mm": sl_straight_px * mm_per_pixel,
        "SL_curve_mm": sl_curve_px * mm_per_pixel,
        "curvature_index": curvature_index,
        "max_axis_deviation_mm": deviation_px * mm_per_pixel,
        "axis_bend_angle_deg": bend_angle,
        "curvature_qc_level": level,
        "curvature_qc_pass": qc_pass,
        "curvature_review_points": review_points,
        "measurements_needs_review": measurements_needs_review,
        "curvature_high_review_required": level == "high",
        "review_reason": review_reason,
    }


def _max_deviation_from_line(
    points: Sequence[tuple[float, float]],
    line_start: tuple[float, float],
    line_end: tuple[float, float],
) -> float:
    dx = line_end[0] - line_start[0]
    dy = line_end[1] - line_start[1]
    denom = hypot(dx, dy)
    if denom <= 1e-9:
        return 0.0
    return max(abs(dy * p[0] - dx * p[1] + line_end[0] * line_start[1] - line_end[1] * line_start[0]) / denom for p in points)


def _max_bend_angle(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    max_angle = 0.0
    for a, b, c in zip(points, points[1:], points[2:]):
        v1 = (b[0] - a[0], b[1] - a[1])
        v2 = (c[0] - b[0], c[1] - b[1])
        l1 = hypot(v1[0], v1[1])
        l2 = hypot(v2[0], v2[1])
        if l1 <= 1e-9 or l2 <= 1e-9:
            continue
        cos_value = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)))
        max_angle = max(max_angle, degrees(acos(cos_value)))
    return float(max_angle)
