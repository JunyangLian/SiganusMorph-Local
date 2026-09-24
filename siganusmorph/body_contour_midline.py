"""Body-contour midline suggestions for subjective axis keypoints.

The C1-C4 axis points are auxiliary geometry, not sharply visible anatomy.
This module derives C1-C3 from a robust fish-body contour midline and derives
C4 from the P4-P5 peduncle axis midpoint.  It is read-only: callers can show
or evaluate the suggestions without writing labels.
"""

from __future__ import annotations

from math import hypot
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw

from .axis_utils import point_xy


P1 = "P1_snout_tip"
P4 = "P4_peduncle_start_midpoint"
P5 = "P5_caudal_base_midpoint"
C1 = "C1_head_axis_point"
C2 = "C2_trunk_axis_point"
C3 = "C3_posterior_trunk_axis_point"
C4 = "C4_peduncle_axis_point"

DEFAULT_CONFIG = {
    "sample_count": 160,
    "body_start_margin": 0.04,
    "body_end_margin": 0.04,
    "dorsal_percentile": 7.0,
    "ventral_percentile": 93.0,
    "median_kernel": 7,
    "smooth_kernel": 9,
    "spike_threshold_px": 28.0,
    "min_pixels_per_bin": 8,
    "C1_BODY_RATIO": 0.25,
    "C2_BODY_RATIO": 0.50,
    "C3_BODY_RATIO": 0.75,
}


def _as_mask(mask: Any) -> np.ndarray | None:
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(bool)


def _point(keypoints: Mapping[str, Any], key: str) -> tuple[float, float] | None:
    value = keypoints.get(key)
    if value is None:
        return None
    try:
        return point_xy(value)
    except Exception:
        return None


def _merge_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_CONFIG)
    if config:
        merged.update(dict(config))
    return merged


def _odd_kernel(value: int, length: int) -> int:
    value = max(1, int(value))
    if value % 2 == 0:
        value += 1
    return min(value, max(1, length if length % 2 == 1 else length - 1))


def _nan_median_filter(values: np.ndarray, kernel: int) -> np.ndarray:
    if len(values) == 0:
        return values
    kernel = _odd_kernel(kernel, len(values))
    half = kernel // 2
    out = values.copy()
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        window = values[lo:hi]
        valid = window[np.isfinite(window)]
        if len(valid):
            out[i] = float(np.median(valid))
    return out


def _nan_moving_average(values: np.ndarray, kernel: int) -> np.ndarray:
    if len(values) == 0:
        return values
    kernel = _odd_kernel(kernel, len(values))
    half = kernel // 2
    out = values.copy()
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        window = values[lo:hi]
        valid = window[np.isfinite(window)]
        if len(valid):
            out[i] = float(np.mean(valid))
    return out


def _fill_nan(values: np.ndarray) -> np.ndarray:
    if np.isfinite(values).all():
        return values
    indices = np.arange(len(values))
    valid = np.isfinite(values)
    if valid.sum() == 0:
        return values
    if valid.sum() == 1:
        return np.full_like(values, float(values[valid][0]), dtype=float)
    return np.interp(indices, indices[valid], values[valid])


def _transform_to_axis(
    xs: np.ndarray,
    ys: np.ndarray,
    origin: tuple[float, float],
    tangent: tuple[float, float],
    normal: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    dx = xs.astype(float) - origin[0]
    dy = ys.astype(float) - origin[1]
    u = dx * tangent[0] + dy * tangent[1]
    v = dx * normal[0] + dy * normal[1]
    return u, v


def _axis_to_xy(
    u: np.ndarray | float,
    v: np.ndarray | float,
    origin: tuple[float, float],
    tangent: tuple[float, float],
    normal: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    uu = np.asarray(u, dtype=float)
    vv = np.asarray(v, dtype=float)
    x = origin[0] + uu * tangent[0] + vv * normal[0]
    y = origin[1] + uu * tangent[1] + vv * normal[1]
    return x, y


def _list_points(xs: np.ndarray, ys: np.ndarray) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y)]


def _sample_at_ratio(
    u_centers: np.ndarray,
    mid_v: np.ndarray,
    ratio: float,
    length_px: float,
    origin: tuple[float, float],
    tangent: tuple[float, float],
    normal: tuple[float, float],
) -> list[float] | None:
    if len(u_centers) == 0 or not np.isfinite(mid_v).any():
        return None
    target = float(length_px) * float(ratio)
    mid_filled = _fill_nan(mid_v)
    v = float(np.interp(target, u_centers, mid_filled))
    x, y = _axis_to_xy(target, v, origin, tangent, normal)
    return [float(x), float(y)]


def _mask_polygon_from_contours(
    image_shape: tuple[int, int] | tuple[int, int, int],
    dorsal: list[list[float]],
    ventral: list[list[float]],
) -> np.ndarray:
    height, width = int(image_shape[0]), int(image_shape[1])
    image = Image.new("L", (width, height), 0)
    if len(dorsal) >= 2 and len(ventral) >= 2:
        polygon = [(float(x), float(y)) for x, y in dorsal] + [(float(x), float(y)) for x, y in reversed(ventral)]
        ImageDraw.Draw(image).polygon(polygon, fill=255)
    return np.asarray(image) > 0


def estimate_body_contour_midline_points(
    fish_mask: Any,
    keypoints: Mapping[str, Any],
    image_shape: tuple[int, int] | tuple[int, int, int],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate C1-C4 from fish-body contour geometry.

    C1-C3 are sampled from the smoothed body midline between P1 and P4.
    C4 is the midpoint between P4 and P5.
    """

    cfg = _merge_config(config)
    mask = _as_mask(fish_mask)
    p1 = _point(keypoints, P1)
    p4 = _point(keypoints, P4)
    p5 = _point(keypoints, P5)
    if mask is None or p1 is None or p4 is None:
        reason = "missing_mask_or_P1_P4"
        return {
            "body_core_mask": None,
            "dorsal_body_contour": [],
            "ventral_body_contour": [],
            "body_midline_points": [],
            "C1_geometric": None,
            "C2_geometric": None,
            "C3_geometric": None,
            "C4_geometric": None,
            "body_midline_qc": {
                "body_midline_qc_pass": False,
                "body_midline_qc_reason": reason,
                "C1_source": "failed",
                "C2_source": "failed",
                "C3_source": "failed",
                "C4_source": "failed",
            },
        }

    dx = p4[0] - p1[0]
    dy = p4[1] - p1[1]
    length = hypot(dx, dy)
    if length <= 1e-6:
        reason = "P1_P4_overlap"
        return {
            "body_core_mask": None,
            "dorsal_body_contour": [],
            "ventral_body_contour": [],
            "body_midline_points": [],
            "C1_geometric": None,
            "C2_geometric": None,
            "C3_geometric": None,
            "C4_geometric": None,
            "body_midline_qc": {"body_midline_qc_pass": False, "body_midline_qc_reason": reason},
        }
    tangent = (dx / length, dy / length)
    normal = (-tangent[1], tangent[0])

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {
            "body_core_mask": None,
            "dorsal_body_contour": [],
            "ventral_body_contour": [],
            "body_midline_points": [],
            "C1_geometric": None,
            "C2_geometric": None,
            "C3_geometric": None,
            "C4_geometric": None,
            "body_midline_qc": {"body_midline_qc_pass": False, "body_midline_qc_reason": "empty_fish_mask"},
        }
    u, v = _transform_to_axis(xs, ys, p1, tangent, normal)
    lo_u = length * float(cfg["body_start_margin"])
    hi_u = length * (1.0 - float(cfg["body_end_margin"]))
    in_body = (u >= lo_u) & (u <= hi_u)
    if in_body.sum() < int(cfg["sample_count"]):
        in_body = (u >= 0) & (u <= length)

    u_body = u[in_body]
    v_body = v[in_body]
    sample_count = int(cfg["sample_count"])
    u_centers = np.linspace(0, length, sample_count)
    edges = np.linspace(0, length, sample_count + 1)
    dorsal_raw = np.full(sample_count, np.nan, dtype=float)
    ventral_raw = np.full(sample_count, np.nan, dtype=float)
    counts = np.zeros(sample_count, dtype=int)
    min_pixels = int(cfg["min_pixels_per_bin"])
    for i in range(sample_count):
        values = v_body[(u_body >= edges[i]) & (u_body < edges[i + 1])]
        counts[i] = int(len(values))
        if len(values) < min_pixels:
            continue
        dorsal_raw[i] = float(np.percentile(values, float(cfg["dorsal_percentile"])))
        ventral_raw[i] = float(np.percentile(values, float(cfg["ventral_percentile"])))

    dorsal = _nan_moving_average(_nan_median_filter(_fill_nan(dorsal_raw), int(cfg["median_kernel"])), int(cfg["smooth_kernel"]))
    ventral = _nan_moving_average(_nan_median_filter(_fill_nan(ventral_raw), int(cfg["median_kernel"])), int(cfg["smooth_kernel"]))
    mid_v = (dorsal + ventral) / 2.0

    dorsal_x, dorsal_y = _axis_to_xy(u_centers, dorsal, p1, tangent, normal)
    ventral_x, ventral_y = _axis_to_xy(u_centers, ventral, p1, tangent, normal)
    # Label dorsal/ventral visually by image y; the coordinate normal sign can flip.
    swap = dorsal_y > ventral_y
    if np.any(swap):
        d_x = dorsal_x.copy()
        d_y = dorsal_y.copy()
        dorsal_x[swap] = ventral_x[swap]
        dorsal_y[swap] = ventral_y[swap]
        ventral_x[swap] = d_x[swap]
        ventral_y[swap] = d_y[swap]
    mid_x, mid_y = _axis_to_xy(u_centers, mid_v, p1, tangent, normal)

    c1 = _sample_at_ratio(u_centers, mid_v, float(cfg["C1_BODY_RATIO"]), length, p1, tangent, normal)
    c2 = _sample_at_ratio(u_centers, mid_v, float(cfg["C2_BODY_RATIO"]), length, p1, tangent, normal)
    c3 = _sample_at_ratio(u_centers, mid_v, float(cfg["C3_BODY_RATIO"]), length, p1, tangent, normal)
    c4 = [float((p4[0] + p5[0]) / 2.0), float((p4[1] + p5[1]) / 2.0)] if p5 is not None else None

    finite_raw = np.isfinite(dorsal_raw) & np.isfinite(ventral_raw)
    spike_d = np.abs(_fill_nan(dorsal_raw) - dorsal) > float(cfg["spike_threshold_px"])
    spike_v = np.abs(_fill_nan(ventral_raw) - ventral) > float(cfg["spike_threshold_px"])
    spike_count = int(np.sum((spike_d | spike_v) & finite_raw))
    widths = ventral - dorsal
    contour_cross = bool(np.any(widths <= 0))
    midline_jump = float(np.nanmax(np.abs(np.diff(mid_v)))) if len(mid_v) > 1 else 0.0
    smoothness = float(np.nanmedian(np.abs(np.diff(mid_v, n=2)))) if len(mid_v) > 2 else 0.0
    missing_fraction = float(1.0 - finite_raw.mean()) if len(finite_raw) else 1.0

    def point_inside(point: list[float] | None) -> bool:
        if point is None:
            return False
        x = int(round(point[0]))
        y = int(round(point[1]))
        return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and bool(mask[y, x])

    review = []
    if contour_cross:
        review.append("body_contours_cross")
    if missing_fraction > 0.25:
        review.append("many_empty_body_bins")
    if midline_jump > 60:
        review.append("body_midline_jump")
    if not all(point_inside(p) for p in (c1, c2, c3)):
        review.append("C1_C3_not_all_inside_mask")
    if c4 is None:
        review.append("missing_P5_for_C4")

    qc_pass = len(review) == 0
    dorsal_list = _list_points(dorsal_x, dorsal_y)
    ventral_list = _list_points(ventral_x, ventral_y)
    midline_list = _list_points(mid_x, mid_y)
    body_core_mask = _mask_polygon_from_contours(image_shape, dorsal_list, ventral_list)

    return {
        "body_core_mask": body_core_mask,
        "dorsal_body_contour": dorsal_list,
        "ventral_body_contour": ventral_list,
        "body_midline_points": midline_list,
        "C1_geometric": c1,
        "C2_geometric": c2,
        "C3_geometric": c3,
        "C4_geometric": c4,
        "body_midline_qc": {
            "body_midline_qc_pass": qc_pass,
            "body_midline_qc_reason": ";".join(review),
            "fin_spike_removed_count": spike_count,
            "dorsal_contour_quality": "warning" if spike_count > sample_count * 0.20 else "good",
            "ventral_contour_quality": "warning" if missing_fraction > 0.15 else "good",
            "body_midline_smoothness": smoothness,
            "body_midline_max_jump_px": midline_jump,
            "missing_bin_fraction": missing_fraction,
            "C1_source": "body_contour_midline_rule" if c1 is not None else "failed",
            "C2_source": "body_contour_midline_rule" if c2 is not None else "failed",
            "C3_source": "body_contour_midline_rule" if c3 is not None else "failed",
            "C4_source": "P4_P5_midpoint_rule" if c4 is not None else "failed",
            "body_axis_length_px": length,
            "tangent": [float(tangent[0]), float(tangent[1])],
            "normal": [float(normal[0]), float(normal[1])],
        },
    }
