"""Mask-based geometric suggestions and QC for enhanced preannotation."""

from __future__ import annotations

import math
from typing import Any, Mapping

import cv2
import numpy as np


FullKeypoints = Mapping[str, tuple[float, float] | list[float]]


def _as_xy(point: tuple[float, float] | list[float]) -> tuple[float, float]:
    return float(point[0]), float(point[1])


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _mask_bounds_by_column(
    mask: np.ndarray,
    x_start: int,
    x_end: int,
    min_column_pixels: int = 12,
) -> list[tuple[int, int, int, int]]:
    """Return x, top y, bottom y, and span for useful mask columns."""
    h, w = mask.shape[:2]
    x_start = max(0, min(w - 1, int(x_start)))
    x_end = max(x_start + 1, min(w, int(x_end)))
    rows: list[tuple[int, int, int, int]] = []
    for x in range(x_start, x_end):
        ys = np.where(mask[:, x] > 0)[0]
        if len(ys) < min_column_pixels:
            continue
        top = int(ys.min())
        bottom = int(ys.max())
        rows.append((x, top, bottom, bottom - top))
    return rows


def _best_span_column(rows: list[tuple[int, int, int, int]], mode: str) -> tuple[float, float, float, float] | None:
    if not rows:
        return None
    if mode == "max":
        x, top, bottom, span = max(rows, key=lambda item: item[3])
    else:
        usable = [row for row in rows if row[3] > 20]
        if not usable:
            usable = rows
        x, top, bottom, span = min(usable, key=lambda item: item[3])
    return float(x), float(top), float(bottom), float(span)


def _point_from_dict(points: FullKeypoints, key: str) -> tuple[float, float] | None:
    value = points.get(key)
    if value is None:
        return None
    return _as_xy(value)


def _right_tail_region(bbox: tuple[int, int, int, int], fraction: float = 0.35) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    width = max(1, x1 - x0)
    return int(round(x1 - width * fraction)), y0, x1, y1


def _tail_tip_candidates(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[str, tuple[float, float]]:
    tx0, ty0, tx1, ty1 = _right_tail_region(bbox, fraction=0.35)
    tail = np.zeros_like(mask)
    tail[ty0:ty1, tx0:tx1] = mask[ty0:ty1, tx0:tx1]
    ys, xs = np.where(tail > 0)
    if len(xs) == 0:
        return {}

    center_y = (ty0 + ty1) / 2.0
    upper_idx = np.where(ys <= center_y)[0]
    lower_idx = np.where(ys > center_y)[0]
    suggestions: dict[str, tuple[float, float]] = {}
    if len(upper_idx) > 0:
        ux = xs[upper_idx].astype(float)
        uy = ys[upper_idx].astype(float)
        score = ux + 0.15 * (center_y - uy)
        best = int(np.argmax(score))
        suggestions["P7U_caudal_fin_upper_tip"] = (float(ux[best]), float(uy[best]))
    if len(lower_idx) > 0:
        lx = xs[lower_idx].astype(float)
        ly = ys[lower_idx].astype(float)
        score = lx + 0.15 * (ly - center_y)
        best = int(np.argmax(score))
        suggestions["P7L_caudal_fin_lower_tip"] = (float(lx[best]), float(ly[best]))
    return suggestions


def generate_mask_suggestions(
    fish_mask: np.ndarray,
    fish_bbox: tuple[int, int, int, int],
    model_keypoints: FullKeypoints,
) -> dict[str, tuple[float, float]]:
    """Generate mask/geometric suggestion points without forcing replacements."""
    suggestions: dict[str, tuple[float, float]] = {}
    if fish_mask.size == 0 or np.count_nonzero(fish_mask) == 0:
        return suggestions

    clean_mask = cv2.medianBlur((fish_mask > 0).astype(np.uint8) * 255, 7)
    x0, y0, x1, y1 = fish_bbox
    width = max(1, x1 - x0)

    body_rows = _mask_bounds_by_column(
        clean_mask,
        int(round(x0 + 0.18 * width)),
        int(round(x0 + 0.68 * width)),
    )
    body_best = _best_span_column(body_rows, mode="max")
    if body_best is not None:
        x, top, bottom, _span = body_best
        suggestions["P8_body_depth_dorsal"] = (x, top)
        suggestions["P9_body_depth_ventral"] = (x, bottom)

    p4 = _point_from_dict(model_keypoints, "P4_peduncle_start_midpoint")
    p5 = _point_from_dict(model_keypoints, "P5_caudal_base_midpoint")
    if p4 is not None and p5 is not None and abs(p5[0] - p4[0]) > 35:
        ped_x0 = max(x0 + 0.55 * width, min(p4[0], p5[0]) - 0.04 * width)
        ped_x1 = min(x1 - 0.03 * width, max(p4[0], p5[0]) + 0.04 * width)
    else:
        ped_x0 = x0 + 0.68 * width
        ped_x1 = x0 + 0.90 * width
    if ped_x1 <= ped_x0:
        ped_x0 = x0 + 0.68 * width
        ped_x1 = x0 + 0.90 * width
    ped_rows = _mask_bounds_by_column(clean_mask, int(round(ped_x0)), int(round(ped_x1)), min_column_pixels=8)
    ped_best = _best_span_column(ped_rows, mode="min")
    if ped_best is not None:
        x, top, bottom, _span = ped_best
        suggestions["P10_peduncle_depth_dorsal"] = (x, top)
        suggestions["P11_peduncle_depth_ventral"] = (x, bottom)

    suggestions.update(_tail_tip_candidates(clean_mask, fish_bbox))
    return suggestions


def _p7v_from_keypoints(points: FullKeypoints) -> tuple[float, float] | None:
    p5 = _point_from_dict(points, "P5_caudal_base_midpoint")
    p6 = _point_from_dict(points, "P6_caudal_fork_midpoint")
    p7u = _point_from_dict(points, "P7U_caudal_fin_upper_tip")
    p7l = _point_from_dict(points, "P7L_caudal_fin_lower_tip")
    c4 = _point_from_dict(points, "C4_peduncle_axis_point")
    if p5 is None or p7u is None or p7l is None:
        return None
    if p6 is not None and _distance(p5, p6) > 1:
        axis = (p6[0] - p5[0], p6[1] - p5[1])
    elif c4 is not None and _distance(c4, p5) > 1:
        axis = (p5[0] - c4[0], p5[1] - c4[1])
    else:
        return None
    length = math.hypot(axis[0], axis[1])
    if length <= 0:
        return None
    ux, uy = axis[0] / length, axis[1] / length
    proj_u = (p7u[0] - p5[0]) * ux + (p7u[1] - p5[1]) * uy
    proj_l = (p7l[0] - p5[0]) * ux + (p7l[1] - p5[1]) * uy
    extension = max(proj_u, proj_l)
    return (p5[0] + extension * ux, p5[1] + extension * uy)


def run_preannotation_qc(
    corrected_keypoints: FullKeypoints,
    mask_suggestions: Mapping[str, tuple[float, float]],
    fish_bbox: tuple[int, int, int, int],
    segmentation_success: bool,
) -> dict[str, Any]:
    """Run first-pass QC for enhanced preannotation."""
    x0, y0, x1, y1 = fish_bbox
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    review_reasons: list[str] = []
    flagged: dict[str, str] = {}
    suggestion_distances: dict[str, float] = {}

    if not segmentation_success:
        review_reasons.append("segmentation_failed")

    if width < 300 or height < 80 or width / max(1, height) < 1.5:
        review_reasons.append("fish_bbox_unusual")

    if "P1_snout_tip" not in corrected_keypoints:
        review_reasons.append("missing_p1")

    thresholds = {
        "P8_body_depth_dorsal": max(35.0, 0.035 * width),
        "P9_body_depth_ventral": max(35.0, 0.035 * width),
        "P10_peduncle_depth_dorsal": max(30.0, 0.030 * width),
        "P11_peduncle_depth_ventral": max(30.0, 0.030 * width),
        "P7U_caudal_fin_upper_tip": max(55.0, 0.060 * width),
        "P7L_caudal_fin_lower_tip": max(55.0, 0.060 * width),
    }
    for key, suggestion in mask_suggestions.items():
        if key not in thresholds or key not in corrected_keypoints:
            continue
        distance = _distance(_as_xy(corrected_keypoints[key]), suggestion)
        suggestion_distances[key] = float(distance)
        if distance > thresholds[key]:
            flagged[key] = "far_from_mask_suggestion"
            review_reasons.append(f"{key}_far_from_mask_suggestion")

    tx0, ty0, tx1, ty1 = _right_tail_region(fish_bbox, fraction=0.38)
    for key in ("P7U_caudal_fin_upper_tip", "P7L_caudal_fin_lower_tip"):
        point = _point_from_dict(corrected_keypoints, key)
        if point is None:
            flagged[key] = "missing"
            review_reasons.append(f"{key}_missing")
            continue
        if not (tx0 <= point[0] <= tx1 + 0.10 * width and ty0 - 0.20 * height <= point[1] <= ty1 + 0.20 * height):
            flagged[key] = "outside_tail_region"
            review_reasons.append(f"{key}_outside_tail_region")

    p7v = _p7v_from_keypoints(corrected_keypoints)
    p7v_valid = True
    if p7v is None:
        p7v_valid = False
    else:
        if p7v[0] < x0 + 0.55 * width or p7v[0] > x1 + 0.20 * width:
            p7v_valid = False
        if p7v[1] < y0 - 0.30 * height or p7v[1] > y1 + 0.30 * height:
            p7v_valid = False
    if not p7v_valid:
        review_reasons.append("invalid_p7v_or_tail_keypoints")

    body_axis_keys = [
        "P1_snout_tip",
        "C1_head_axis_point",
        "C2_trunk_axis_point",
        "C3_posterior_trunk_axis_point",
        "P4_peduncle_start_midpoint",
        "C4_peduncle_axis_point",
        "P5_caudal_base_midpoint",
    ]
    axis_points = [_point_from_dict(corrected_keypoints, key) for key in body_axis_keys]
    axis_order_valid = all(point is not None for point in axis_points)
    if axis_order_valid:
        xs = [point[0] for point in axis_points if point is not None]
        axis_order_valid = all(b >= a - 0.04 * width for a, b in zip(xs, xs[1:]))
    if not axis_order_valid:
        review_reasons.append("body_axis_order_unusual")

    p1 = _point_from_dict(corrected_keypoints, "P1_snout_tip")
    p5 = _point_from_dict(corrected_keypoints, "P5_caudal_base_midpoint")
    curvature_index = ""
    if p1 is not None and p5 is not None and axis_order_valid:
        straight = _distance(p1, p5)
        axis_len = 0.0
        for a, b in zip(axis_points, axis_points[1:]):
            axis_len += _distance(a, b)
        if straight > 1:
            curvature_index = axis_len / straight
            if curvature_index > 1.12:
                review_reasons.append("curvature_index_unusual")

    unique_reasons = list(dict.fromkeys(review_reasons))
    return {
        "needs_review": bool(unique_reasons),
        "review_reason": ";".join(unique_reasons),
        "flagged_keypoints": flagged,
        "suggestion_distances_px": suggestion_distances,
        "p7v_valid": bool(p7v_valid),
        "p7v_suggestion": p7v,
        "body_axis_order_valid": bool(axis_order_valid),
        "curvature_index_polyline_px": curvature_index,
    }
