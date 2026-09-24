"""Tail-focused mask geometry and QC for enhanced preannotation."""

from __future__ import annotations

import math
from typing import Any, Mapping

import cv2
import numpy as np


FullKeypoints = Mapping[str, tuple[float, float] | list[float]]

P4 = "P4_peduncle_start_midpoint"
P5 = "P5_caudal_base_midpoint"
P6 = "P6_caudal_fork_midpoint"
P7U = "P7U_caudal_fin_upper_tip"
P7L = "P7L_caudal_fin_lower_tip"
P8 = "P8_body_depth_dorsal"
P9 = "P9_body_depth_ventral"
P10 = "P10_peduncle_depth_dorsal"
P11 = "P11_peduncle_depth_ventral"
C1 = "C1_head_axis_point"
C2 = "C2_trunk_axis_point"
C3 = "C3_posterior_trunk_axis_point"
C4 = "C4_peduncle_axis_point"
P1 = "P1_snout_tip"

TAIL_REVIEW_REASONS = {
    "tail_mask_rule_failed",
    "P5_caudal_base_unreliable",
    "P6_caudal_fork_unreliable",
    "invalid_p7v_after_tail_qc",
    "peduncle_depth_points_unreliable",
    "body_depth_points_need_review",
    "body_axis_qc_failed",
}


def _as_xy(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    return float(value[0]), float(value[1])


def _point(points: Mapping[str, Any], key: str) -> tuple[float, float] | None:
    return _as_xy(points.get(key))


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _unit(start: tuple[float, float] | None, end: tuple[float, float] | None) -> tuple[float, float] | None:
    if start is None or end is None:
        return None
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return None
    return dx / length, dy / length


def _dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _sub(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return a[0] - b[0], a[1] - b[1]


def _finite_point(point: tuple[float, float] | None) -> bool:
    return point is not None and math.isfinite(point[0]) and math.isfinite(point[1])


def _bbox_tuple(fish_bbox: tuple[int, int, int, int] | list[int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [int(round(float(value))) for value in fish_bbox]
    return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)


def _mask_clean(fish_mask: np.ndarray | None) -> np.ndarray | None:
    if fish_mask is None or fish_mask.size == 0 or int(np.count_nonzero(fish_mask)) == 0:
        return None
    mask = (fish_mask > 0).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def _largest_contour_points(mask: np.ndarray) -> np.ndarray:
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2), dtype=np.float32)
    contour = max(contours, key=cv2.contourArea)
    return contour.reshape(-1, 2).astype(np.float32)


def _mask_bounds_by_column(
    mask: np.ndarray,
    x_start: int,
    x_end: int,
    *,
    min_column_pixels: int = 8,
) -> list[tuple[int, int, int, int]]:
    height, width = mask.shape[:2]
    x_start = max(0, min(width - 1, int(round(x_start))))
    x_end = max(x_start + 1, min(width, int(round(x_end))))
    rows: list[tuple[int, int, int, int]] = []
    for x in range(x_start, x_end):
        ys = np.where(mask[:, x] > 0)[0]
        if len(ys) < min_column_pixels:
            continue
        top = int(ys.min())
        bottom = int(ys.max())
        rows.append((x, top, bottom, bottom - top))
    return rows


def _span_suggestion(rows: list[tuple[int, int, int, int]], mode: str) -> tuple[float, float, float, float] | None:
    if not rows:
        return None
    usable = [row for row in rows if row[3] > 10] or rows
    x, top, bottom, span = (max if mode == "max" else min)(usable, key=lambda row: row[3])
    return float(x), float(top), float(bottom), float(span)


def _point_in_mask(mask: np.ndarray | None, point: tuple[float, float] | None, radius: int = 4) -> bool:
    if mask is None or point is None:
        return False
    height, width = mask.shape[:2]
    x = int(round(point[0]))
    y = int(round(point[1]))
    if x < 0 or y < 0 or x >= width or y >= height:
        return False
    x0 = max(0, x - radius)
    x1 = min(width, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(height, y + radius + 1)
    return bool(np.count_nonzero(mask[y0:y1, x0:x1]) > 0)


def _serial_point(point: tuple[float, float] | None) -> list[float] | None:
    if point is None:
        return None
    return [float(point[0]), float(point[1])]


def _unique_reasons(reasons: list[str]) -> list[str]:
    return [reason for reason in dict.fromkeys(reason for reason in reasons if reason)]


def _p6_basic_qc(
    points: Mapping[str, Any],
    fish_bbox: tuple[int, int, int, int],
    confidences: Mapping[str, float] | None,
    p6_mask_quality: str | None = None,
) -> dict[str, Any]:
    x0, y0, x1, y1 = fish_bbox
    width = max(1, x1 - x0)
    p5 = _point(points, P5)
    p6 = _point(points, P6)
    p7u = _point(points, P7U)
    p7l = _point(points, P7L)
    reasons: list[str] = []
    conf = float((confidences or {}).get(P6, 1.0) or 0.0)
    source = "model_unreliable"
    if p6_mask_quality in {"good", "warning"}:
        source = "mask_fork_rule"
    if p5 is None or p6 is None:
        reasons.append("missing_P5_or_P6")
    elif p6[0] <= p5[0] + max(8.0, 0.015 * width):
        reasons.append("P6_not_right_of_P5")
    elif _distance(p5, p6) < max(20.0, 0.03 * width):
        reasons.append("P5_P6_too_close")
    if conf < 0.20:
        reasons.append("P6_low_confidence")
    if p6_mask_quality in {"good", "warning"} and "P6_low_confidence" in reasons:
        reasons.remove("P6_low_confidence")
    if p6_mask_quality == "failed":
        reasons.append("P6_mask_fork_rule_failed")
    if p6 is not None and p7u is not None and p7l is not None:
        top_y = min(p7u[1], p7l[1])
        bottom_y = max(p7u[1], p7l[1])
        margin = max(25.0, 0.06 * (y1 - y0))
        if not (top_y - margin <= p6[1] <= bottom_y + margin):
            reasons.append("P6_not_between_tail_lobes")
    return {
        "P6_qc_pass": not reasons,
        "P6_review_reason": ";".join(_unique_reasons(reasons)),
        "P6_confidence": conf,
        "P6_source": source,
        "P6_mask_quality": p6_mask_quality or "",
    }


def _choose_tail_axis(
    points: Mapping[str, Any],
    fish_bbox: tuple[int, int, int, int],
    confidences: Mapping[str, float] | None,
    p6_mask_quality: str | None = None,
) -> tuple[tuple[float, float] | None, str, tuple[float, float] | None, tuple[float, float] | None, dict[str, Any]]:
    p5 = _point(points, P5)
    p6 = _point(points, P6)
    c4 = _point(points, C4)
    c3 = _point(points, C3)
    p6_qc = _p6_basic_qc(points, fish_bbox, confidences, p6_mask_quality=p6_mask_quality)
    if p6_mask_quality in {"good", "warning"} and p6_qc["P6_qc_pass"]:
        axis = _unit(p5, p6)
        if axis is not None:
            return axis, "P5_to_P6_mask", p5, p6, p6_qc
    axis = _unit(c4, p5)
    if axis is not None:
        return axis, "C4_to_P5_fallback", c4, p5, p6_qc
    axis = _unit(c3, p5)
    if axis is not None:
        return axis, "C3_to_P5_fallback", c3, p5, p6_qc
    return None, "unavailable", None, None, p6_qc


def _tail_region_bbox(fish_bbox: tuple[int, int, int, int], p5: tuple[float, float] | None) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = fish_bbox
    width = max(1, x1 - x0)
    right_start = x1 - int(round(0.35 * width))
    if p5 is not None:
        right_start = max(right_start, int(round(p5[0] - 0.03 * width)))
    right_start = min(max(x0, right_start), x1 - 1)
    return right_start, y0, x1, y1


def _tail_axis_from_lobes(
    points: Mapping[str, Any],
) -> tuple[tuple[float, float] | None, str, tuple[float, float] | None, tuple[float, float] | None]:
    p5 = _point(points, P5)
    p7u = _point(points, P7U)
    p7l = _point(points, P7L)
    c4 = _point(points, C4)
    c3 = _point(points, C3)
    if p5 is not None and p7u is not None and p7l is not None and p7u[1] < p7l[1]:
        mid = ((p7u[0] + p7l[0]) / 2.0, (p7u[1] + p7l[1]) / 2.0)
        axis = _unit(p5, mid)
        if axis is not None:
            return axis, "P5_to_midpoint_P7U_P7L", p5, mid
    axis = _unit(c4, p5)
    if axis is not None:
        return axis, "C4_to_P5_fallback", c4, p5
    axis = _unit(c3, p5)
    if axis is not None:
        return axis, "C3_to_P5_fallback", c3, p5
    return None, "unavailable", None, None


def select_tail_axis_without_p6(
    keypoints: Mapping[str, Any],
) -> dict[str, Any]:
    """Select a tail axis without using P6 unless caudal lobe tips are unavailable.

    P6 is a fork-notch landmark and can be unstable, so P7V/TL geometry should
    prefer the caudal lobe midpoint or lobe-angle bisector.
    """
    p5 = _point(keypoints, P5)
    p7u = _point(keypoints, P7U)
    p7l = _point(keypoints, P7L)
    c4 = _point(keypoints, C4)
    c3 = _point(keypoints, C3)
    p6 = _point(keypoints, P6)
    if p5 is not None and p7u is not None and p7l is not None:
        mid = ((p7u[0] + p7l[0]) / 2.0, (p7u[1] + p7l[1]) / 2.0)
        axis = _unit(p5, mid)
        if axis is not None:
            return {
                "tail_axis": axis,
                "tail_axis_source": "tail_axis_from_tail_tips",
                "tail_axis_start": _serial_point(p5),
                "tail_axis_end": _serial_point(mid),
                "tail_axis_uses_P6": False,
            }
        axis_u = _unit(p5, p7u)
        axis_l = _unit(p5, p7l)
        if axis_u is not None and axis_l is not None:
            bx = axis_u[0] + axis_l[0]
            by = axis_u[1] + axis_l[1]
            norm = math.hypot(bx, by)
            if norm > 1e-9:
                axis = (bx / norm, by / norm)
                end = (p5[0] + 100.0 * axis[0], p5[1] + 100.0 * axis[1])
                return {
                    "tail_axis": axis,
                    "tail_axis_source": "tail_axis_from_lobe_bisector",
                    "tail_axis_start": _serial_point(p5),
                    "tail_axis_end": _serial_point(end),
                    "tail_axis_uses_P6": False,
                }
    axis = _unit(c4, p5)
    if axis is not None:
        return {
            "tail_axis": axis,
            "tail_axis_source": "C4_to_P5_fallback",
            "tail_axis_start": _serial_point(c4),
            "tail_axis_end": _serial_point(p5),
            "tail_axis_uses_P6": False,
        }
    axis = _unit(c3, p5)
    if axis is not None:
        return {
            "tail_axis": axis,
            "tail_axis_source": "C3_to_P5_fallback",
            "tail_axis_start": _serial_point(c3),
            "tail_axis_end": _serial_point(p5),
            "tail_axis_uses_P6": False,
        }
    axis = _unit(p5, p6)
    if axis is not None:
        return {
            "tail_axis": axis,
            "tail_axis_source": "P5_to_P6_last_resort",
            "tail_axis_start": _serial_point(p5),
            "tail_axis_end": _serial_point(p6),
            "tail_axis_uses_P6": True,
        }
    return {
        "tail_axis": None,
        "tail_axis_source": "unavailable",
        "tail_axis_start": None,
        "tail_axis_end": None,
        "tail_axis_uses_P6": False,
    }


def _tail_axis_fallback_before_lobes(
    points: Mapping[str, Any],
) -> tuple[tuple[float, float] | None, str, tuple[float, float] | None, tuple[float, float] | None]:
    """Axis for finding tail lobes before P7U/P7L have been geometry-corrected."""
    p5 = _point(points, P5)
    c4 = _point(points, C4)
    c3 = _point(points, C3)
    axis = _unit(c4, p5)
    if axis is not None:
        return axis, "C4_to_P5_initial", c4, p5
    axis = _unit(c3, p5)
    if axis is not None:
        return axis, "C3_to_P5_initial", c3, p5
    return None, "unavailable", None, None


def _tail_detail_mask_from_image(
    warped_image: np.ndarray | None,
    full_mask: np.ndarray | None,
    tail_region: tuple[int, int, int, int] | list[int] | None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if tail_region is None:
        return None, {"tail_detail_mask_success": False, "tail_detail_mask_quality": "failed_missing_region"}
    if warped_image is None:
        return full_mask, {
            "tail_detail_mask_success": full_mask is not None,
            "tail_detail_mask_quality": "fallback_full_mask",
            "tail_mask_gap_visible": "",
            "tail_mask_gap_filled": "",
        }
    tx0, ty0, tx1, ty1 = [int(round(float(v))) for v in tail_region]
    tx0, ty0 = max(0, tx0), max(0, ty0)
    tx1 = min(warped_image.shape[1], max(tx0 + 1, tx1))
    ty1 = min(warped_image.shape[0], max(ty0 + 1, ty1))
    roi = warped_image[ty0:ty1, tx0:tx1]
    if roi.size == 0:
        return full_mask, {"tail_detail_mask_success": False, "tail_detail_mask_quality": "failed_empty_roi"}
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    blue_background = ((h >= 82) & (h <= 132) & (s >= 30) & (v >= 30)).astype(np.uint8) * 255
    candidate = cv2.bitwise_not(blue_background)
    # Very light denoise only. Do not close/fill, because the fork gap matters.
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, open_kernel, iterations=1)
    if full_mask is not None and full_mask.size:
        # Keep local fish-like pixels plus a small neighborhood around the full mask.
        local_full = full_mask[ty0:ty1, tx0:tx1]
        gate = cv2.dilate((local_full > 0).astype(np.uint8) * 255, np.ones((35, 35), np.uint8), iterations=1)
        candidate = cv2.bitwise_and(candidate, gate)
    detail = np.zeros(full_mask.shape if full_mask is not None else warped_image.shape[:2], dtype=np.uint8)
    detail[ty0:ty1, tx0:tx1] = candidate
    success = int(np.count_nonzero(candidate)) > 80
    return detail, {
        "tail_detail_mask_success": bool(success),
        "tail_detail_mask_quality": "ok" if success else "failed_small_tail_detail_mask",
    }


def _tail_lobe_tips_from_mask(
    mask: np.ndarray | None,
    fish_bbox: tuple[int, int, int, int],
    p5: tuple[float, float] | None,
    tail_axis: tuple[float, float] | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "tail_mask_quality": "failed",
        "tail_region_bbox": None,
        "P7U_mask": None,
        "P7L_mask": None,
        "review_reason": "",
    }
    if mask is None or p5 is None or tail_axis is None:
        result["review_reason"] = "missing_mask_p5_or_tail_axis"
        return result

    region = _tail_region_bbox(fish_bbox, p5)
    result["tail_region_bbox"] = list(region)
    tx0, ty0, tx1, ty1 = region
    tail_mask = np.zeros_like(mask)
    tail_mask[ty0:ty1, tx0:tx1] = mask[ty0:ty1, tx0:tx1]
    tail_area = int(np.count_nonzero(tail_mask))
    if tail_area < max(120, int(0.002 * max(1, np.count_nonzero(mask)))):
        result["review_reason"] = "tail_region_mask_too_small"
        return result

    contour = _largest_contour_points(mask)
    if len(contour) < 20:
        result["review_reason"] = "tail_contour_too_small"
        return result
    in_region = (
        (contour[:, 0] >= tx0)
        & (contour[:, 0] <= tx1)
        & (contour[:, 1] >= ty0)
        & (contour[:, 1] <= ty1)
    )
    candidates = contour[in_region]
    if len(candidates) < 12:
        result["review_reason"] = "tail_region_contour_too_sparse"
        return result

    normal = (-tail_axis[1], tail_axis[0])
    rel = candidates - np.asarray(p5, dtype=np.float32)
    side = rel[:, 0] * normal[0] + rel[:, 1] * normal[1]
    projection = rel[:, 0] * tail_axis[0] + rel[:, 1] * tail_axis[1]
    projection_ok = projection > -max(12.0, 0.025 * (fish_bbox[2] - fish_bbox[0]))
    candidates = candidates[projection_ok]
    side = side[projection_ok]
    projection = projection[projection_ok]
    if len(candidates) < 12:
        result["review_reason"] = "tail_candidates_have_negative_projection"
        return result

    group_a = candidates[side <= 0]
    proj_a = projection[side <= 0]
    group_b = candidates[side > 0]
    proj_b = projection[side > 0]
    if len(group_a) < 4 or len(group_b) < 4:
        result["review_reason"] = "tail_lobe_groups_missing"
        return result
    # Image y increases downward. Whichever side has the lower median y is visually upper.
    a_is_upper = float(np.median(group_a[:, 1])) <= float(np.median(group_b[:, 1]))
    upper_group, upper_proj = (group_a, proj_a) if a_is_upper else (group_b, proj_b)
    upper_side = side[side <= 0] if a_is_upper else side[side > 0]
    lower_group, lower_proj = (group_b, proj_b) if a_is_upper else (group_a, proj_a)
    upper_candidate_info = _select_p7u_candidate(
        contour=contour,
        candidates=upper_group,
        projections=upper_proj,
        side_values=upper_side,
        p5=p5,
        tail_axis=tail_axis,
    )
    upper = np.asarray(upper_candidate_info["point"], dtype=np.float32)
    lower = lower_group[int(np.argmax(lower_proj))]
    upper_point = (float(upper[0]), float(upper[1]))
    lower_point = (float(lower[0]), float(lower[1]))
    if upper_point[1] >= lower_point[1]:
        result["review_reason"] = "tail_lobes_not_vertically_ordered"
        return result

    result.update(
        {
            "tail_mask_quality": "ok",
            "tail_area_px": tail_area,
            "P7U_mask": upper_point,
            "P7L_mask": lower_point,
            "P7U_mask_quality": upper_candidate_info.get("quality", "good"),
            "P7U_candidate_method": upper_candidate_info.get("method", ""),
            "P7U_candidates": upper_candidate_info.get("candidates", []),
            "P7U_candidate_count": upper_candidate_info.get("candidate_count", 0),
            "review_reason": "",
        }
    )
    return result


def _curvature_score(contour: np.ndarray, point: np.ndarray, step: int = 18) -> float:
    if len(contour) < step * 2 + 1:
        return 0.0
    distances = np.sum((contour - point) ** 2, axis=1)
    idx = int(np.argmin(distances))
    prev = contour[(idx - step) % len(contour)]
    curr = contour[idx]
    nxt = contour[(idx + step) % len(contour)]
    v1 = prev - curr
    v2 = nxt - curr
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 <= 1e-6 or n2 <= 1e-6:
        return 0.0
    cos_value = float(np.dot(v1, v2) / (n1 * n2))
    cos_value = max(-1.0, min(1.0, cos_value))
    angle = math.acos(cos_value)
    return float(max(0.0, min(1.0, (math.pi - angle) / math.pi)))


def _select_p7u_candidate(
    *,
    contour: np.ndarray,
    candidates: np.ndarray,
    projections: np.ndarray,
    side_values: np.ndarray,
    p5: tuple[float, float],
    tail_axis: tuple[float, float],
) -> dict[str, Any]:
    if len(candidates) == 0:
        return {"point": (0.0, 0.0), "quality": "failed", "method": "none", "candidates": [], "candidate_count": 0}
    proj = projections.astype(float)
    y_values = candidates[:, 1].astype(float)
    side_abs = np.abs(side_values.astype(float))
    proj_range = max(1e-6, float(proj.max() - proj.min()))
    y_range = max(1e-6, float(y_values.max() - y_values.min()))
    side_range = max(1e-6, float(side_abs.max() - side_abs.min()))
    norm_proj = (proj - proj.min()) / proj_range
    upper_lobe_separation = (side_abs - side_abs.min()) / side_range

    quantile90 = float(np.quantile(proj, 0.90))
    top_projection_idx = np.where(proj >= quantile90)[0]
    if len(top_projection_idx) == 0:
        top_projection_idx = np.asarray([int(np.argmax(proj))])
    projection_idx = int(np.argmax(proj))
    uppermost_idx = int(top_projection_idx[np.argmin(y_values[top_projection_idx])])

    curvature_scores = np.asarray([_curvature_score(contour, point) for point in candidates], dtype=float)
    top20_idx = np.where(proj >= float(np.quantile(proj, 0.80)))[0]
    if len(top20_idx) == 0:
        top20_idx = np.arange(len(candidates))
    curvature_idx = int(top20_idx[np.argmax(curvature_scores[top20_idx])])
    score = norm_proj + 0.30 * upper_lobe_separation + 0.20 * curvature_scores
    # A small upper-edge preference helps when the projection maximum lands on
    # the side edge of a broad upper lobe instead of the visible lobe end.
    score += 0.15 * ((y_values.max() - y_values) / y_range)
    best_idx = int(np.argmax(score))

    def candidate_payload(idx: int, method: str) -> dict[str, Any]:
        point = candidates[idx]
        return {
            "method": method,
            "x": float(point[0]),
            "y": float(point[1]),
            "projection": float(proj[idx]),
            "score": float(score[idx]),
            "curvature_score": float(curvature_scores[idx]),
            "upper_lobe_separation": float(upper_lobe_separation[idx]),
        }

    payloads = [
        candidate_payload(projection_idx, "projection"),
        candidate_payload(uppermost_idx, "uppermost_top10_projection"),
        candidate_payload(curvature_idx, "curvature_top20_projection"),
        candidate_payload(best_idx, "weighted_score"),
    ]
    point = candidates[best_idx]
    quality = "good"
    if float(score[best_idx]) - float(score[projection_idx]) < 0.04:
        quality = "warning"
    return {
        "point": (float(point[0]), float(point[1])),
        "quality": quality,
        "method": "weighted_score",
        "candidates": payloads,
        "candidate_count": int(len(candidates)),
    }


def _arc_between(contour: np.ndarray, start_idx: int, end_idx: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(contour)
    if n == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=int)
    if start_idx <= end_idx:
        idx = np.arange(start_idx, end_idx + 1)
    else:
        idx = np.concatenate([np.arange(start_idx, n), np.arange(0, end_idx + 1)])
    return contour[idx], idx


def estimate_caudal_fork_from_mask(
    fish_mask: np.ndarray | None,
    p5: tuple[float, float] | None,
    p7u: tuple[float, float] | None,
    p7l: tuple[float, float] | None,
    tail_axis: tuple[float, float] | None,
    tail_region: tuple[int, int, int, int] | list[int] | None,
) -> dict[str, Any]:
    """Estimate P6 as the fork notch between upper/lower caudal lobes."""
    result: dict[str, Any] = {
        "P6_mask_suggestion": None,
        "P6_mask_quality": "failed",
        "P6_review_reason": "",
        "P6_candidate_count": 0,
        "P6_candidate_method": "",
    }
    if fish_mask is None or p5 is None or p7u is None or p7l is None or tail_axis is None or tail_region is None:
        result["P6_review_reason"] = "missing_inputs_for_fork_rule"
        return result
    mask = (fish_mask > 0).astype(np.uint8) * 255
    contour = _largest_contour_points(mask)
    if len(contour) < 20:
        result["P6_review_reason"] = "tail_contour_too_small"
        return result
    tx0, ty0, tx1, ty1 = [int(round(float(v))) for v in tail_region]
    width = max(1, tx1 - tx0)
    normal = (-tail_axis[1], tail_axis[0])

    idx_u = int(np.argmin(np.sum((contour - np.asarray(p7u, dtype=np.float32)) ** 2, axis=1)))
    idx_l = int(np.argmin(np.sum((contour - np.asarray(p7l, dtype=np.float32)) ** 2, axis=1)))
    arc1, _idx1 = _arc_between(contour, idx_u, idx_l)
    arc2, _idx2 = _arc_between(contour, idx_l, idx_u)

    def arc_score(arc: np.ndarray) -> tuple[float, float, float]:
        if len(arc) == 0:
            return -1.0, -1.0, 0.0
        in_region = (
            (arc[:, 0] >= tx0)
            & (arc[:, 0] <= tx1)
            & (arc[:, 1] >= ty0)
            & (arc[:, 1] <= ty1)
        )
        tail_fraction = float(np.mean(in_region))
        rel = arc - np.asarray(p5, dtype=np.float32)
        projection = rel[:, 0] * tail_axis[0] + rel[:, 1] * tail_axis[1]
        return tail_fraction, float(np.mean(projection[in_region])) if np.any(in_region) else float(np.mean(projection)), float(len(arc))

    score1 = arc_score(arc1)
    score2 = arc_score(arc2)
    chosen = arc1 if (score1[0], score1[1]) >= (score2[0], score2[1]) else arc2
    if len(chosen) < 8:
        result["P6_review_reason"] = "fork_arc_too_short"
        return result

    rel = chosen - np.asarray(p5, dtype=np.float32)
    projection = rel[:, 0] * tail_axis[0] + rel[:, 1] * tail_axis[1]
    normal_coord = rel[:, 0] * normal[0] + rel[:, 1] * normal[1]
    max_tip_projection = max(
        _dot(_sub(p7u, p5), tail_axis),
        _dot(_sub(p7l, p5), tail_axis),
        1.0,
    )
    n_u = _dot(_sub(p7u, p5), normal)
    n_l = _dot(_sub(p7l, p5), normal)
    n_min = min(n_u, n_l)
    n_max = max(n_u, n_l)
    valid = (
        (chosen[:, 0] >= tx0)
        & (chosen[:, 0] <= tx1)
        & (chosen[:, 1] >= ty0)
        & (chosen[:, 1] <= ty1)
        & (projection > 0.45 * max_tip_projection)
        & (projection < 0.95 * max_tip_projection)
        & (normal_coord >= n_min - 0.08 * width)
        & (normal_coord <= n_max + 0.08 * width)
    )
    candidates = chosen[valid]
    cand_proj = projection[valid]
    cand_normal = normal_coord[valid]
    if len(candidates) < 3:
        return _estimate_caudal_fork_from_gap(
            fish_mask=fish_mask,
            p5=p5,
            p7u=p7u,
            p7l=p7l,
            tail_axis=tail_axis,
            tail_region=(tx0, ty0, tx1, ty1),
            max_tip_projection=max_tip_projection,
            fallback_reason="fork_arc_candidates_too_sparse",
        )

    midpoint_normal = (n_u + n_l) / 2.0
    normal_penalty = np.abs(cand_normal - midpoint_normal) / max(1.0, abs(n_u - n_l))
    # Choose the deepest inward notch, while preferring the middle of the fork gap.
    score = cand_proj + 0.08 * max_tip_projection * normal_penalty
    best_idx = int(np.argmin(score))
    point = candidates[best_idx]
    quality = "good"
    if len(candidates) < 8:
        quality = "warning"
    result.update(
        {
            "P6_mask_suggestion": (float(point[0]), float(point[1])),
            "P6_mask_quality": quality,
            "P6_review_reason": "",
            "P6_candidate_count": int(len(candidates)),
            "P6_candidate_method": "contour_arc_min_projection",
            "P6_arc_score_1": list(score1),
            "P6_arc_score_2": list(score2),
        }
    )
    return result


def estimate_caudal_fork_gap_center(
    warped_image: np.ndarray | None,
    fish_mask: np.ndarray | None,
    tail_detail_mask: np.ndarray | None,
    p5: tuple[float, float] | None,
    p7u: tuple[float, float] | None,
    p7l: tuple[float, float] | None,
    tail_axis: tuple[float, float] | None,
    tail_region: tuple[int, int, int, int] | list[int] | None,
) -> dict[str, Any]:
    """Estimate P6 from the blue/background gap between caudal lobes."""
    result = {
        "P6_gap_suggestion": None,
        "P6_gap_quality": "failed",
        "P6_gap_review_reason": "",
        "tail_mask_gap_visible": False,
        "tail_mask_gap_filled": "",
        "P6_gap_candidate_count": 0,
    }
    if tail_detail_mask is None or p5 is None or p7u is None or p7l is None or tail_axis is None or tail_region is None:
        result["P6_gap_review_reason"] = "missing_inputs_for_gap_rule"
        return result
    tx0, ty0, tx1, ty1 = [int(round(float(v))) for v in tail_region]
    tx0, ty0 = max(0, tx0), max(0, ty0)
    tx1 = min(tail_detail_mask.shape[1], max(tx0 + 1, tx1))
    ty1 = min(tail_detail_mask.shape[0], max(ty0 + 1, ty1))
    if tx1 <= tx0 or ty1 <= ty0:
        result["P6_gap_review_reason"] = "empty_tail_region"
        return result

    normal = (-tail_axis[1], tail_axis[0])
    max_tip_projection = max(
        _dot(_sub(p7u, p5), tail_axis),
        _dot(_sub(p7l, p5), tail_axis),
        1.0,
    )
    n_u = _dot(_sub(p7u, p5), normal)
    n_l = _dot(_sub(p7l, p5), normal)
    n_min, n_max = min(n_u, n_l), max(n_u, n_l)
    n_mid = (n_u + n_l) / 2.0
    n_span = max(1.0, n_max - n_min)
    yy, xx = np.mgrid[ty0:ty1, tx0:tx1]
    fish = tail_detail_mask[ty0:ty1, tx0:tx1] > 0
    background = ~fish
    rel_x = xx.astype(float) - p5[0]
    rel_y = yy.astype(float) - p5[1]
    projection = rel_x * tail_axis[0] + rel_y * tail_axis[1]
    normal_coord = rel_x * normal[0] + rel_y * normal[1]
    candidate = (
        background
        & (projection > 0.45 * max_tip_projection)
        & (projection < 0.92 * max_tip_projection)
        & (normal_coord > n_min + 0.15 * n_span)
        & (normal_coord < n_max - 0.15 * n_span)
        & (np.abs(normal_coord - n_mid) < 0.34 * n_span)
    )
    if int(np.count_nonzero(candidate)) < 20:
        result["P6_gap_review_reason"] = "gap_candidate_too_sparse"
        result["tail_mask_gap_filled"] = True
        return result

    labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=8)
    components = []
    for label_id in range(1, labels_count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < 15:
            continue
        ys, xs = np.where(labels == label_id)
        comp_proj = projection[ys, xs]
        comp_norm = normal_coord[ys, xs]
        components.append(
            {
                "label": label_id,
                "area": area,
                "min_projection": float(np.min(comp_proj)),
                "median_projection": float(np.median(comp_proj)),
                "mid_distance": float(abs(np.median(comp_norm) - n_mid)),
                "ys": ys,
                "xs": xs,
            }
        )
    if not components:
        result["P6_gap_review_reason"] = "gap_components_missing"
        result["tail_mask_gap_filled"] = True
        return result
    # Prefer the connected background gap that reaches furthest toward the body
    # and stays near the normal-axis midpoint between lobes.
    components.sort(key=lambda item: (item["min_projection"] + 0.12 * max_tip_projection * item["mid_distance"] / n_span, -item["area"]))
    comp = components[0]
    comp_proj = projection[comp["ys"], comp["xs"]]
    cutoff = float(np.quantile(comp_proj, 0.05))
    left = comp_proj <= cutoff + 1.0
    if int(np.count_nonzero(left)) == 0:
        left = comp_proj == float(np.min(comp_proj))
    xs_abs = comp["xs"][left] + tx0
    ys_abs = comp["ys"][left] + ty0
    raw_point = (float(np.median(xs_abs)), float(np.median(ys_abs)))
    raw_rel = (raw_point[0] - p5[0], raw_point[1] - p5[1])
    raw_projection = _dot(raw_rel, tail_axis)
    # The gap pixels often lie on one edge of the blue fork gap. P6 is the
    # midpoint of the fork notch, so keep the gap's axial depth but recenter it
    # between upper/lower caudal lobes along the normal axis.
    point = (
        float(p5[0] + raw_projection * tail_axis[0] + n_mid * normal[0]),
        float(p5[1] + raw_projection * tail_axis[1] + n_mid * normal[1]),
    )
    quality = "good" if comp["area"] > 60 and len(components) <= 8 else "warning"
    result.update(
        {
            "P6_gap_suggestion": point,
            "P6_gap_raw_suggestion": raw_point,
            "P6_gap_quality": quality,
            "P6_gap_review_reason": "",
            "tail_mask_gap_visible": True,
            "tail_mask_gap_filled": False,
            "P6_gap_candidate_count": int(np.count_nonzero(candidate)),
            "P6_gap_component_area": int(comp["area"]),
        }
    )
    return result


def estimate_p6_from_tail_fork_gap(
    tail_mask: np.ndarray | None,
    fish_mask: np.ndarray | None,
    keypoints: Mapping[str, Any],
    image_shape: tuple[int, ...] | list[int] | None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive P6 from the V-shaped background gap between caudal lobes.

    The returned point is a geometric suggestion/QC target.  Callers decide
    whether it is used for an unconfirmed preannotation draft; confirmed labels
    must not be overwritten by this function.
    """
    del image_shape
    cfg = dict(config or {})
    p5 = _point(keypoints, P5)
    p7u = _point(keypoints, P7U)
    p7l = _point(keypoints, P7L)
    current_p6 = _point(keypoints, P6)
    axis_payload = select_tail_axis_without_p6(keypoints)
    tail_axis = _as_xy(axis_payload.get("tail_axis"))
    mask = _mask_clean(fish_mask) if fish_mask is not None else None
    detail = tail_mask if tail_mask is not None else mask
    result: dict[str, Any] = {
        "P6_geometric": None,
        "P6_geometric_confidence": 0.0,
        "P6_gap_qc_pass": False,
        "P6_gap_review_reason": "",
        "P6_geometric_available": False,
        "P6_current_to_geometric_distance_px": "",
        "P6_current_to_geometric_distance_mm": "",
        "P6_fallback_recommended": False,
        "P6_fallback_used": False,
        "P6_review_reason": "",
        "tail_gap_component": None,
        "tail_gap_tip": None,
        "tail_gap_centerline": [],
        "tail_gap_depth": "",
        "tail_gap_width": "",
        "tail_axis_source": axis_payload.get("tail_axis_source", "unavailable"),
        "tail_axis_uses_P6": bool(axis_payload.get("tail_axis_uses_P6", False)),
        "tail_axis_vector": _serial_point(tail_axis),
        "tail_region_bbox": None,
    }
    if mask is None or p5 is None or p7u is None or p7l is None or tail_axis is None:
        result["P6_gap_review_reason"] = "missing_mask_or_tail_points_for_gap_derivation"
        result["P6_review_reason"] = result["P6_gap_review_reason"]
        return result

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        result["P6_gap_review_reason"] = "empty_fish_mask"
        result["P6_review_reason"] = result["P6_gap_review_reason"]
        return result
    fish_bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    tail_region = cfg.get("tail_region") or _tail_region_bbox(fish_bbox, p5)
    result["tail_region_bbox"] = list(tail_region)
    if detail is None:
        detail = mask

    gap = estimate_caudal_fork_gap_center(
        warped_image=cfg.get("warped_image"),
        fish_mask=mask,
        tail_detail_mask=detail,
        p5=p5,
        p7u=p7u,
        p7l=p7l,
        tail_axis=tail_axis,
        tail_region=tail_region,
    )
    point = _as_xy(gap.get("P6_gap_suggestion"))
    if point is None or gap.get("P6_gap_quality") not in {"good", "warning"}:
        concavity = estimate_caudal_fork_from_mask(
            fish_mask=detail,
            p5=p5,
            p7u=p7u,
            p7l=p7l,
            tail_axis=tail_axis,
            tail_region=tail_region,
        )
        point = _as_xy(concavity.get("P6_mask_suggestion"))
        if point is None or concavity.get("P6_mask_quality") not in {"good", "warning"}:
            reason = str(gap.get("P6_gap_review_reason") or concavity.get("P6_review_reason") or "tail_gap_rule_failed")
            result.update(
                {
                    "P6_gap_review_reason": reason,
                    "P6_review_reason": reason,
                    "P6_gap_qc_pass": False,
                    "P6_geometric_available": False,
                }
            )
            return result
        gap_quality = str(concavity.get("P6_mask_quality", "warning"))
        method = "concavity_fallback"
        raw_point = point
        candidate_count = int(concavity.get("P6_candidate_count", 0) or 0)
    else:
        gap_quality = str(gap.get("P6_gap_quality", "warning"))
        method = "tail_fork_gap_geometric"
        raw_point = _as_xy(gap.get("P6_gap_raw_suggestion")) or point
        candidate_count = int(gap.get("P6_gap_candidate_count", 0) or 0)

    qc = validate_p6_geometry(
        p5,
        point,
        p7u,
        p7l,
        None,
        tail_axis,
        fish_mask=mask,
        tail_mask=detail,
        config={"P6_fallback": point, "leaf_balance_min": 0.4, "leaf_balance_max": 2.5},
    )
    gap_qc_pass = bool(qc.get("P6_geometry_qc_pass", False))
    dist_px = _distance(current_p6, point) if current_p6 is not None else None
    mm_per_pixel = float(cfg.get("mm_per_pixel", 0.1) or 0.1)
    dist_mm = None if dist_px is None else dist_px * mm_per_pixel
    threshold_mm = float(cfg.get("P6_current_to_geometric_threshold_mm", 8.0))
    fallback_recommended = gap_qc_pass and (not bool(qc.get("P6_geometry_qc_pass", False)) or (dist_mm is not None and dist_mm > threshold_mm))
    # qc above validates the geometric point itself. Check the current P6 separately.
    current_qc = validate_p6_geometry(
        p5,
        current_p6,
        p7u,
        p7l,
        None,
        tail_axis,
        fish_mask=mask,
        tail_mask=detail,
        config={"P6_fallback": point, "leaf_balance_min": 0.4, "leaf_balance_max": 2.5},
    )
    fallback_recommended = bool(point is not None and gap_qc_pass and not bool(current_qc.get("P6_geometry_qc_pass", False)))
    if dist_mm is not None and dist_mm > threshold_mm and gap_qc_pass:
        fallback_recommended = True
    review_parts = []
    if not gap_qc_pass:
        review_parts.append(str(qc.get("P6_review_reason") or "P6_gap_geometry_qc_failed"))
    if fallback_recommended:
        review_parts.append("P6_current_differs_from_tail_fork_gap")

    result.update(
        {
            "P6_geometric": _serial_point(point),
            "P6_geometric_confidence": 0.90 if gap_quality == "good" else 0.65,
            "P6_gap_qc_pass": gap_qc_pass,
            "P6_gap_review_reason": str(qc.get("P6_review_reason", "")),
            "P6_geometric_available": True,
            "P6_current_to_geometric_distance_px": "" if dist_px is None else float(dist_px),
            "P6_current_to_geometric_distance_mm": "" if dist_mm is None else float(dist_mm),
            "P6_fallback_recommended": fallback_recommended,
            "P6_fallback_used": False,
            "P6_review_reason": ";".join(_unique_reasons(review_parts)),
            "tail_gap_component": {
                "method": method,
                "quality": gap_quality,
                "candidate_count": candidate_count,
                "component_area": gap.get("P6_gap_component_area", ""),
            },
            "tail_gap_tip": _serial_point(point),
            "tail_gap_raw_tip": _serial_point(raw_point),
            "tail_gap_centerline": [_serial_point(raw_point), _serial_point(point)] if raw_point is not None else [_serial_point(point)],
            "tail_gap_depth": _dot(_sub(point, p5), tail_axis),
            "tail_gap_width": abs(_dot(_sub(p7u, p5), (-tail_axis[1], tail_axis[0])) - _dot(_sub(p7l, p5), (-tail_axis[1], tail_axis[0]))),
            "P6_geometry_qc": qc,
            "P6_current_geometry_qc": current_qc,
            "tail_mask_gap_visible": gap.get("tail_mask_gap_visible", ""),
            "tail_mask_gap_filled": gap.get("tail_mask_gap_filled", ""),
        }
    )
    return result


def _estimate_caudal_fork_from_gap(
    *,
    fish_mask: np.ndarray,
    p5: tuple[float, float],
    p7u: tuple[float, float],
    p7l: tuple[float, float],
    tail_axis: tuple[float, float],
    tail_region: tuple[int, int, int, int],
    max_tip_projection: float,
    fallback_reason: str,
) -> dict[str, Any]:
    tx0, ty0, tx1, ty1 = tail_region
    mask = fish_mask > 0
    best: tuple[float, float, float] | None = None
    min_proj = 0.45 * max_tip_projection
    max_proj = 0.95 * max_tip_projection
    for x in range(tx0, tx1):
        ys = np.where(mask[ty0:ty1, x])[0] + ty0
        if len(ys) < 8:
            continue
        runs: list[tuple[int, int]] = []
        start = int(ys[0])
        prev = int(ys[0])
        for y in ys[1:]:
            y = int(y)
            if y > prev + 1:
                runs.append((start, prev))
                start = y
            prev = y
        runs.append((start, prev))
        if len(runs) < 2:
            continue
        gaps = [(runs[i][1], runs[i + 1][0], runs[i + 1][0] - runs[i][1]) for i in range(len(runs) - 1)]
        gap_top, gap_bottom, gap_size = max(gaps, key=lambda item: item[2])
        if gap_size < 8:
            continue
        y_mid = (gap_top + gap_bottom) / 2.0
        projection = _dot((float(x) - p5[0], y_mid - p5[1]), tail_axis)
        if not (min_proj <= projection <= max_proj):
            continue
        if best is None or projection < best[2]:
            best = (float(x), float(y_mid), float(projection))
    if best is None:
        return {
            "P6_mask_suggestion": None,
            "P6_mask_quality": "failed",
            "P6_review_reason": f"{fallback_reason};fork_gap_rule_failed",
            "P6_candidate_count": 0,
            "P6_candidate_method": "gap_fallback_failed",
        }
    return {
        "P6_mask_suggestion": (best[0], best[1]),
        "P6_mask_quality": "warning",
        "P6_review_reason": fallback_reason,
        "P6_candidate_count": 1,
        "P6_candidate_method": "vertical_gap_leftmost_split",
    }


def _p5_qc_and_suggestion(
    mask: np.ndarray | None,
    points: Mapping[str, Any],
    fish_bbox: tuple[int, int, int, int],
) -> tuple[dict[str, Any], dict[str, tuple[float, float]]]:
    x0, y0, x1, y1 = fish_bbox
    width = max(1, x1 - x0)
    p4 = _point(points, P4)
    p5 = _point(points, P5)
    p6 = _point(points, P6)
    c4 = _point(points, C4)
    reasons: list[str] = []
    suggestions: dict[str, tuple[float, float]] = {}
    if p5 is None:
        reasons.append("missing_P5")
    else:
        if p4 is not None and p5[0] <= p4[0] + max(15.0, 0.03 * width):
            reasons.append("P5_not_right_of_P4")
        if p6 is not None and p5[0] >= p6[0] - max(10.0, 0.015 * width):
            reasons.append("P5_not_left_of_P6")
        if p5[0] < x0 + 0.58 * width or p5[0] > x1 + 0.03 * width:
            reasons.append("P5_outside_expected_tail_base_zone")
        if c4 is not None and abs(p5[1] - c4[1]) > max(80.0, 0.25 * (y1 - y0)):
            reasons.append("P5_far_from_C4_axis")

    if mask is not None:
        search_x0 = int(round(max(x0 + 0.62 * width, (p4[0] if p4 else x0 + 0.62 * width))))
        search_x1 = int(round(min(x1 - 0.04 * width, (p6[0] if p6 else x1 - 0.04 * width))))
        if search_x1 > search_x0:
            rows = _mask_bounds_by_column(mask, search_x0, search_x1, min_column_pixels=8)
            if rows:
                spans = np.asarray([row[3] for row in rows], dtype=float)
                xs = np.asarray([row[0] for row in rows], dtype=float)
                min_idx = int(np.argmin(spans))
                target_idx = min_idx
                for idx in range(min_idx + 1, len(rows)):
                    if spans[idx] > spans[min_idx] * 1.25 and xs[idx] > xs[min_idx] + max(8.0, 0.015 * width):
                        target_idx = idx
                        break
                sx, top, bottom, _span = rows[target_idx]
                suggestions[P5] = (float(sx), float((top + bottom) / 2.0))
    suggestion = suggestions.get(P5)
    if suggestion is not None and p5 is not None:
        if _distance(p5, suggestion) > max(45.0, 0.05 * width):
            reasons.append("P5_far_from_mask_transition_suggestion")
    return {
        "P5_qc_pass": not reasons,
        "P5_review_reason": ";".join(_unique_reasons(reasons)),
        "P5_mask_suggestion": _serial_point(suggestion),
    }, suggestions


def _depth_suggestions_and_qc(
    mask: np.ndarray | None,
    points: Mapping[str, Any],
    fish_bbox: tuple[int, int, int, int],
) -> tuple[dict[str, tuple[float, float]], dict[str, Any], dict[str, Any]]:
    suggestions: dict[str, tuple[float, float]] = {}
    x0, y0, x1, y1 = fish_bbox
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    if mask is None:
        return suggestions, {"body_depth_qc_pass": False, "body_depth_review_reason": "missing_mask"}, {
            "peduncle_depth_qc_pass": False,
            "peduncle_depth_review_reason": "missing_mask",
        }

    c1 = _point(points, C1)
    c3 = _point(points, C3)
    body_x0 = int(round(max(x0 + 0.20 * width, c1[0] if c1 else x0 + 0.20 * width)))
    body_x1 = int(round(min(x0 + 0.70 * width, c3[0] if c3 else x0 + 0.70 * width)))
    if body_x1 <= body_x0:
        body_x0 = int(round(x0 + 0.20 * width))
        body_x1 = int(round(x0 + 0.70 * width))
    body_rows = _mask_bounds_by_column(mask, body_x0, body_x1, min_column_pixels=12)
    body_best = _span_suggestion(body_rows, "max")
    body_reason: list[str] = []
    if body_best is not None:
        sx, top, bottom, span = body_best
        # Pull the suggestion slightly inward to reduce fin-tip bias from a noisy mask.
        inset = max(0.0, min(8.0, 0.03 * span))
        suggestions[P8] = (sx, top + inset)
        suggestions[P9] = (sx, bottom - inset)
        p8 = _point(points, P8)
        p9 = _point(points, P9)
        if p8 is not None and _distance(p8, suggestions[P8]) > max(40.0, 0.045 * width):
            body_reason.append("P8_far_from_body_depth_suggestion")
        if p9 is not None and _distance(p9, suggestions[P9]) > max(40.0, 0.045 * width):
            body_reason.append("P9_far_from_body_depth_suggestion")
        if p8 is not None and p9 is not None and p8[1] >= p9[1]:
            body_reason.append("P8_not_above_P9")
    else:
        body_reason.append("body_depth_suggestion_unavailable")

    p4 = _point(points, P4)
    p5 = _point(points, P5)
    if p4 is not None and p5 is not None and p5[0] > p4[0] + 20:
        ped_x0 = int(round(max(x0 + 0.62 * width, min(p4[0], p5[0]))))
        ped_x1 = int(round(min(x1 - 0.03 * width, max(p4[0], p5[0]))))
    else:
        ped_x0 = int(round(x0 + 0.70 * width))
        ped_x1 = int(round(x0 + 0.90 * width))
    if ped_x1 <= ped_x0:
        ped_x0 = int(round(x0 + 0.70 * width))
        ped_x1 = int(round(x0 + 0.90 * width))
    ped_rows = _mask_bounds_by_column(mask, ped_x0, ped_x1, min_column_pixels=7)
    ped_best = _span_suggestion(ped_rows, "min")
    ped_reason: list[str] = []
    if ped_best is not None:
        sx, top, bottom, span = ped_best
        suggestions[P10] = (sx, top)
        suggestions[P11] = (sx, bottom)
        p10 = _point(points, P10)
        p11 = _point(points, P11)
        c4 = _point(points, C4)
        if p10 is not None and _distance(p10, suggestions[P10]) > max(32.0, 0.035 * width):
            ped_reason.append("P10_far_from_peduncle_suggestion")
        if p11 is not None and _distance(p11, suggestions[P11]) > max(32.0, 0.035 * width):
            ped_reason.append("P11_far_from_peduncle_suggestion")
        if p10 is not None and p11 is not None and p10[1] >= p11[1]:
            ped_reason.append("P10_not_above_P11")
        if c4 is not None and p10 is not None and p11 is not None:
            mid_y = (p10[1] + p11[1]) / 2.0
            if abs(c4[1] - mid_y) > max(35.0, 0.11 * height):
                ped_reason.append("C4_far_from_peduncle_depth_midpoint")
    else:
        ped_reason.append("peduncle_depth_suggestion_unavailable")

    return suggestions, {
        "body_depth_qc_pass": not body_reason,
        "body_depth_review_reason": ";".join(_unique_reasons(body_reason)),
        "P8_mask_suggestion": _serial_point(suggestions.get(P8)),
        "P9_mask_suggestion": _serial_point(suggestions.get(P9)),
    }, {
        "peduncle_depth_qc_pass": not ped_reason,
        "peduncle_depth_review_reason": ";".join(_unique_reasons(ped_reason)),
        "P10_mask_suggestion": _serial_point(suggestions.get(P10)),
        "P11_mask_suggestion": _serial_point(suggestions.get(P11)),
    }


def _axis_qc(
    mask: np.ndarray | None,
    points: Mapping[str, Any],
    fish_bbox: tuple[int, int, int, int],
    suggestions: Mapping[str, tuple[float, float]],
) -> dict[str, Any]:
    x0, _y0, x1, y1 = fish_bbox
    width = max(1, x1 - x0)
    reasons: list[str] = []
    axis_keys = [P1, C1, C2, C3, P4, C4, P5]
    axis_points = [_point(points, key) for key in axis_keys]
    if not all(_finite_point(point) for point in axis_points):
        reasons.append("axis_points_missing")
        curvature = ""
    else:
        xs = [point[0] for point in axis_points if point is not None]
        if not all(b >= a - 0.035 * width for a, b in zip(xs, xs[1:])):
            reasons.append("axis_projection_not_monotonic")
        if mask is not None:
            for key, point in zip(axis_keys[1:], axis_points[1:]):
                if not _point_in_mask(mask, point, radius=8):
                    reasons.append(f"{key}_outside_mask")
        p1 = axis_points[0]
        p5 = axis_points[-1]
        straight = _distance(p1, p5) if p1 is not None and p5 is not None else 0.0
        polyline = sum(_distance(a, b) for a, b in zip(axis_points, axis_points[1:]) if a is not None and b is not None)
        curvature = polyline / straight if straight > 1 else ""
        if curvature and curvature > 1.14:
            reasons.append("curvature_index_unusual")
        for a, b, c in zip(axis_points, axis_points[1:], axis_points[2:]):
            if a is None or b is None or c is None:
                continue
            v1 = _unit(b, a)
            v2 = _unit(b, c)
            if v1 is None or v2 is None:
                continue
            angle_cos = max(-1.0, min(1.0, _dot(v1, v2)))
            angle = math.degrees(math.acos(angle_cos))
            if angle < 120:
                reasons.append("body_axis_sharp_bend")
                break

    c2 = _point(points, C2)
    p8s = suggestions.get(P8)
    p9s = suggestions.get(P9)
    if c2 is not None and p8s is not None and p9s is not None:
        body_mid = (p8s[0], (p8s[1] + p9s[1]) / 2.0)
        if _distance(c2, body_mid) > max(55.0, 0.055 * width):
            reasons.append("C2_far_from_body_depth_midpoint")
    c4 = _point(points, C4)
    p10s = suggestions.get(P10)
    p11s = suggestions.get(P11)
    if c4 is not None and p10s is not None and p11s is not None:
        ped_mid = (p10s[0], (p10s[1] + p11s[1]) / 2.0)
        if _distance(c4, ped_mid) > max(35.0, 0.04 * width):
            reasons.append("C4_far_from_peduncle_depth_midpoint")

    return {
        "axis_qc_pass": not reasons,
        "axis_qc_reason": ";".join(_unique_reasons(reasons)),
        "curvature_index": float(curvature) if isinstance(curvature, float) else "",
    }


def _p7v_qc(
    points: Mapping[str, Any],
    fish_bbox: tuple[int, int, int, int],
    tail_axis: tuple[float, float] | None,
    p5: tuple[float, float] | None,
    upstream_reasons: list[str],
) -> dict[str, Any]:
    x0, y0, x1, y1 = fish_bbox
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    p7u = _point(points, P7U)
    p7l = _point(points, P7L)
    reasons = list(upstream_reasons)
    p7v = None
    selected = ""
    extension = ""
    proj_upper = ""
    proj_lower = ""

    if p5 is None or tail_axis is None or p7u is None or p7l is None:
        reasons.append("missing_tail_points_for_p7v")
    else:
        proj_u = _dot(_sub(p7u, p5), tail_axis)
        proj_l = _dot(_sub(p7l, p5), tail_axis)
        proj_upper = float(proj_u)
        proj_lower = float(proj_l)
        if proj_u >= proj_l:
            selected = "upper"
            ext = proj_u
        else:
            selected = "lower"
            ext = proj_l
        extension = float(max(0.0, ext))
        p7v = (p5[0] + float(extension) * tail_axis[0], p5[1] + float(extension) * tail_axis[1])
        if p7u[1] >= p7l[1]:
            reasons.append("P7U_not_above_P7L")
        if float(extension) < max(20.0, 0.035 * width):
            reasons.append("caudal_extension_too_short")
        if float(extension) > max(180.0, 0.40 * width):
            reasons.append("caudal_extension_too_long")
        if p7v[0] < x0 + 0.60 * width or p7v[0] > x1 + 0.12 * width:
            reasons.append("P7V_x_outside_tail_zone")
        tail_y_min = min(p7u[1], p7l[1])
        tail_y_max = max(p7u[1], p7l[1])
        if not (tail_y_min - 0.22 * height <= p7v[1] <= tail_y_max + 0.22 * height):
            reasons.append("P7V_y_far_from_tail_lobes")

    clean_reasons = _unique_reasons(reasons)
    return {
        "p7v_valid": not clean_reasons,
        "p7v_suggestion": _serial_point(p7v),
        "caudal_tip_selected": selected,
        "caudal_extension_axis_px": extension,
        "proj_upper_px": proj_upper,
        "proj_lower_px": proj_lower,
        "p7v_review_reason": ";".join(clean_reasons),
    }


def apply_tail_geometry_rules(
    *,
    warped_image: np.ndarray | None = None,
    fish_mask: np.ndarray | None,
    fish_bbox: tuple[int, int, int, int] | list[int],
    corrected_keypoints: FullKeypoints,
    model_keypoints_raw: FullKeypoints,
    confidences: Mapping[str, float] | None = None,
    segmentation_success: bool = True,
) -> dict[str, Any]:
    """Apply v0.3.1 mask/geometric tail suggestions and QC.

    The returned ``corrected_keypoints`` is a preannotation draft. Users can still
    edit every point; this function is not used to overwrite confirmed labels.
    """
    bbox = _bbox_tuple(fish_bbox)
    mask = _mask_clean(fish_mask) if segmentation_success else None
    corrected: dict[str, tuple[float, float]] = {
        key: tuple(_as_xy(value) or (0.0, 0.0)) for key, value in corrected_keypoints.items()
    }
    point_sources: dict[str, str] = {key: "model_or_manual" for key in corrected}
    suggestions: dict[str, tuple[float, float]] = {}
    geometric_suggestions: dict[str, tuple[float, float]] = {}
    review_reasons: list[str] = []
    flagged: dict[str, str] = {}

    depth_suggestions, body_depth_qc, peduncle_depth_qc = _depth_suggestions_and_qc(mask, corrected, bbox)
    suggestions.update(depth_suggestions)
    geometric_suggestions.update(depth_suggestions)

    initial_axis, initial_axis_source, axis_start, axis_end = _tail_axis_fallback_before_lobes(corrected)
    tail_rule = _tail_lobe_tips_from_mask(mask, bbox, _point(corrected, P5), initial_axis)
    if tail_rule.get("tail_mask_quality") == "ok":
        p7u_mask = tail_rule["P7U_mask"]
        p7l_mask = tail_rule["P7L_mask"]
        corrected[P7U] = p7u_mask
        corrected[P7L] = p7l_mask
        suggestions[P7U] = p7u_mask
        suggestions[P7L] = p7l_mask
        geometric_suggestions[P7U] = p7u_mask
        geometric_suggestions[P7L] = p7l_mask
        point_sources[P7U] = "mask_tail_rule"
        point_sources[P7L] = "mask_tail_rule"
    else:
        review_reasons.append("tail_mask_rule_failed")
        if tail_rule.get("review_reason"):
            review_reasons.append(str(tail_rule["review_reason"]))
        flagged[P7U] = "tail_mask_rule_failed"
        flagged[P7L] = "tail_mask_rule_failed"

    lobe_axis, lobe_axis_source, lobe_axis_start, lobe_axis_end = _tail_axis_from_lobes(corrected)
    tail_detail_mask, tail_detail_quality = _tail_detail_mask_from_image(
        warped_image,
        mask,
        tail_rule.get("tail_region_bbox"),
    )
    p6_gap_rule = estimate_caudal_fork_gap_center(
        warped_image=warped_image,
        p5=_point(corrected, P5),
        p7u=_point(corrected, P7U),
        p7l=_point(corrected, P7L),
        fish_mask=mask,
        tail_detail_mask=tail_detail_mask,
        tail_axis=lobe_axis,
        tail_region=tail_rule.get("tail_region_bbox"),
    )
    p6_concavity_rule = {"P6_mask_suggestion": None, "P6_mask_quality": "not_used"}
    if p6_gap_rule.get("P6_gap_quality") in {"good", "warning"} and p6_gap_rule.get("P6_gap_suggestion") is not None:
        corrected[P6] = p6_gap_rule["P6_gap_suggestion"]
        suggestions[P6] = p6_gap_rule["P6_gap_suggestion"]
        geometric_suggestions[P6] = p6_gap_rule["P6_gap_suggestion"]
        point_sources[P6] = "tail_gap_rule"
        p6_quality = str(p6_gap_rule.get("P6_gap_quality", "good"))
    else:
        p6_concavity_rule = estimate_caudal_fork_from_mask(
            fish_mask=tail_detail_mask if tail_detail_mask is not None else mask,
            p5=_point(corrected, P5),
            p7u=_point(corrected, P7U),
            p7l=_point(corrected, P7L),
            tail_axis=lobe_axis,
            tail_region=tail_rule.get("tail_region_bbox"),
        )
        if p6_concavity_rule.get("P6_mask_quality") in {"good", "warning"} and p6_concavity_rule.get("P6_mask_suggestion") is not None:
            corrected[P6] = p6_concavity_rule["P6_mask_suggestion"]
            suggestions[P6] = p6_concavity_rule["P6_mask_suggestion"]
            geometric_suggestions[P6] = p6_concavity_rule["P6_mask_suggestion"]
            point_sources[P6] = "concavity_fallback"
            p6_quality = str(p6_concavity_rule.get("P6_mask_quality", "warning"))
        else:
            p6_quality = "failed"
            review_reasons.append("P6_rule_failed")
            flagged[P6] = "P6_rule_failed"
            point_sources[P6] = "model_unreliable"

    p5_qc, p5_suggestions = _p5_qc_and_suggestion(mask, corrected, bbox)
    suggestions.update(p5_suggestions)
    geometric_suggestions.update(p5_suggestions)
    tail_axis, tail_axis_source, axis_start, axis_end = _tail_axis_from_lobes(corrected)
    p6_qc = _p6_basic_qc(corrected, bbox, confidences, p6_mask_quality=p6_quality)
    p7v_inputs_bad: list[str] = []
    if not p5_qc["P5_qc_pass"]:
        review_reasons.append("P5_caudal_base_unreliable")
        p7v_inputs_bad.append("P5_qc_failed")
        flagged[P5] = "P5_caudal_base_unreliable"
    if not p6_qc["P6_qc_pass"]:
        review_reasons.append("P6_caudal_fork_unreliable")
        flagged[P6] = "P6_caudal_fork_unreliable"
        if tail_axis_source in {"C4_to_P5_fallback", "C3_to_P5_fallback"}:
            review_reasons.append("P6_unreliable_but_tail_axis_fallback_used")
        else:
            review_reasons.append("P6_fork_notch_unclear")
    if tail_axis is None:
        p7v_inputs_bad.append("tail_axis_unavailable")
    p7v_qc = _p7v_qc(corrected, bbox, tail_axis, _point(corrected, P5), p7v_inputs_bad)
    if not p7v_qc["p7v_valid"]:
        review_reasons.append("invalid_p7v_after_tail_qc")
        flagged[P7U] = flagged.get(P7U, "p7v_invalid")
        flagged[P7L] = flagged.get(P7L, "p7v_invalid")

    if not body_depth_qc["body_depth_qc_pass"]:
        review_reasons.append("body_depth_points_need_review")
        flagged.setdefault(P8, "body_depth_qc_failed")
        flagged.setdefault(P9, "body_depth_qc_failed")
    if not peduncle_depth_qc["peduncle_depth_qc_pass"]:
        review_reasons.append("peduncle_depth_points_unreliable")
        flagged.setdefault(P10, "peduncle_depth_qc_failed")
        flagged.setdefault(P11, "peduncle_depth_qc_failed")
    axis_qc = _axis_qc(mask, corrected, bbox, suggestions)
    if not axis_qc["axis_qc_pass"]:
        review_reasons.append("body_axis_qc_failed")

    tail_geometry_qc = {
        "tail_qc_pass": not any(reason in review_reasons for reason in ("tail_mask_rule_failed", "invalid_p7v_after_tail_qc")),
        "tail_mask_quality": tail_rule.get("tail_mask_quality", "failed"),
        "tail_mask_review_reason": tail_rule.get("review_reason", ""),
        "tail_axis_source": tail_axis_source,
        "tail_axis_used_without_P6": True,
        "tail_axis_used_without_model_P6": True,
        "tail_axis_start": _serial_point(axis_start),
        "tail_axis_end": _serial_point(axis_end),
        "tail_axis_vector": _serial_point(tail_axis),
        "tail_axis_dx": tail_axis[0] if tail_axis is not None else "",
        "tail_axis_dy": tail_axis[1] if tail_axis is not None else "",
        "tail_region_bbox": tail_rule.get("tail_region_bbox"),
        "P7U_source": point_sources.get(P7U, "model_or_manual"),
        "P7L_source": point_sources.get(P7L, "model_or_manual"),
        "tail_detail_mask_success": tail_detail_quality.get("tail_detail_mask_success", ""),
        "tail_detail_mask_quality": tail_detail_quality.get("tail_detail_mask_quality", ""),
        "tail_mask_gap_visible": p6_gap_rule.get("tail_mask_gap_visible", ""),
        "tail_mask_gap_filled": p6_gap_rule.get("tail_mask_gap_filled", ""),
        "P6_source": point_sources.get(P6, "model_unreliable"),
        "P6_mask": _serial_point(corrected.get(P6)),
        "P6_mask_quality": p6_quality,
        "P6_gap": _serial_point(p6_gap_rule.get("P6_gap_suggestion")),
        "P6_gap_quality": p6_gap_rule.get("P6_gap_quality", "failed"),
        "P6_concavity": _serial_point(p6_concavity_rule.get("P6_mask_suggestion")),
        "P6_concavity_quality": p6_concavity_rule.get("P6_mask_quality", "not_used"),
        "P6_final": _serial_point(corrected.get(P6)),
        "P6_mask_review_reason": p6_gap_rule.get("P6_gap_review_reason", "") or p6_concavity_rule.get("P6_review_reason", ""),
        "P6_candidate_count": p6_gap_rule.get("P6_gap_candidate_count", p6_concavity_rule.get("P6_candidate_count", 0)),
        "P6_candidate_method": point_sources.get(P6, "model_unreliable"),
        "P7U_mask": _serial_point(tail_rule.get("P7U_mask")),
        "P7L_mask": _serial_point(tail_rule.get("P7L_mask")),
        "P7U_mask_quality": tail_rule.get("P7U_mask_quality", ""),
        "P7U_candidate_method": tail_rule.get("P7U_candidate_method", ""),
        "P7U_candidate_count": tail_rule.get("P7U_candidate_count", 0),
        "P7U_candidates": tail_rule.get("P7U_candidates", []),
        "P7U_review_reason": "" if tail_rule.get("P7U_mask_quality") in {"good", "warning"} else "P7U_mask_rule_unreliable",
        "P7L_mask_quality": "good" if tail_rule.get("P7L_mask") is not None else "failed",
        "P7L_review_reason": "" if tail_rule.get("P7L_mask") is not None else "P7L_mask_rule_unreliable",
        **p5_qc,
        **p6_qc,
        **p7v_qc,
    }

    review_reasons.extend(
        reason
        for reason in (
            p5_qc.get("P5_review_reason", ""),
            p6_qc.get("P6_review_reason", ""),
            body_depth_qc.get("body_depth_review_reason", ""),
            peduncle_depth_qc.get("peduncle_depth_review_reason", ""),
            axis_qc.get("axis_qc_reason", ""),
            p7v_qc.get("p7v_review_reason", ""),
        )
        if reason
    )
    clean_reasons = _unique_reasons(review_reasons)

    return {
        "corrected_keypoints": {key: [float(value[0]), float(value[1])] for key, value in corrected.items()},
        "mask_suggestions": {key: [float(value[0]), float(value[1])] for key, value in suggestions.items()},
        "geometric_suggestions": {key: [float(value[0]), float(value[1])] for key, value in geometric_suggestions.items()},
        "point_sources": point_sources,
        "tail_geometry_qc": tail_geometry_qc,
        "body_depth_qc": body_depth_qc,
        "peduncle_depth_qc": peduncle_depth_qc,
        "axis_qc": axis_qc,
        "tail_detail_mask": tail_detail_mask,
        "flagged_keypoints": flagged,
        "needs_review": bool(clean_reasons),
        "review_reason": ";".join(clean_reasons),
    }


def validate_p6_geometry(
    P5: tuple[float, float] | list[float] | None,
    P6: tuple[float, float] | list[float] | None,
    P7U: tuple[float, float] | list[float] | None,
    P7L: tuple[float, float] | list[float] | None,
    P7V: tuple[float, float] | list[float] | None = None,
    tail_axis: tuple[float, float] | list[float] | None = None,
    fish_mask: np.ndarray | None = None,
    tail_mask: np.ndarray | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate P6 caudal fork geometry without mutating annotations.

    This v0.6.5 helper is intentionally conservative.  It checks whether P6
    sits between P5 and the caudal lobe tips along the tail axis, is closer to
    both lobe tips than P5 is, and is not strongly biased toward one lobe.  A
    caller may pass a mask fork suggestion in ``config["P6_mask_fork_rule"]`` or
    ``config["mask_fork_rule"]``; if selected P6 fails and the suggestion is
    usable, the returned payload reports it as a fallback candidate.  Confirmed
    labels are not changed here.
    """

    cfg = dict(config or {})
    p5 = _as_xy(P5)
    p6 = _as_xy(P6)
    p7u = _as_xy(P7U)
    p7l = _as_xy(P7L)
    p7v = _as_xy(P7V)
    axis = _as_xy(tail_axis)
    if axis is None and p5 is not None and p7u is not None and p7l is not None:
        mid = ((p7u[0] + p7l[0]) / 2.0, (p7u[1] + p7l[1]) / 2.0)
        axis = _unit(p5, mid)
    if axis is None and p5 is not None and p7v is not None:
        axis = _unit(p5, p7v)

    reasons: list[str] = []
    values: dict[str, Any] = {
        "P6_geometry_qc_pass": False,
        "P6_projection_between_P5_and_tailtips": False,
        "P6_projection_ok": False,
        "P6_distance_to_P7U_ok": False,
        "P6_distance_to_P7L_ok": False,
        "P6_tail_leaf_balance_ok": False,
        "P6_fork_gap_distance_ok": "",
        "P6_fallback_used": False,
        "P6_fallback_x": "",
        "P6_fallback_y": "",
        "P6_review_reason": "",
        "P6_projection_px": "",
        "P7U_projection_px": "",
        "P7L_projection_px": "",
        "P6_to_P7U_px": "",
        "P6_to_P7L_px": "",
        "P5_to_P7U_px": "",
        "P5_to_P7L_px": "",
        "P6_leaf_distance_ratio": "",
    }
    if p5 is None or p6 is None or p7u is None or p7l is None or axis is None:
        values["P6_review_reason"] = "missing_tail_points_or_tail_axis"
        return values

    proj_p6 = _dot(_sub(p6, p5), axis)
    proj_u = _dot(_sub(p7u, p5), axis)
    proj_l = _dot(_sub(p7l, p5), axis)
    max_tip_proj = max(proj_u, proj_l)
    values.update(
        {
            "P6_projection_px": float(proj_p6),
            "P7U_projection_px": float(proj_u),
            "P7L_projection_px": float(proj_l),
        }
    )
    if proj_p6 <= 0:
        reasons.append("P6_not_after_P5")
    if proj_p6 >= max_tip_proj:
        reasons.append("P6_not_before_tail_tips")
    projection_ok = proj_p6 > 0 and proj_p6 < max_tip_proj
    values["P6_projection_between_P5_and_tailtips"] = projection_ok
    values["P6_projection_ok"] = projection_ok

    p6_u = _distance(p6, p7u)
    p6_l = _distance(p6, p7l)
    p5_u = _distance(p5, p7u)
    p5_l = _distance(p5, p7l)
    values.update(
        {
            "P6_to_P7U_px": float(p6_u),
            "P6_to_P7L_px": float(p6_l),
            "P5_to_P7U_px": float(p5_u),
            "P5_to_P7L_px": float(p5_l),
        }
    )
    p6_u_ok = p6_u < p5_u
    p6_l_ok = p6_l < p5_l
    values["P6_distance_to_P7U_ok"] = p6_u_ok
    values["P6_distance_to_P7L_ok"] = p6_l_ok
    if not p6_u_ok:
        reasons.append("P6_not_closer_to_P7U_than_P5")
    if not p6_l_ok:
        reasons.append("P6_not_closer_to_P7L_than_P5")

    ratio = p6_u / max(p6_l, 1e-6)
    values["P6_leaf_distance_ratio"] = float(ratio)
    lo = float(cfg.get("leaf_balance_min", 0.4))
    hi = float(cfg.get("leaf_balance_max", 2.5))
    balance_ok = lo <= ratio <= hi
    values["P6_tail_leaf_balance_ok"] = balance_ok
    if not balance_ok:
        reasons.append("P6_tail_leaf_distance_unbalanced")

    mask_to_check = tail_mask if tail_mask is not None else fish_mask
    if mask_to_check is not None and p6 is not None:
        # P6 is a gap/notch point and may lie on boundary or just outside fish
        # mask; being very far from any tail mask pixel is still suspicious.
        height, width = mask_to_check.shape[:2]
        x = int(round(p6[0]))
        y = int(round(p6[1]))
        radius = int(cfg.get("mask_near_radius_px", 10))
        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        if x < 0 or y < 0 or x >= width or y >= height or int(np.count_nonzero(mask_to_check[y0:y1, x0:x1])) == 0:
            reasons.append("P6_far_from_tail_mask")

    gap_point = _as_xy(cfg.get("P6_geometric") or cfg.get("P6_gap_geometric") or cfg.get("P6_gap_suggestion"))
    if gap_point is not None:
        gap_dist = _distance(p6, gap_point)
        values["P6_current_to_geometric_distance_px"] = float(gap_dist)
        threshold_px = float(cfg.get("P6_gap_distance_threshold_px", 80.0))
        gap_ok = gap_dist <= threshold_px
        values["P6_fork_gap_distance_ok"] = gap_ok
        if not gap_ok:
            reasons.append("P6_far_from_tail_fork_gap_geometric")

    fallback = _as_xy(cfg.get("P6_mask_fork_rule") or cfg.get("mask_fork_rule") or cfg.get("P6_fallback") or gap_point)
    if reasons and fallback is not None:
        fallback_proj = _dot(_sub(fallback, p5), axis)
        fallback_u = _distance(fallback, p7u)
        fallback_l = _distance(fallback, p7l)
        fallback_ratio = fallback_u / max(fallback_l, 1e-6)
        fallback_ok = (
            fallback_proj > 0
            and fallback_proj < max_tip_proj
            and fallback_u < p5_u
            and fallback_l < p5_l
            and lo <= fallback_ratio <= hi
        )
        if fallback_ok:
            values["P6_fallback_used"] = True
            values["P6_fallback_x"] = float(fallback[0])
            values["P6_fallback_y"] = float(fallback[1])
            reasons.append("P6_fallback_to_mask_fork_rule_available")

    values["P6_geometry_qc_pass"] = not reasons
    values["P6_review_reason"] = ";".join(_unique_reasons(reasons))
    return values
