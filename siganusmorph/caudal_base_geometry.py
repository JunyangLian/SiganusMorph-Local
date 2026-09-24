"""Caudal-base transition suggestions for enhanced preannotation."""

from __future__ import annotations

from typing import Any, Mapping

import cv2
import numpy as np


P4 = "P4_peduncle_start_midpoint"
P5 = "P5_caudal_base_midpoint"
P7U = "P7U_caudal_fin_upper_tip"
P7L = "P7L_caudal_fin_lower_tip"
C4 = "C4_peduncle_axis_point"


def _point(points: Mapping[str, Any], key: str) -> tuple[float, float] | None:
    value = points.get(key)
    if isinstance(value, Mapping) and "x" in value and "y" in value:
        return float(value["x"]), float(value["y"])
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return None


def _unit(a: tuple[float, float] | None, b: tuple[float, float] | None) -> tuple[float, float] | None:
    if a is None or b is None:
        return None
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = float((dx * dx + dy * dy) ** 0.5)
    if length < 1e-6:
        return None
    return dx / length, dy / length


def _sample_width(mask: np.ndarray, center: tuple[float, float], normal: tuple[float, float], max_radius: int) -> tuple[float, tuple[float, float], tuple[float, float]] | None:
    height, width = mask.shape[:2]
    hits: list[tuple[float, float, float]] = []
    for t in np.linspace(-max_radius, max_radius, max(25, max_radius * 2 + 1)):
        x = int(round(center[0] + t * normal[0]))
        y = int(round(center[1] + t * normal[1]))
        if 0 <= x < width and 0 <= y < height and mask[y, x] > 0:
            hits.append((float(t), float(x), float(y)))
    if len(hits) < 6:
        return None
    first = hits[0]
    last = hits[-1]
    return float(abs(last[0] - first[0])), (first[1], first[2]), (last[1], last[2])


def estimate_caudal_base_transition(
    fish_mask: np.ndarray | None,
    tail_detail_mask: np.ndarray | None,
    keypoints: Mapping[str, Any],
    tail_axis: tuple[float, float] | list[float] | None,
) -> dict[str, Any]:
    """Suggest P5 by finding the tail-peduncle-to-caudal-fin width transition."""
    result: dict[str, Any] = {
        "P5_transition_suggestion": None,
        "P5_transition_quality": "failed",
        "P5_transition_method": "tail_width_profile",
        "tail_width_profile": [],
        "tail_width_transition_score": "",
        "P5_review_reason": "",
    }
    if fish_mask is None or fish_mask.size == 0 or int(np.count_nonzero(fish_mask)) == 0:
        result["P5_review_reason"] = "missing_fish_mask"
        return result
    mask = cv2.medianBlur((fish_mask > 0).astype(np.uint8) * 255, 5)
    p4 = _point(keypoints, P4)
    p5 = _point(keypoints, P5)
    p7u = _point(keypoints, P7U)
    p7l = _point(keypoints, P7L)
    c4 = _point(keypoints, C4)
    if tail_axis is not None and len(tail_axis) >= 2:
        axis = (float(tail_axis[0]), float(tail_axis[1]))
    else:
        tail_mid = ((p7u[0] + p7l[0]) / 2.0, (p7u[1] + p7l[1]) / 2.0) if p7u and p7l else None
        axis = _unit(p5 or c4, tail_mid) or _unit(c4, p5)
    if axis is None or p4 is None or p5 is None:
        result["P5_review_reason"] = "missing_axis_or_P4_P5"
        return result
    normal = (-axis[1], axis[0])
    ys, xs = np.where(mask > 0)
    bbox_width = int(xs.max() - xs.min() + 1) if len(xs) else 500
    max_radius = max(50, int(0.18 * bbox_width))
    tail_mid = ((p7u[0] + p7l[0]) / 2.0, (p7u[1] + p7l[1]) / 2.0) if p7u and p7l else (p5[0] + 0.28 * bbox_width, p5[1])
    start = np.asarray(p4, dtype=float)
    end = np.asarray(tail_mid, dtype=float)
    scan = end - start
    scan_len = float(np.linalg.norm(scan))
    if scan_len < 40:
        result["P5_review_reason"] = "scan_region_too_short"
        return result
    profile: list[dict[str, float]] = []
    for s in np.linspace(0.20, 0.86, 46):
        center = (float(start[0] + s * scan[0]), float(start[1] + s * scan[1]))
        sample = _sample_width(mask, center, normal, max_radius)
        if sample is None:
            continue
        width, upper, lower = sample
        mid = ((upper[0] + lower[0]) / 2.0, (upper[1] + lower[1]) / 2.0)
        profile.append({"s": float(s), "width": float(width), "x": mid[0], "y": mid[1]})
    result["tail_width_profile"] = profile[:]
    if len(profile) < 8:
        result["P5_review_reason"] = "width_profile_too_sparse"
        return result
    widths = np.asarray([row["width"] for row in profile], dtype=float)
    smooth = np.convolve(widths, np.ones(5) / 5.0, mode="same")
    gradient = np.gradient(smooth)
    # Ignore very early peduncle fluctuations and choose the strongest expansion
    # before the far caudal-fin tips.
    valid = np.arange(len(profile))
    valid = valid[(valid > 2) & (valid < len(profile) - 3)]
    if len(valid) == 0:
        result["P5_review_reason"] = "no_valid_transition_window"
        return result
    baseline = float(np.percentile(smooth[: max(4, len(smooth) // 3)], 35))
    score = gradient[valid] + 0.015 * np.maximum(0, smooth[valid] - baseline)
    best_idx = int(valid[int(np.argmax(score))])
    point = (float(profile[best_idx]["x"]), float(profile[best_idx]["y"]))
    transition_score = float(score[int(np.argmax(score))])
    quality = "good" if transition_score > max(2.5, 0.05 * max(1.0, baseline)) else "warning"
    result.update(
        {
            "P5_transition_suggestion": point,
            "P5_transition_quality": quality,
            "tail_width_transition_score": transition_score,
            "P5_review_reason": "",
        }
    )
    return result
