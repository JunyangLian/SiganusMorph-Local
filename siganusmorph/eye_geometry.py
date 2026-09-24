"""Eye-region suggestions for enhanced preannotation."""

from __future__ import annotations

from typing import Any, Mapping

import cv2
import numpy as np

from .image_utils import ensure_rgb


P1 = "P1_snout_tip"
P2 = "P2_eye_front"
P3 = "P3_operculum_posterior"
C1 = "C1_head_axis_point"


def _point(points: Mapping[str, Any], key: str) -> tuple[float, float] | None:
    value = points.get(key)
    if isinstance(value, Mapping) and "x" in value and "y" in value:
        return float(value["x"]), float(value["y"])
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return None


def _bbox_from_mask(mask: np.ndarray | None) -> tuple[int, int, int, int] | None:
    if mask is None or mask.size == 0 or int(np.count_nonzero(mask)) == 0:
        return None
    ys, xs = np.where(mask > 0)
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _clip_roi(x0: float, y0: float, x1: float, y1: float, shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    ix0 = max(0, min(width - 1, int(round(x0))))
    iy0 = max(0, min(height - 1, int(round(y0))))
    ix1 = max(ix0 + 1, min(width, int(round(x1))))
    iy1 = max(iy0 + 1, min(height, int(round(y1))))
    return ix0, iy0, ix1, iy1


def estimate_eye_front_from_head_roi(
    warped_image: np.ndarray,
    fish_mask: np.ndarray | None,
    keypoints: Mapping[str, Any],
    model_keypoints_raw: Mapping[str, Any],
) -> dict[str, Any]:
    """Detect a dark, round eye candidate and return its left/front edge."""
    rgb = ensure_rgb(warped_image)
    fish_bbox = _bbox_from_mask(fish_mask)
    p1 = _point(keypoints, P1) or _point(model_keypoints_raw, P1)
    p3 = _point(keypoints, P3) or _point(model_keypoints_raw, P3)
    c1 = _point(keypoints, C1) or _point(model_keypoints_raw, C1)
    model_p2 = _point(model_keypoints_raw, P2) or _point(keypoints, P2)

    result: dict[str, Any] = {
        "P2_eye_suggestion": None,
        "P2_eye_quality": "failed",
        "P2_eye_bbox": None,
        "P2_eye_candidate_count": 0,
        "P2_source_suggestion": "eye_detection_rule",
        "P2_review_reason": "",
    }
    if fish_bbox is None:
        result["P2_review_reason"] = "missing_fish_mask"
        return result
    bx0, by0, bx1, by1 = fish_bbox
    bw, bh = max(1, bx1 - bx0), max(1, by1 - by0)
    if p1 is not None and p3 is not None:
        x0 = min(p1[0], p3[0]) - 0.04 * bw
        x1 = max(p1[0], p3[0]) + 0.08 * bw
        cy = c1[1] if c1 is not None else (p1[1] + p3[1]) / 2
        y0 = cy - 0.23 * bh
        y1 = cy + 0.23 * bh
    else:
        x0, x1 = bx0, bx0 + 0.32 * bw
        y0, y1 = by0 + 0.12 * bh, by0 + 0.58 * bh
    x0, y0, x1, y1 = _clip_roi(x0, y0, x1, y1, rgb.shape)
    roi = rgb[y0:y1, x0:x1]
    if roi.size == 0:
        result["P2_review_reason"] = "empty_head_roi"
        return result

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
    l_chan = lab[:, :, 0]
    # Dark eye/pupil candidates. Use adaptive percentile so it still works on
    # wet or slightly overexposed specimens.
    threshold = max(18, min(85, int(np.percentile(l_chan, 12))))
    dark = ((l_chan <= threshold) | (gray <= max(25, threshold - 8))).astype(np.uint8) * 255
    if fish_mask is not None and fish_mask.size:
        local_mask = fish_mask[y0:y1, x0:x1]
        dark = cv2.bitwise_and(dark, (local_mask > 0).astype(np.uint8) * 255)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, connectivity=8)

    candidates: list[tuple[float, int, tuple[float, float], tuple[int, int, int, int]]] = []
    area_min = max(12, int(0.00008 * bw * bh))
    area_max = max(area_min + 1, int(0.012 * bw * bh))
    for label_id in range(1, count):
        x, y, w, h, area = stats[label_id]
        if area < area_min or area > area_max or w <= 1 or h <= 1:
            continue
        aspect = w / max(1, h)
        if aspect < 0.35 or aspect > 2.6:
            continue
        fill = area / max(1, w * h)
        if fill < 0.18:
            continue
        cx, cy = centroids[label_id]
        abs_center = (float(x0 + cx), float(y0 + cy))
        if p1 is not None and abs_center[0] <= p1[0]:
            continue
        if p3 is not None and abs_center[0] >= p3[0] + 0.05 * bw:
            continue
        distance_to_model = 0.0
        if model_p2 is not None:
            distance_to_model = ((abs_center[0] - model_p2[0]) ** 2 + (abs_center[1] - model_p2[1]) ** 2) ** 0.5
        roundness = 1.0 - min(1.0, abs(1.0 - aspect))
        darkness = 255.0 - float(np.median(gray[labels == label_id]))
        score = 2.5 * roundness + 0.01 * darkness + 0.0008 * area - 0.006 * distance_to_model
        candidates.append((score, int(area), abs_center, (int(x0 + x), int(y0 + y), int(x0 + x + w), int(y0 + y + h))))

    result["P2_eye_candidate_count"] = len(candidates)
    if not candidates:
        result["P2_review_reason"] = "no_eye_like_dark_component"
        return result
    candidates.sort(key=lambda item: item[0], reverse=True)
    _score, area, _center, bbox = candidates[0]
    suggestion = (float(bbox[0]), float((bbox[1] + bbox[3]) / 2.0))
    quality = "good" if len(candidates) <= 5 and area >= area_min * 1.3 else "warning"
    result.update(
        {
            "P2_eye_suggestion": suggestion,
            "P2_eye_quality": quality,
            "P2_eye_bbox": list(bbox),
            "P2_review_reason": "",
        }
    )
    return result
