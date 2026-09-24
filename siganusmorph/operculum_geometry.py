"""Operculum posterior-edge suggestions for enhanced preannotation."""

from __future__ import annotations

from typing import Any, Mapping
from math import hypot

import cv2
import numpy as np

from .image_utils import ensure_rgb


P1 = "P1_snout_tip"
P3 = "P3_operculum_posterior"


def _point(points: Mapping[str, Any], key: str) -> tuple[float, float] | None:
    value = points.get(key)
    if isinstance(value, Mapping) and "x" in value and "y" in value:
        return float(value["x"]), float(value["y"])
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return None


def _mask_bbox(mask: np.ndarray | None) -> tuple[int, int, int, int] | None:
    if mask is None or mask.size == 0 or int(np.count_nonzero(mask)) == 0:
        return None
    ys, xs = np.where(mask > 0)
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def estimate_operculum_posterior_edge(
    warped_image: np.ndarray,
    fish_mask: np.ndarray | None,
    keypoints: Mapping[str, Any],
    model_keypoints_raw: Mapping[str, Any],
) -> dict[str, Any]:
    """Suggest P3 by snapping near model P3 to a strong local edge."""
    rgb = ensure_rgb(warped_image)
    bbox = _mask_bbox(fish_mask)
    model_p3 = _point(model_keypoints_raw, P3) or _point(keypoints, P3)
    p1 = _point(keypoints, P1) or _point(model_keypoints_raw, P1)
    result: dict[str, Any] = {
        "P3_edge_suggestion": None,
        "P3_edge_quality": "failed",
        "P3_edge_strength": "",
        "P3_edge_method": "local_scharr_edge",
        "P3_review_reason": "",
    }
    if bbox is None or model_p3 is None:
        result["P3_review_reason"] = "missing_mask_or_model_P3"
        return result
    bx0, by0, bx1, by1 = bbox
    bw, bh = max(1, bx1 - bx0), max(1, by1 - by0)
    half_w = max(45, int(0.07 * bw))
    half_h = max(70, int(0.16 * bh))
    x0 = max(0, int(round(model_p3[0] - half_w)))
    x1 = min(rgb.shape[1], int(round(model_p3[0] + half_w)))
    y0 = max(0, int(round(model_p3[1] - half_h)))
    y1 = min(rgb.shape[0], int(round(model_p3[1] + half_h)))
    if x1 <= x0 or y1 <= y0:
        result["P3_review_reason"] = "empty_P3_roi"
        return result

    roi = rgb[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    grad_x = cv2.Scharr(clahe, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(clahe, cv2.CV_32F, 0, 1)
    strength = np.abs(grad_x) + 0.25 * np.abs(grad_y)
    if fish_mask is not None:
        local_mask = fish_mask[y0:y1, x0:x1] > 0
        strength = np.where(local_mask, strength, 0)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist = np.sqrt((xx - model_p3[0]) ** 2 + (yy - model_p3[1]) ** 2)
    score = strength - 1.2 * dist
    if p1 is not None:
        score = np.where(xx > p1[0] + 0.08 * bw, score, -1e9)
    score = np.where(xx < bx0 + 0.43 * bw, score, -1e9)
    if float(np.max(score)) <= 0:
        result["P3_review_reason"] = "no_local_edge_candidate"
        return result
    iy, ix = np.unravel_index(int(np.argmax(score)), score.shape)
    point = (float(x0 + ix), float(y0 + iy))
    edge_strength = float(strength[iy, ix])
    edge_threshold = float(np.percentile(strength[strength > 0], 85)) if np.any(strength > 0) else float("inf")
    quality = "good" if edge_strength > edge_threshold else "warning"
    result.update(
        {
            "P3_edge_suggestion": point,
            "P3_edge_quality": quality,
            "P3_edge_strength": edge_strength,
            "P3_review_reason": "",
        }
    )
    return result


def validate_p3_operculum_position(
    P1: Any = None,
    P2: Any = None,
    P3: Any = None,
    C1: Any = None,
    P4: Any = None,
    body_midline_axis: Any = None,
    fish_mask: np.ndarray | None = None,
    image: np.ndarray | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """QC P3 against the head-region geometry and optional edge suggestion.

    This is deliberately conservative: it does not replace P3, it only returns
    a warning/suggestion when the current point appears too far posterior or
    away from a plausible operculum edge.
    """
    del body_midline_axis
    cfg = dict(config or {})
    mm_per_pixel = float(cfg.get("mm_per_pixel", 0.1))
    p1 = _point({"p": P1}, "p")
    p2 = _point({"p": P2}, "p")
    p3 = _point({"p": P3}, "p")
    c1 = _point({"p": C1}, "p")
    p4 = _point({"p": P4}, "p")
    reasons: list[str] = []
    edge_suggestion: tuple[float, float] | None = None
    edge_quality = "failed"
    edge_distance_mm: float | None = None
    if p1 is None or p3 is None or p4 is None:
        return {
            "P3_qc_pass": False,
            "P3_needs_review": True,
            "P3_review_reason": "missing_P1_P3_or_P4",
            "P3_operculum_edge_suggestion": None,
            "P3_current_to_suggestion_distance_mm": None,
        }
    axis = (p4[0] - p1[0], p4[1] - p1[1])
    axis_len2 = axis[0] * axis[0] + axis[1] * axis[1]
    if axis_len2 <= 1e-9:
        reasons.append("P1_P4_axis_invalid")
        ratio = None
    else:
        ratio = ((p3[0] - p1[0]) * axis[0] + (p3[1] - p1[1]) * axis[1]) / axis_len2
        if ratio < float(cfg.get("P3_min_P1P4_ratio", 0.08)):
            reasons.append("P3_too_close_to_snout")
        if ratio > float(cfg.get("P3_max_P1P4_ratio", 0.40)):
            reasons.append("P3_too_far_posterior_possible_body_pattern")
    if p2 is not None and axis_len2 > 1e-9:
        p2_ratio = ((p2[0] - p1[0]) * axis[0] + (p2[1] - p1[1]) * axis[1]) / axis_len2
        if ratio is not None and ratio <= p2_ratio:
            reasons.append("P3_not_posterior_to_P2")
    if c1 is not None:
        p1_p3 = hypot(p3[0] - p1[0], p3[1] - p1[1])
        p1_c1 = hypot(c1[0] - p1[0], c1[1] - p1[1])
        if (
            p1_c1 > 1e-9
            and ratio is not None
            and ratio > float(cfg.get("P3_C1_warning_min_P1P4_ratio", 0.35))
            and p1_p3 > p1_c1 * float(cfg.get("P3_to_C1_distance_ratio_max", 1.80))
        ):
            reasons.append("P3_too_close_to_trunk_axis_region")
    if fish_mask is not None:
        h, w = fish_mask.shape[:2]
        x, y = int(round(p3[0])), int(round(p3[1]))
        if x < 0 or x >= w or y < 0 or y >= h or not bool(fish_mask[y, x]):
            reasons.append("P3_outside_fish_mask")
    if image is not None:
        keypoints = {
            "P1_snout_tip": {"x": p1[0], "y": p1[1]},
            "P3_operculum_posterior": {"x": p3[0], "y": p3[1]},
        }
        if p2 is not None:
            keypoints["P2_eye_front"] = {"x": p2[0], "y": p2[1]}
        edge = estimate_operculum_posterior_edge(image, fish_mask, keypoints, keypoints)
        suggestion = edge.get("P3_edge_suggestion")
        if isinstance(suggestion, (list, tuple)) and len(suggestion) >= 2:
            edge_suggestion = (float(suggestion[0]), float(suggestion[1]))
            edge_quality = str(edge.get("P3_edge_quality", "warning"))
            edge_distance_mm = hypot(edge_suggestion[0] - p3[0], edge_suggestion[1] - p3[1]) * mm_per_pixel
            if (
                edge_quality == "good"
                and ratio is not None
                and ratio > float(cfg.get("P3_edge_warning_min_P1P4_ratio", 0.32))
                and edge_distance_mm > float(cfg.get("P3_edge_distance_threshold_mm", 10.0))
            ):
                reasons.append("P3_far_from_operculum_edge_suggestion")
        elif edge.get("P3_review_reason"):
            reasons.append(f"P3_edge_suggestion_unavailable:{edge.get('P3_review_reason')}")
    review_reason = ";".join(dict.fromkeys(reasons))
    return {
        "P3_qc_pass": not reasons,
        "P3_needs_review": bool(reasons),
        "P3_review_reason": review_reason,
        "P3_operculum_edge_suggestion": list(edge_suggestion) if edge_suggestion is not None else None,
        "P3_operculum_edge_quality": edge_quality,
        "P3_current_to_suggestion_distance_mm": edge_distance_mm,
        "P3_P1P4_projection_ratio": ratio,
    }
