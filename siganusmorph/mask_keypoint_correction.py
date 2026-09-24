"""Mask-aware keypoint post-processing helpers."""

from __future__ import annotations

import math
from typing import Any, Mapping

import cv2
import numpy as np


P1_KEY = "P1_snout_tip"


def _leftmost_mask_point(mask: np.ndarray, model_y: float | None = None) -> tuple[float, float] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    min_x = int(xs.min())
    candidate_ys = ys[xs == min_x]
    if len(candidate_ys) == 0:
        return None
    if model_y is not None:
        best_y = float(candidate_ys[np.argmin(np.abs(candidate_ys.astype(float) - float(model_y)))])
    else:
        best_y = float(np.median(candidate_ys))
    return float(min_x), best_y


def _point_inside_mask(mask: np.ndarray, point: tuple[float, float]) -> bool:
    x, y = int(round(point[0])), int(round(point[1]))
    return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and bool(mask[y, x] > 0)


def correct_keypoints_with_mask(
    keypoints: Mapping[str, tuple[float, float] | list[float]],
    fish_mask: np.ndarray,
    fish_bbox: tuple[int, int, int, int],
    confidences: Mapping[str, float] | None = None,
    force_p1_mask: bool = False,
) -> tuple[dict[str, tuple[float, float]], dict[str, Any]]:
    """Correct P1 with the leftmost fish-mask boundary and log simple QC."""
    corrected = {key: (float(value[0]), float(value[1])) for key, value in keypoints.items()}
    confidences = confidences or {}
    log: dict[str, Any] = {
        "p1_source": "model",
        "p1_correction_applied": False,
        "p1_model_x": "",
        "p1_model_y": "",
        "p1_mask_x": "",
        "p1_mask_y": "",
        "p1_distance_px": "",
        "threshold_px": "",
        "boundary_qc": {},
    }

    x0, _y0, x1, _y1 = fish_bbox
    bbox_width = max(1, x1 - x0)
    threshold_px = max(30.0, 0.02 * bbox_width)
    log["threshold_px"] = threshold_px

    if P1_KEY not in corrected or fish_mask.size == 0 or np.count_nonzero(fish_mask) == 0:
        log["p1_source"] = "model"
        return corrected, log

    p1_model = corrected[P1_KEY]
    p1_mask = _leftmost_mask_point(fish_mask, model_y=p1_model[1])
    log["p1_model_x"] = p1_model[0]
    log["p1_model_y"] = p1_model[1]
    if p1_mask is None:
        return corrected, log

    distance = math.hypot(p1_model[0] - p1_mask[0], p1_model[1] - p1_mask[1])
    log["p1_mask_x"] = p1_mask[0]
    log["p1_mask_y"] = p1_mask[1]
    log["p1_distance_px"] = distance
    if force_p1_mask or distance > threshold_px:
        corrected[P1_KEY] = p1_mask
        log["p1_source"] = "mask_corrected"
        log["p1_correction_applied"] = True
        log["P1"] = "snapped_to_leftmost_mask_boundary"
        if force_p1_mask:
            log["P1_rule"] = "forced_mask_left_boundary"

    boundary_keys = [
        "P8_body_depth_dorsal",
        "P9_body_depth_ventral",
        "P10_peduncle_depth_dorsal",
        "P11_peduncle_depth_ventral",
        "P7U_caudal_fin_upper_tip",
        "P7L_caudal_fin_lower_tip",
    ]
    qc: dict[str, Any] = {}
    contours, _hier = cv2.findContours((fish_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for key in boundary_keys:
        if key not in corrected:
            continue
        point = corrected[key]
        inside = _point_inside_mask(fish_mask, point)
        distance_to_mask = ""
        if contours:
            distance_to_mask = max(cv2.pointPolygonTest(contours[0], point, True), -9999.0)
        qc[key] = {
            "inside_mask": inside,
            "signed_distance_to_mask_px": distance_to_mask,
            "confidence": confidences.get(key, ""),
        }
    log["boundary_qc"] = qc
    return corrected, log
