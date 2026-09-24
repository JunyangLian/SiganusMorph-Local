"""Scale calibration helpers."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .image_utils import ensure_rgb
from .measurements import PointLike, euclidean_distance


def calculate_scale_from_two_points(
    point1: PointLike,
    point2: PointLike,
    real_distance_mm: float,
) -> float:
    """Calculate millimetres per pixel from two clicked ruler points."""
    if real_distance_mm <= 0:
        raise ValueError("real_distance_mm must be greater than 0.")

    pixel_distance = euclidean_distance(point1, point2)
    if pixel_distance <= 0:
        raise ValueError("The two calibration points must not be identical.")

    return real_distance_mm / pixel_distance


def estimate_mm_per_pixel_from_board_width(
    image_width_px: int,
    board_width_mm: float,
) -> float:
    """Estimate scale after perspective warping to the known board width."""
    if image_width_px <= 0:
        raise ValueError("image_width_px must be greater than 0.")
    if board_width_mm <= 0:
        raise ValueError("board_width_mm must be greater than 0.")
    return board_width_mm / image_width_px


def estimate_scale_from_ruler_ticks(
    image: np.ndarray,
    real_length_mm: float = 350.0,
    expected_tick_count: int = 36,
    threshold_value: int = 90,
) -> dict[str, Any]:
    """Estimate scale from the 0-35 cm ruler after board rectification.

    The A3 board has centimetre ticks from 0 to 35 cm on the top ruler.
    After ArUco rectification, this function looks for the long vertical
    centimetre ticks in the ruler band and uses the first-to-last span as
    350 mm. This is intentionally independent of the paper edge because the
    ArUco markers are inset from the printable area.
    """
    if real_length_mm <= 0:
        raise ValueError("real_length_mm must be greater than 0.")
    if expected_tick_count < 2:
        raise ValueError("expected_tick_count must be at least 2.")

    rgb = ensure_rgb(image)
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # The printed 0-35 cm ruler sits in the upper quarter of the rectified A3 board.
    y1 = int(round(height * 0.145))
    y2 = int(round(height * 0.265))
    x1 = int(round(width * 0.03))
    x2 = int(round(width * 0.97))
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        return {"success": False, "message": "Ruler search region is empty."}

    max_tick_width = max(8, int(round(width * 0.003)))

    def find_column_candidates(local_y_offset: int) -> list[tuple[float, int, int, int]]:
        tick_crop = crop[local_y_offset:, :]
        mask = tick_crop < threshold_value
        col_counts = mask.sum(axis=0)
        if col_counts.size == 0 or int(col_counts.max()) == 0:
            return []

        col_threshold = max(25, int(round(float(col_counts.max()) * 0.70)))
        strong_columns = np.where(col_counts >= col_threshold)[0]
        runs: list[tuple[int, int]] = []
        if len(strong_columns):
            start = previous = int(strong_columns[0])
            for column in strong_columns[1:]:
                column = int(column)
                if column <= previous + 1:
                    previous = column
                else:
                    runs.append((start, previous))
                    start = previous = column
            runs.append((start, previous))

        found: list[tuple[float, int, int, int]] = []
        for start, end in runs:
            component_width = end - start + 1
            if not (1 <= component_width <= max_tick_width):
                continue
            weights = col_counts[start : end + 1].astype(float)
            xs = np.arange(start, end + 1, dtype=float)
            centroid_x = float(np.average(xs, weights=weights)) if weights.sum() else float((start + end) / 2)
            found.append((float(x1 + centroid_x), int(component_width), int(weights.max()), int(weights.sum())))
        return found

    # First search in the lower tick area. If the printed ruler uses separated
    # checker/tick lines, fall back to the whole band.
    local_y_offset = int(round(crop.shape[0] * 0.35))
    candidates = find_column_candidates(local_y_offset)
    if len(candidates) < expected_tick_count:
        local_y_offset = 0
        candidates = find_column_candidates(local_y_offset)

    if not candidates:
        return {
            "success": False,
            "message": "No dark ruler columns found.",
            "tick_count": 0,
            "search_region": [x1, y1 + local_y_offset, x2, y2],
        }

    if len(candidates) < expected_tick_count:
        return {
            "success": False,
            "message": f"Only found {len(candidates)} ruler tick candidates.",
            "tick_count": len(candidates),
            "search_region": [x1, y1 + local_y_offset, x2, y2],
        }

    candidates.sort(key=lambda item: item[0])
    min_spacing = width * 0.015
    max_spacing = width * 0.040
    if len(candidates) <= expected_tick_count + 5:
        all_centers = np.asarray([item[0] for item in candidates], dtype=float)
        span_px = float(all_centers[-1] - all_centers[0])
        mean_spacing = span_px / float(expected_tick_count - 1)
        if min_spacing <= mean_spacing <= max_spacing:
            expected_centers = all_centers[0] + np.arange(expected_tick_count, dtype=float) * mean_spacing
            nearest_residuals = np.asarray(
                [float(np.min(np.abs(all_centers - center))) for center in expected_centers],
                dtype=float,
            )
            best = {
                "score": float(np.mean(nearest_residuals) / mean_spacing),
                "centers": np.asarray([all_centers[0], all_centers[-1]], dtype=float),
                "mean_spacing_px": mean_spacing,
                "spacing_cv": float(np.std(nearest_residuals) / mean_spacing),
                "span_px": span_px,
            }
        else:
            best = None
    else:
        best = None

    if best is None:
        for start_index in range(0, len(candidates) - expected_tick_count + 1):
            group = candidates[start_index : start_index + expected_tick_count]
            centers = np.asarray([item[0] for item in group], dtype=float)
            diffs = np.diff(centers)
            mean_spacing = float(np.mean(diffs))
            if mean_spacing < min_spacing or mean_spacing > max_spacing:
                continue
            spacing_cv = float(np.std(diffs) / mean_spacing) if mean_spacing else 999.0
            span_px = float(centers[-1] - centers[0])
            expected_span = mean_spacing * (expected_tick_count - 1)
            span_error = abs(span_px - expected_span) / max(expected_span, 1.0)
            score = spacing_cv + span_error
            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "centers": centers,
                    "mean_spacing_px": mean_spacing,
                    "spacing_cv": spacing_cv,
                    "span_px": span_px,
                }

    if best is None:
        return {
            "success": False,
            "message": "Found tick candidates, but not a stable 0-35 cm sequence.",
            "tick_count": len(candidates),
            "search_region": [x1, y1 + local_y_offset, x2, y2],
        }

    centers = best["centers"]
    mm_per_pixel = real_length_mm / best["span_px"]
    endpoint_y = float(y1 + local_y_offset + y2) / 2.0
    return {
        "success": True,
        "message": "Detected 0-35 cm ruler ticks.",
        "mm_per_pixel": float(mm_per_pixel),
        "real_length_mm": float(real_length_mm),
        "pixel_span": float(best["span_px"]),
        "tick_count": expected_tick_count,
        "candidate_count": len(candidates),
        "mean_spacing_px": float(best["mean_spacing_px"]),
        "spacing_cv": float(best["spacing_cv"]),
        "endpoints": [
            {"x": float(centers[0]), "y": endpoint_y},
            {"x": float(centers[-1]), "y": endpoint_y},
        ],
        "search_region": [x1, y1 + local_y_offset, x2, y2],
    }
