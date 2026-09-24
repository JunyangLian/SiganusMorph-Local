"""Plumb-line style straight-line quality checks."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .image_utils import ensure_rgb


def detect_reference_line_segments(image: np.ndarray) -> dict[str, Any]:
    """Detect long horizontal/vertical line segments for straightness QC.

    This is not a full plumb-line lens-calibration solver yet. It provides a
    repeatable quality signal: after correction, printed straight reference
    lines should remain close to horizontal/vertical and should appear as long
    straight Hough segments.
    """
    rgb = ensure_rgb(image)
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    min_line_length = int(min(width, height) * 0.18)
    max_gap = int(min(width, height) * 0.015)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=min_line_length,
        maxLineGap=max_gap,
    )

    horizontal: list[dict[str, float]] = []
    vertical: list[dict[str, float]] = []
    if lines is not None:
        for line in lines.reshape(-1, 4):
            x1, y1, x2, y2 = [float(value) for value in line]
            dx = x2 - x1
            dy = y2 - y1
            length = float(np.hypot(dx, dy))
            if length <= 0:
                continue
            angle = float(np.degrees(np.arctan2(dy, dx)))
            abs_angle = abs(angle)
            segment = {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "length_px": length,
                "angle_deg": angle,
            }
            if abs_angle <= 8 or abs(abs_angle - 180) <= 8:
                segment["deviation_deg"] = min(abs_angle, abs(abs_angle - 180))
                horizontal.append(segment)
            elif abs(abs_angle - 90) <= 8:
                segment["deviation_deg"] = abs(abs_angle - 90)
                vertical.append(segment)

    return {
        "horizontal": sorted(horizontal, key=lambda item: item["length_px"], reverse=True),
        "vertical": sorted(vertical, key=lambda item: item["length_px"], reverse=True),
    }


def calculate_plumbline_quality(image: np.ndarray) -> dict[str, Any]:
    """Summarize straight-line quality after image correction."""
    segments = detect_reference_line_segments(image)
    top_h = segments["horizontal"][:8]
    top_v = segments["vertical"][:8]
    deviations = [item["deviation_deg"] for item in top_h + top_v]
    if deviations:
        median_dev = float(np.median(deviations))
        max_dev = float(np.max(deviations))
    else:
        median_dev = float("nan")
        max_dev = float("nan")

    return {
        "horizontal_count": len(segments["horizontal"]),
        "vertical_count": len(segments["vertical"]),
        "median_angle_deviation_deg": median_dev,
        "max_angle_deviation_deg": max_dev,
        "needs_review": (len(top_h) < 2 or len(top_v) < 2 or max_dev > 3.0) if deviations else True,
        "top_horizontal": top_h,
        "top_vertical": top_v,
    }
