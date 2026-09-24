"""Fish segmentation helpers for blue-board rabbitfish images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .config import A3_V2_CHARUCO_PLUMB_CONFIG
from .image_utils import ensure_rgb


def _board_fish_box_px(shape: tuple[int, int, int], padding_mm: float = 5.0) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    px_per_mm = width / float(A3_V2_CHARUCO_PLUMB_CONFIG["board_width_mm"])
    x0_mm, y0_mm, x1_mm, y1_mm = A3_V2_CHARUCO_PLUMB_CONFIG["plumb_lines"]["fish_box_mm"]
    return (
        max(0, int(round((float(x0_mm) - padding_mm) * px_per_mm))),
        max(0, int(round((float(y0_mm) - padding_mm) * px_per_mm))),
        min(width, int(round((float(x1_mm) + padding_mm) * px_per_mm))),
        min(height, int(round((float(y1_mm) + padding_mm) * px_per_mm))),
    )


def _fill_holes(binary: np.ndarray) -> np.ndarray:
    """Fill holes in a uint8 binary mask."""
    h, w = binary.shape
    flood = binary.copy()
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(binary, holes)


def _largest_component(binary: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return np.zeros_like(binary), {"area": 0, "component_count": 0, "bbox": (0, 0, 0, 0), "centroid": (0.0, 0.0)}

    candidates = []
    for label_id in range(1, count):
        x, y, w, h, area = stats[label_id]
        if area <= 0:
            continue
        aspect = w / max(1, h)
        candidates.append((area, aspect, label_id, x, y, w, h))
    if not candidates:
        return np.zeros_like(binary), {"area": 0, "component_count": count - 1, "bbox": (0, 0, 0, 0), "centroid": (0.0, 0.0)}

    # Fish is by far the largest non-blue object in the clean placement area.
    _area, _aspect, label_id, x, y, w, h = max(candidates, key=lambda item: item[0])
    component = np.where(labels == label_id, 255, 0).astype(np.uint8)
    return component, {
        "area": int(_area),
        "component_count": count - 1,
        "bbox": (int(x), int(y), int(x + w), int(y + h)),
        "centroid": tuple(float(v) for v in centroids[label_id]),
    }


def segment_fish_from_blue_board(warped_image: np.ndarray | Image.Image) -> tuple[np.ndarray, tuple[int, int, int, int], dict[str, Any]]:
    """Segment a fish from a rectified blue calibration board.

    Returns a full-size binary mask, fish bbox in warped-image coordinates, and
    a quality dictionary. The implementation intentionally restricts component
    search to the known fish-placement area so board markers, rulers, text, and
    numbering fields do not dominate the segmentation.
    """
    rgb = ensure_rgb(warped_image)
    height, width = rgb.shape[:2]
    x0, y0, x1, y1 = _board_fish_box_px(rgb.shape, padding_mm=5.0)
    roi = rgb[y0:y1, x0:x1]
    if roi.size == 0:
        quality = {
            "segmentation_success": False,
            "segmentation_quality": "failed_empty_roi",
            "mask_area_px": 0,
            "fish_bbox_xyxy": (0, 0, 0, 0),
        }
        return np.zeros((height, width), dtype=np.uint8), (0, 0, 0, 0), quality

    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    blue_background = ((h >= 82) & (h <= 132) & (s >= 35) & (v >= 35)).astype(np.uint8) * 255
    candidate = cv2.bitwise_not(blue_background)

    # Remove thin box lines/text specks, then reconnect fish edges and fill holes.
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, open_kernel, iterations=1)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    candidate = _fill_holes(candidate)

    component, info = _largest_component(candidate)
    component = cv2.morphologyEx(component, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    component = _fill_holes(component)
    component, info = _largest_component(component)

    full_mask = np.zeros((height, width), dtype=np.uint8)
    full_mask[y0:y1, x0:x1] = component
    area = int(np.count_nonzero(component))
    bx0, by0, bx1, by1 = info["bbox"]
    bbox = (x0 + bx0, y0 + by0, x0 + bx1, y0 + by1)
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    roi_area = max(1, roi.shape[0] * roi.shape[1])
    area_ratio = area / roi_area
    touches_roi_edge = bx0 <= 3 or by0 <= 3 or bx1 >= roi.shape[1] - 3 or by1 >= roi.shape[0] - 3

    success = area > 20_000 and bbox_width > 250 and bbox_height > 80
    warnings: list[str] = []
    if not success:
        segmentation_quality = "failed"
    else:
        if touches_roi_edge:
            warnings.append("mask_touches_roi_edge")
        if area_ratio < 0.02:
            warnings.append("small_mask_area")
        if bbox_width / max(1, bbox_height) < 1.5:
            warnings.append("unexpected_aspect_ratio")
        segmentation_quality = "warning:" + ";".join(warnings) if warnings else "ok"

    quality = {
        "segmentation_success": bool(success),
        "segmentation_quality": segmentation_quality,
        "mask_area_px": area,
        "fish_bbox_xyxy": bbox,
        "search_roi_xyxy": (x0, y0, x1, y1),
        "area_ratio": area_ratio,
        "component_count": info.get("component_count", 0),
    }
    if not success:
        full_mask[:, :] = 0
        bbox = (0, 0, 0, 0)
        quality["fish_bbox_xyxy"] = bbox
    return full_mask, bbox, quality


def padded_bbox(
    bbox_xyxy: tuple[int, int, int, int],
    image_shape: tuple[int, int] | tuple[int, int, int],
    padding_ratio: float = 0.08,
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x0, y0, x1, y1 = bbox_xyxy
    pad_x = int(round((x1 - x0) * padding_ratio))
    pad_y = int(round((y1 - y0) * padding_ratio))
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(width, x1 + pad_x),
        min(height, y1 + pad_y),
    )


def save_mask_png(mask: np.ndarray, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(mask > 0, 255, 0).astype(np.uint8)).save(output_path)
    return output_path


def save_segmentation_preview(
    image: np.ndarray | Image.Image,
    mask: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    quality: dict[str, Any],
    output_path: Path,
) -> Path:
    rgb = ensure_rgb(image)
    preview = rgb.copy()
    overlay = preview.copy()
    overlay[mask > 0] = (255, 80, 0)
    preview = cv2.addWeighted(overlay, 0.35, preview, 0.65, 0)
    x0, y0, x1, y1 = bbox_xyxy
    if x1 > x0 and y1 > y0:
        cv2.rectangle(preview, (x0, y0), (x1, y1), (255, 220, 0), 8)
    text = f"{quality.get('segmentation_quality', 'unknown')} area={quality.get('mask_area_px', 0)}"
    cv2.putText(preview, text, (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 6, cv2.LINE_AA)
    cv2.putText(preview, text, (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(preview).save(output_path)
    return output_path

