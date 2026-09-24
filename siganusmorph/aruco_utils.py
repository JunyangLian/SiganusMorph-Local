"""ArUco marker detection and board warping."""

from __future__ import annotations

from typing import Any, Mapping

import cv2
import numpy as np

from .config import A3_V2_CHARUCO_PLUMB_CONFIG, DEFAULT_BOARD_CONFIG
from .image_utils import ensure_rgb


ARUCO_DICTIONARIES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
}


def detect_aruco_markers(
    image: np.ndarray,
    dictionary_name: str = "DICT_4X4_50",
) -> dict[str, Any]:
    """Detect ArUco markers and return IDs plus 4-corner coordinates."""
    if not hasattr(cv2, "aruco"):
        return {"markers": {}, "ids": [], "rejected_count": 0, "error": "cv2.aruco is unavailable."}

    dictionary_id = ARUCO_DICTIONARIES.get(dictionary_name, cv2.aruco.DICT_4X4_50)
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    gray = cv2.cvtColor(ensure_rgb(image), cv2.COLOR_RGB2GRAY)

    if hasattr(cv2.aruco, "ArucoDetector"):
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

    markers: dict[int, list[list[float]]] = {}
    if ids is not None:
        for marker_id, marker_corners in zip(ids.flatten().tolist(), corners):
            markers[int(marker_id)] = np.asarray(marker_corners[0], dtype=float).tolist()

    return {
        "markers": markers,
        "ids": sorted(markers),
        "rejected_count": len(rejected) if rejected is not None else 0,
        "dictionary": dictionary_name,
    }


def marker_centers(markers: Mapping[int, list[list[float]]]) -> dict[int, tuple[float, float]]:
    """Return the center point of each detected marker."""
    centers: dict[int, tuple[float, float]] = {}
    for marker_id, corners in markers.items():
        points = np.asarray(corners, dtype=float)
        center = points.mean(axis=0)
        centers[int(marker_id)] = (float(center[0]), float(center[1]))
    return centers


def select_board_config_for_markers(markers: Mapping[int, list[list[float]]]) -> dict[str, Any]:
    """Choose the most specific known board layout from detected marker IDs."""
    detected_ids = set(int(marker_id) for marker_id in markers)
    v2_ids = set(A3_V2_CHARUCO_PLUMB_CONFIG["location_marker_squares_mm"])
    if v2_ids.issubset(detected_ids):
        return dict(A3_V2_CHARUCO_PLUMB_CONFIG)
    return dict(DEFAULT_BOARD_CONFIG)


def _outer_corner(corners: list[list[float]], corner_name: str) -> np.ndarray:
    points = np.asarray(corners, dtype=np.float32)
    sums = points[:, 0] + points[:, 1]
    diffs = points[:, 0] - points[:, 1]
    if corner_name == "top_left":
        return points[int(np.argmin(sums))]
    if corner_name == "top_right":
        return points[int(np.argmax(diffs))]
    if corner_name == "bottom_left":
        return points[int(np.argmin(diffs))]
    if corner_name == "bottom_right":
        return points[int(np.argmax(sums))]
    raise ValueError(f"Unknown corner name: {corner_name}")


def warp_board_by_aruco(
    image: np.ndarray,
    marker_positions: Mapping[int, list[list[float]]],
    board_config: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Warp the board using four corner ArUco markers.

    The current V0.2 prototype uses the outer corner of each marker as the
    perspective anchor. If later versions record exact marker offsets from the
    paper edge, this function can be extended without changing the app flow.
    """
    config = dict(DEFAULT_BOARD_CONFIG)
    if board_config:
        config.update(board_config)

    if "location_marker_squares_mm" in config:
        return _warp_board_by_marker_layout(image, marker_positions, config)

    corner_ids = config["aruco_corner_ids"]
    missing = [name for name, marker_id in corner_ids.items() if int(marker_id) not in marker_positions]
    if missing:
        raise ValueError(f"Missing required ArUco markers: {', '.join(missing)}")

    source_points = np.float32(
        [
            _outer_corner(marker_positions[int(corner_ids["top_left"])], "top_left"),
            _outer_corner(marker_positions[int(corner_ids["top_right"])], "top_right"),
            _outer_corner(marker_positions[int(corner_ids["bottom_left"])], "bottom_left"),
            _outer_corner(marker_positions[int(corner_ids["bottom_right"])], "bottom_right"),
        ]
    )

    output_width = int(round(float(config["board_width_mm"]) * float(config["output_px_per_mm"])))
    output_height = int(round(float(config["board_height_mm"]) * float(config["output_px_per_mm"])))
    destination_points = np.float32(
        [
            [0, 0],
            [output_width - 1, 0],
            [0, output_height - 1],
            [output_width - 1, output_height - 1],
        ]
    )

    transform = cv2.getPerspectiveTransform(source_points, destination_points)
    warped = cv2.warpPerspective(ensure_rgb(image), transform, (output_width, output_height))
    info = {
        "output_width_px": output_width,
        "output_height_px": output_height,
        "mm_per_pixel": float(config["board_width_mm"]) / output_width,
        "anchor_note": "outer marker corners",
    }
    return warped, transform, info


def _marker_square_corners(marker: Mapping[str, float]) -> np.ndarray:
    x = float(marker["x"])
    y = float(marker["y"])
    size = float(marker["size"])
    return np.float32(
        [
            [x, y],
            [x + size, y],
            [x + size, y + size],
            [x, y + size],
        ]
    )


def _warp_board_by_marker_layout(
    image: np.ndarray,
    marker_positions: Mapping[int, list[list[float]]],
    board_config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Warp a board using all detected ArUco markers with known board coordinates."""
    marker_layout = board_config["location_marker_squares_mm"]
    detected_layout_ids = sorted(set(int(marker_id) for marker_id in marker_positions) & set(marker_layout))
    corner_ids = {int(marker_id) for marker_id in board_config["aruco_corner_ids"].values()}
    missing_corners = sorted(corner_ids - set(detected_layout_ids))
    if missing_corners:
        raise ValueError(f"Missing required corner ArUco markers: {missing_corners}")
    if len(detected_layout_ids) < 4:
        raise ValueError("At least four known ArUco markers are required for board warping.")

    source_points: list[np.ndarray] = []
    destination_points: list[np.ndarray] = []
    px_per_mm = float(board_config["output_px_per_mm"])
    for marker_id in detected_layout_ids:
        source_points.append(np.asarray(marker_positions[marker_id], dtype=np.float32))
        destination_points.append(_marker_square_corners(marker_layout[marker_id]) * px_per_mm)

    source = np.concatenate(source_points, axis=0)
    destination = np.concatenate(destination_points, axis=0)
    transform, inlier_mask = cv2.findHomography(source, destination, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if transform is None:
        raise ValueError("Could not estimate a multi-marker homography.")

    output_width = int(round(float(board_config["board_width_mm"]) * px_per_mm))
    output_height = int(round(float(board_config["board_height_mm"]) * px_per_mm))
    warped = cv2.warpPerspective(ensure_rgb(image), transform, (output_width, output_height))

    projected = cv2.perspectiveTransform(source.reshape(-1, 1, 2), transform).reshape(-1, 2)
    errors = np.linalg.norm(projected - destination, axis=1)
    if inlier_mask is not None and inlier_mask.size:
        inlier_errors = errors[inlier_mask.ravel().astype(bool)]
    else:
        inlier_errors = errors

    info = {
        "board_name": board_config.get("name", "marker_layout"),
        "output_width_px": output_width,
        "output_height_px": output_height,
        "mm_per_pixel": float(board_config["board_width_mm"]) / output_width,
        "anchor_note": "multi-marker layout",
        "detected_layout_ids": detected_layout_ids,
        "reprojection_error_px_mean": float(np.mean(inlier_errors)) if len(inlier_errors) else float(np.mean(errors)),
        "reprojection_error_px_max": float(np.max(inlier_errors)) if len(inlier_errors) else float(np.max(errors)),
        "inlier_count": int(inlier_mask.sum()) if inlier_mask is not None else int(len(source)),
        "point_count": int(len(source)),
    }
    return warped, transform, info


def draw_detected_aruco(image: np.ndarray, markers: Mapping[int, list[list[float]]]) -> np.ndarray:
    """Draw detected markers on a copy of the image."""
    rgb = ensure_rgb(image).copy()
    if not markers:
        return rgb

    corners = [np.asarray([corners], dtype=np.float32) for corners in markers.values()]
    ids = np.asarray([[marker_id] for marker_id in markers.keys()], dtype=np.int32)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.aruco.drawDetectedMarkers(bgr, corners, ids)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
