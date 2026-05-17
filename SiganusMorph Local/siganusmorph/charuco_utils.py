"""ChArUco board helpers for camera calibration prototypes."""

from __future__ import annotations

from typing import Any, Mapping

import cv2
import numpy as np

from .aruco_utils import ARUCO_DICTIONARIES
from .config import A3_V2_CHARUCO_PLUMB_CONFIG
from .image_utils import ensure_rgb


def create_charuco_board(config: Mapping[str, Any] | None = None) -> cv2.aruco.CharucoBoard:
    """Create the V2 ChArUco panel board."""
    board_config = dict(A3_V2_CHARUCO_PLUMB_CONFIG["charuco"])
    if config:
        board_config.update(config)
    dictionary_id = ARUCO_DICTIONARIES[board_config["dictionary_name"]]
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    return cv2.aruco.CharucoBoard(
        (int(board_config["squares_x"]), int(board_config["squares_y"])),
        float(board_config["square_length_mm"]),
        float(board_config["marker_length_mm"]),
        dictionary,
    )


def detect_charuco_panel(
    image: np.ndarray,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Detect ChArUco chessboard corners from the V2 calibration panel."""
    board = create_charuco_board(config)
    gray = cv2.cvtColor(ensure_rgb(image), cv2.COLOR_RGB2GRAY)
    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

    if marker_ids is None:
        marker_ids_list: list[int] = []
    else:
        marker_ids_list = [int(value) for value in marker_ids.flatten().tolist()]

    if charuco_ids is None:
        charuco_ids_list: list[int] = []
        corners_list: list[list[float]] = []
    else:
        charuco_ids_list = [int(value) for value in charuco_ids.flatten().tolist()]
        corners_list = np.asarray(charuco_corners, dtype=float).reshape(-1, 2).tolist()

    return {
        "success": len(charuco_ids_list) >= 4,
        "marker_count": len(marker_ids_list),
        "marker_ids": marker_ids_list,
        "charuco_corner_count": len(charuco_ids_list),
        "charuco_ids": charuco_ids_list,
        "charuco_corners": corners_list,
        "board_size": tuple(int(v) for v in board.getChessboardSize()),
    }
