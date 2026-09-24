"""Helpers for the formal V1.0 Streamlit front-end.

The functions here are intentionally read-only with respect to project labels
and historical result files.  They provide a clean user-facing wrapper around
existing calibration, preannotation, measurement, and export utilities.
"""

from __future__ import annotations

from io import BytesIO
from datetime import datetime
import math
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .aruco_utils import detect_aruco_markers, draw_detected_aruco, select_board_config_for_markers, warp_board_by_aruco
from .config import DEFAULT_BOARD_CONFIG
from .dual_axis_measurement import MEASUREMENT_AXIS_AUTO_QC, compute_measurement_axis_metadata
from .heatmap_preannotation import default_heatmap_model
from .image_utils import ensure_rgb, image_from_bytes
from .measurements import calculate_measurements
from .preannotation import default_preannotation_model
from .realworld_review import full_to_short_keypoints
from .runtime_resources import collect_runtime_memory, configure_runtime_threads, low_resource_mode, measurement_inference_guard
from .v06_preannotation import default_v05_heatmap_model, preannotate_warped_image_v06_keypointwise
from .visualization import draw_enhanced_preannotation_overlay, draw_keypoints_and_measurements


FORMAL_OUTPUT_DIR = Path("results") / "formal_v1_user_outputs"
DEFAULT_MM_PER_PIXEL = 0.1

FORMAL_POINT_LABELS = {
    "P1_snout_tip": "P1",
    "P2_eye_anterior": "P2",
    "P3_operculum_posterior": "P3",
    "P4_peduncle_anterior": "P4",
    "P5_caudal_base": "P5",
    "P6_caudal_fork": "P6",
    "P7U_upper_lobe_tip": "P7U",
    "P7L_lower_lobe_tip": "P7L",
    "body_depth_upper": "body_depth_upper",
    "body_depth_lower": "body_depth_lower",
    "peduncle_depth_upper": "peduncle_depth_upper",
    "peduncle_depth_lower": "peduncle_depth_lower",
}

FORMAL_TO_FULL_KEYPOINT = {
    "P1_snout_tip": "P1_snout_tip",
    "P2_eye_anterior": "P2_eye_front",
    "P3_operculum_posterior": "P3_operculum_posterior",
    "P4_peduncle_anterior": "P4_peduncle_start_midpoint",
    "P5_caudal_base": "P5_caudal_base_midpoint",
    "P6_caudal_fork": "P6_caudal_fork_midpoint",
    "P7U_upper_lobe_tip": "P7U_caudal_fin_upper_tip",
    "P7L_lower_lobe_tip": "P7L_caudal_fin_lower_tip",
    "body_depth_upper": "P8_body_depth_dorsal",
    "body_depth_lower": "P9_body_depth_ventral",
    "peduncle_depth_upper": "P10_peduncle_depth_dorsal",
    "peduncle_depth_lower": "P11_peduncle_depth_ventral",
}

SIMPLIFIED_EXPORT_COLUMNS = [
    "image_name",
    "specimen_id",
    "weight_g",
    "TL_compressed_virtual_mm",
    "TL_open_projection_mm",
    "SL_mm",
    "FL_mm",
    "body_depth_mm",
    "head_length_mm",
    "snout_length_mm",
    "caudal_peduncle_length_mm",
    "caudal_peduncle_depth_mm",
    "measurement_status",
    "notes",
]


def safe_name(name: str | None) -> str:
    """Return a UI-safe basename without exposing local paths."""
    if not name:
        return "未命名图片"
    return Path(str(name)).name


def image_from_upload(uploaded_file: Any) -> np.ndarray:
    """Read an uploaded Streamlit file into an RGB numpy image."""
    return image_from_bytes(uploaded_file.getvalue())


def image_bytes_png(image: np.ndarray | Image.Image) -> bytes:
    """Encode an image as PNG bytes for download buttons."""
    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        pil_image = Image.fromarray(ensure_rgb(image))
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def _xy(value: Any) -> tuple[float, float] | None:
    if isinstance(value, Mapping):
        if "x" in value and "y" in value:
            try:
                return float(value["x"]), float(value["y"])
            except (TypeError, ValueError):
                return None
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None
    return None


def _point_payload(point: tuple[float, float] | None) -> list[float] | None:
    if point is None:
        return None
    return [float(point[0]), float(point[1])]


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _unit_from_points(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float] | None:
    length = _distance(a, b)
    if length <= 1e-9:
        return None
    return (b[0] - a[0]) / length, (b[1] - a[1]) / length


def _dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _angle_deg(a: tuple[float, float], b: tuple[float, float]) -> float | None:
    la = math.hypot(a[0], a[1])
    lb = math.hypot(b[0], b[1])
    if la <= 1e-9 or lb <= 1e-9:
        return None
    value = max(-1.0, min(1.0, _dot(a, b) / (la * lb)))
    return math.degrees(math.acos(value))


def _curve_length(points: list[tuple[float, float]]) -> float:
    return sum(_distance(a, b) for a, b in zip(points, points[1:]))


def _axis_points_from_metadata(metadata: Mapping[str, Any]) -> list[tuple[float, float]]:
    axes = metadata.get("measurement_axes", {}) if isinstance(metadata.get("measurement_axes", {}), Mapping) else {}
    body_axis = axes.get("body_midline_axis", {}) if isinstance(axes.get("body_midline_axis", {}), Mapping) else {}
    points = body_axis.get("axis_points", [])
    parsed: list[tuple[float, float]] = []
    if isinstance(points, list):
        for point in points:
            xy = _xy(point)
            if xy is not None:
                parsed.append(xy)
    return parsed


def body_midline_tail_axis_unit(
    formal_points: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> tuple[tuple[float, float] | None, str]:
    """Return the formal tail direction from body_midline_axis near P5."""
    axis_points = _axis_points_from_metadata(metadata)
    p5 = _xy(formal_points.get("P5_caudal_base"))
    if len(axis_points) >= 2:
        if p5 is not None:
            best_index = min(range(len(axis_points)), key=lambda idx: _distance(axis_points[idx], p5))
            if best_index > 0:
                unit = _unit_from_points(axis_points[best_index - 1], axis_points[best_index])
                if unit is not None:
                    return unit, "body_midline_axis_local_tangent_near_P5"
            if best_index < len(axis_points) - 1:
                unit = _unit_from_points(axis_points[best_index], axis_points[best_index + 1])
                if unit is not None:
                    return unit, "body_midline_axis_local_tangent_near_P5"
        unit = _unit_from_points(axis_points[-2], axis_points[-1])
        if unit is not None:
            return unit, "body_midline_axis_terminal_tangent"
    p4 = _xy(formal_points.get("P4_peduncle_anterior"))
    if p4 is not None and p5 is not None:
        unit = _unit_from_points(p4, p5)
        if unit is not None:
            return unit, "P4_to_P5_fallback"
    return None, "tail_axis_unavailable"


def build_formal_measurement_points(
    full_keypoints: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, list[float]]:
    """Build display/measurement points for the formal UI.

    P8/P9 and P10/P11 are converted into named measurement endpoints and may
    come from geometric suggestions.  C1-C4 remain hidden and are not exported
    as formal display points.
    """
    body_depth = metadata.get("body_depth_geometry", {}) if isinstance(metadata.get("body_depth_geometry", {}), Mapping) else {}
    peduncle = metadata.get("peduncle_depth_geometry", {}) if isinstance(metadata.get("peduncle_depth_geometry", {}), Mapping) else {}
    out: dict[str, list[float]] = {}
    for formal_name, full_name in FORMAL_TO_FULL_KEYPOINT.items():
        xy = None
        if formal_name == "body_depth_upper":
            xy = _xy(body_depth.get("P8_geometric"))
        elif formal_name == "body_depth_lower":
            xy = _xy(body_depth.get("P9_geometric"))
        elif formal_name == "peduncle_depth_upper":
            xy = _xy(peduncle.get("P10_geometric"))
        elif formal_name == "peduncle_depth_lower":
            xy = _xy(peduncle.get("P11_geometric"))
        if xy is None:
            xy = _xy(full_keypoints.get(full_name))
        if xy is not None:
            out[formal_name] = [xy[0], xy[1]]
    return out


def formal_points_to_full_keypoints(formal_points: Mapping[str, Any], base_full: Mapping[str, Any]) -> dict[str, list[float]]:
    """Convert formal measurement points back to full keypoint names for session export."""
    out: dict[str, list[float]] = {}
    for key, value in base_full.items():
        xy = _xy(value)
        if xy is not None:
            out[str(key)] = [xy[0], xy[1]]
    for formal_name, full_name in FORMAL_TO_FULL_KEYPOINT.items():
        xy = _xy(formal_points.get(formal_name))
        if xy is not None:
            out[full_name] = [xy[0], xy[1]]
    return out


def recalculate_formal_measurements(
    formal_points: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    mm_per_pixel: float = DEFAULT_MM_PER_PIXEL,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recalculate formal table values from current draggable measurement points."""
    p1 = _xy(formal_points.get("P1_snout_tip"))
    p2 = _xy(formal_points.get("P2_eye_anterior"))
    p3 = _xy(formal_points.get("P3_operculum_posterior"))
    p4 = _xy(formal_points.get("P4_peduncle_anterior"))
    p5 = _xy(formal_points.get("P5_caudal_base"))
    p6 = _xy(formal_points.get("P6_caudal_fork"))
    p7u = _xy(formal_points.get("P7U_upper_lobe_tip"))
    p7l = _xy(formal_points.get("P7L_lower_lobe_tip"))
    bd_u = _xy(formal_points.get("body_depth_upper"))
    bd_l = _xy(formal_points.get("body_depth_lower"))
    pd_u = _xy(formal_points.get("peduncle_depth_upper"))
    pd_l = _xy(formal_points.get("peduncle_depth_lower"))

    measurements: dict[str, Any] = {}
    derived: dict[str, Any] = {}
    axis_unit, axis_source = body_midline_tail_axis_unit(formal_points, metadata)
    axis_points = _axis_points_from_metadata(metadata)
    axis_sl_px = _curve_length(axis_points) if len(axis_points) >= 2 else None
    sl_px = _distance(p1, p5) if p1 and p5 else axis_sl_px
    sl_mm = sl_px * mm_per_pixel if sl_px is not None else None
    if p1 and p5:
        measurements["SL_straight_mm"] = _distance(p1, p5) * mm_per_pixel
    if sl_mm is not None:
        measurements["SL_mm"] = sl_mm
        measurements["SL_final_mm"] = sl_mm
        measurements["SL_curve_mm"] = sl_mm
    if axis_sl_px is not None:
        measurements["SL_body_midline_axis_reference_mm"] = axis_sl_px * mm_per_pixel
    if p1 and p6:
        measurements["FL_mm"] = _distance(p1, p6) * mm_per_pixel
    if bd_u and bd_l:
        measurements["body_depth_mm"] = _distance(bd_u, bd_l) * mm_per_pixel
    if p1 and p3:
        measurements["head_length_mm"] = _distance(p1, p3) * mm_per_pixel
        measurements["head_length_straight_mm"] = measurements["head_length_mm"]
    if p1 and p2:
        measurements["snout_length_mm"] = _distance(p1, p2) * mm_per_pixel
        measurements["snout_length_straight_mm"] = measurements["snout_length_mm"]
    if p4 and p5:
        measurements["caudal_peduncle_length_mm"] = _distance(p4, p5) * mm_per_pixel
        measurements["caudal_peduncle_length_straight_mm"] = measurements["caudal_peduncle_length_mm"]
    if pd_u and pd_l:
        measurements["caudal_peduncle_depth_mm"] = _distance(pd_u, pd_l) * mm_per_pixel

    if p5 and p7u and p7l and axis_unit is not None:
        upper_vec = (p7u[0] - p5[0], p7u[1] - p5[1])
        lower_vec = (p7l[0] - p5[0], p7l[1] - p5[1])
        proj_u = _dot(upper_vec, axis_unit)
        proj_l = _dot(lower_vec, axis_unit)
        open_extension_px = max(0.0, proj_u, proj_l)
        ru = _distance(p5, p7u)
        rl = _distance(p5, p7l)
        rmax = max(ru, rl)
        p7v_open = (p5[0] + open_extension_px * axis_unit[0], p5[1] + open_extension_px * axis_unit[1])
        p7v_comp = (p5[0] + rmax * axis_unit[0], p5[1] + rmax * axis_unit[1])
        tl_open = (sl_mm + open_extension_px * mm_per_pixel) if sl_mm is not None else None
        tl_comp = (sl_mm + rmax * mm_per_pixel) if sl_mm is not None else None
        derived["P7V_open_projection"] = {"x": p7v_open[0], "y": p7v_open[1]}
        derived["P7V_virtual_tail_tip"] = {"x": p7v_comp[0], "y": p7v_comp[1]}
        derived["P7V_compressed_virtual"] = {"x": p7v_comp[0], "y": p7v_comp[1]}
        measurements.update(
            {
                "P7V_open_projection_x": p7v_open[0],
                "P7V_open_projection_y": p7v_open[1],
                "P7V_compressed_virtual_x": p7v_comp[0],
                "P7V_compressed_virtual_y": p7v_comp[1],
                "TL_open_projection_mm": tl_open,
                "TL_compressed_virtual_mm": tl_comp,
                "TL_final_mm": tl_comp,
                "TL_curve_mm": tl_comp,
                "RU_mm": ru * mm_per_pixel,
                "RL_mm": rl * mm_per_pixel,
                "Rmax_mm": rmax * mm_per_pixel,
                "upper_lobe_angle_to_axis_deg": _angle_deg(upper_vec, axis_unit),
                "lower_lobe_angle_to_axis_deg": _angle_deg(lower_vec, axis_unit),
                "inter_lobe_open_angle_deg": _angle_deg(upper_vec, lower_vec),
                "tail_axis_source": axis_source,
            }
        )
        if tl_open not in (None, "") and tl_comp not in (None, ""):
            diff = float(tl_comp) - float(tl_open)
            measurements["TL_difference_mm"] = diff
            measurements["TL_difference_percent"] = 100.0 * diff / float(tl_open) if abs(float(tl_open)) > 1e-9 else None
    return measurements, derived


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Return an in-memory XLSX workbook."""
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return buffer.getvalue()


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Return UTF-8-SIG CSV bytes so Excel opens Chinese headers cleanly."""
    return df.to_csv(index=False).encode("utf-8-sig")


def calibrate_with_aruco(image: np.ndarray) -> dict[str, Any]:
    """Try ArUco/ChArUco-style board calibration without writing files."""
    detection = detect_aruco_markers(image)
    markers = detection.get("markers", {})
    marker_count = len(markers) if isinstance(markers, Mapping) else 0
    marker_overlay = draw_detected_aruco(image, markers) if isinstance(markers, Mapping) else image
    result: dict[str, Any] = {
        "status": "failed",
        "success": False,
        "calibration_message": "未检测到足够的校准标记",
        "status_text": "未检测到足够的校准标记",
        "marker_count": marker_count,
        "detected_markers": detection.get("ids", []),
        "raw_marker_overlay": marker_overlay,
        "detected_board_corners_raw": [],
        "mm_per_pixel": None,
        "mm_per_pixel_source": "",
        "warped_image": None,
        "homography_raw_to_warped": None,
        "info": {},
    }
    if marker_count < 4 or not isinstance(markers, Mapping):
        return result
    try:
        board_config = select_board_config_for_markers(markers)
        warped, _transform, info = warp_board_by_aruco(image, markers, board_config)
    except Exception as exc:  # noqa: BLE001 - user-facing calibration status.
        result["calibration_message"] = f"校准失败：{exc}"
        result["status_text"] = f"校准失败：{exc}"
        return result
    transform = np.asarray(_transform, dtype=float)
    board_corners = _raw_board_corners_from_homography(transform, warped.shape)
    corner_overlay = draw_board_corner_overlay(marker_overlay, board_corners)
    result.update(
        {
            "status": "success",
            "success": True,
            "calibration_message": "校准成功，后续测量将在校正图像坐标中进行",
            "status_text": "校准成功，已生成透视校正图像",
            "mm_per_pixel": float(info.get("mm_per_pixel", DEFAULT_MM_PER_PIXEL) or DEFAULT_MM_PER_PIXEL),
            "mm_per_pixel_source": "calibration_board",
            "warped_image": warped,
            "homography_raw_to_warped": transform.tolist(),
            "detected_board_corners_raw": board_corners,
            "raw_marker_overlay": corner_overlay,
            "info": info,
        }
    )
    return result


def _raw_board_corners_from_homography(transform: np.ndarray, warped_shape: tuple[int, ...]) -> list[list[float]]:
    """Project warped board corners back to raw-image coordinates for display."""
    if transform.shape != (3, 3):
        return []
    inv = np.linalg.inv(transform)
    height, width = int(warped_shape[0]), int(warped_shape[1])
    warped_corners = np.asarray(
        [[[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]]],
        dtype=np.float32,
    )
    raw = cv2.perspectiveTransform(warped_corners, inv)[0]
    return [[float(x), float(y)] for x, y in raw]


def draw_board_corner_overlay(image: np.ndarray, corners: list[list[float]]) -> np.ndarray:
    """Draw detected/manual board corners on the raw calibration image."""
    canvas = Image.fromarray(ensure_rgb(image))
    draw = ImageDraw.Draw(canvas)
    if len(corners) >= 4:
        pts = [(float(x), float(y)) for x, y in corners[:4]]
        draw.line(pts + [pts[0]], fill=(255, 230, 0), width=5)
        for label, (x, y) in zip(("TL", "TR", "BR", "BL"), pts):
            r = 10
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 230, 0), outline=(0, 0, 0), width=2)
            draw.text((x + r + 4, y - r - 4), label, fill=(255, 230, 0))
    return np.asarray(canvas)


def default_manual_board_corners(image: np.ndarray) -> pd.DataFrame:
    """Return editable approximate board corners for manual calibration."""
    height, width = image.shape[:2]
    margin_x = width * 0.08
    margin_y = height * 0.08
    rows = [
        ("top_left", margin_x, margin_y),
        ("top_right", width - margin_x, margin_y),
        ("bottom_right", width - margin_x, height - margin_y),
        ("bottom_left", margin_x, height - margin_y),
    ]
    return pd.DataFrame(rows, columns=["corner", "x", "y"])


def manual_calibrate_from_corners(
    image: np.ndarray,
    corner_rows: pd.DataFrame,
    *,
    board_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a formal calibration state from manually supplied raw board corners."""
    cfg = dict(DEFAULT_BOARD_CONFIG)
    if board_config:
        cfg.update(dict(board_config))
    expected = ["top_left", "top_right", "bottom_right", "bottom_left"]
    row_map = {str(row["corner"]): row for row in corner_rows.to_dict(orient="records")}
    missing = [name for name in expected if name not in row_map]
    if missing:
        return {
            "status": "manual_required",
            "success": False,
            "calibration_message": f"缺少手动角点：{', '.join(missing)}",
            "mm_per_pixel": None,
        }
    try:
        src = np.float32([[float(row_map[name]["x"]), float(row_map[name]["y"])] for name in expected])
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "manual_required",
            "success": False,
            "calibration_message": f"手动角点格式无效：{exc}",
            "mm_per_pixel": None,
        }
    output_width = int(round(float(cfg["board_width_mm"]) * float(cfg["output_px_per_mm"])))
    output_height = int(round(float(cfg["board_height_mm"]) * float(cfg["output_px_per_mm"])))
    dst = np.float32(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ]
    )
    transform = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(ensure_rgb(image), transform, (output_width, output_height))
    mm_per_pixel = float(cfg["board_width_mm"]) / output_width
    corners = src.astype(float).tolist()
    return {
        "status": "success",
        "success": True,
        "calibration_message": "手动校准成功，后续测量将在校正图像坐标中进行",
        "status_text": "手动校准成功",
        "marker_count": 0,
        "detected_markers": [],
        "raw_marker_overlay": draw_board_corner_overlay(image, corners),
        "detected_board_corners_raw": corners,
        "mm_per_pixel": mm_per_pixel,
        "mm_per_pixel_source": "manual_board_corners",
        "warped_image": warped,
        "homography_raw_to_warped": transform.tolist(),
        "info": {
            "board_name": cfg.get("name", "manual_board"),
            "output_width_px": output_width,
            "output_height_px": output_height,
            "mm_per_pixel": mm_per_pixel,
            "anchor_note": "manual board corners",
        },
    }


def check_formal_coordinate_consistency(
    calibration_state: Mapping[str, Any] | None,
    *,
    formal_points: Mapping[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """Validate that formal measurement uses warped-image coordinates only."""
    errors: list[str] = []
    if not isinstance(calibration_state, Mapping):
        return False, ["尚未运行校准"]
    if calibration_state.get("status") != "success":
        errors.append("校准未成功")
    warped = calibration_state.get("warped_image")
    if warped is None:
        errors.append("缺少校正后图像 warped_image")
    mm_per_pixel = calibration_state.get("mm_per_pixel")
    if mm_per_pixel in ("", None):
        errors.append("缺少校准比例尺 mm_per_pixel")
    if not calibration_state.get("mm_per_pixel_source"):
        errors.append("mm_per_pixel 不是来自成功校准")
    if calibration_state.get("mm_per_pixel_source") == "fallback_default":
        errors.append("不允许使用默认 fallback mm_per_pixel 继续测量")
    if warped is not None and formal_points:
        height, width = warped.shape[:2]
        for name, point in formal_points.items():
            xy = _xy(point)
            if xy is None:
                errors.append(f"{name} 坐标无效")
                continue
            if xy[0] < 0 or xy[1] < 0 or xy[0] >= width or xy[1] >= height:
                errors.append(f"{name} 超出 warped_image 边界")
    return not errors, errors


def run_recommended_measurement(
    image: np.ndarray,
    image_name: str,
    project_root: Path,
    *,
    mm_per_pixel: float = DEFAULT_MM_PER_PIXEL,
) -> dict[str, Any]:
    """Run the recommended v0.6 + auto_qc_gated measurement workflow in memory."""
    configure_runtime_threads()
    with measurement_inference_guard():
        result = preannotate_warped_image_v06_keypointwise(
            image,
            image_name,
            project_root,
            v034_model_path=default_preannotation_model(project_root),
            v01_heatmap_model_path=default_heatmap_model(project_root),
            v05_heatmap_model_path=default_v05_heatmap_model(project_root),
            mm_per_pixel=mm_per_pixel,
        )
    full_keypoints = dict(result.get("keypoints", {}) or {})
    short_keypoints = full_to_short_keypoints(full_keypoints)
    measurements = calculate_measurements(short_keypoints, mm_per_pixel)
    axis_payload = compute_measurement_axis_metadata(
        image,
        full_keypoints,
        mm_per_pixel=mm_per_pixel,
        measurement_axis_mode=MEASUREMENT_AXIS_AUTO_QC,
        fish_mask=result.get("fish_mask"),
    )
    metadata = dict(result.get("metadata", {}) or {})
    metadata["measurement_axis_mode"] = MEASUREMENT_AXIS_AUTO_QC
    metadata["measurement_axes"] = axis_payload.get("measurement_axes", {})
    metadata.update(axis_payload.get("measurement_axis_row_fields", {}) or {})
    metadata["measurements"] = measurements
    metadata["mm_per_pixel"] = float(mm_per_pixel)
    metadata["show_qc_warnings"] = True
    metadata["show_selected_measurement_axis_only"] = True
    metadata["show_body_midline_axis"] = False
    metadata["show_model_axis"] = False
    metadata["show_geometric_c_points"] = False
    metadata["show_body_midline_contour"] = False
    metadata["show_dual_axis_comparison"] = False
    metadata["show_geometric_body_depth"] = True
    metadata["show_body_depth_width_profile"] = True
    metadata["show_geometric_peduncle_depth"] = True
    metadata["show_peduncle_depth_width_profile"] = True
    metadata["show_compressed_tail_p7v"] = True
    metadata["show_open_projection_p7v"] = False
    formal_points = build_formal_measurement_points(full_keypoints, metadata)
    formal_measurements, formal_derived = recalculate_formal_measurements(
        formal_points,
        metadata,
        mm_per_pixel=mm_per_pixel,
    )
    raw_result: dict[str, Any] = result
    if low_resource_mode():
        # The formal page does not consume nested v0.4/v0.5 debug payloads.
        # Keeping them in Streamlit session state retains masks and intermediate
        # arrays after inference, which is costly on small cloud instances.
        raw_result = {
            "keypoints": full_keypoints,
            "confidences": dict(result.get("confidences", {}) or {}),
            "fish_bbox": result.get("fish_bbox"),
            "segmentation_quality": result.get("segmentation_quality"),
            "metadata": metadata,
            "runtime_payload_compacted": True,
        }
        collect_runtime_memory()
    return {
        "raw_result": raw_result,
        "full_keypoints": full_keypoints,
        "short_keypoints": short_keypoints,
        "measurements": measurements,
        "metadata": metadata,
        "formal_measurement_points": formal_points,
        "formal_measurements": formal_measurements,
        "formal_derived_points": formal_derived,
    }


def draw_formal_measurement_overlay(
    image: np.ndarray,
    formal_points: Mapping[str, Any],
    measurements: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    image_name: str = "",
    specimen_id: str = "",
) -> np.ndarray:
    """Draw a simplified overlay suitable for normal users and screenshots."""
    from PIL import ImageDraw, ImageFont

    canvas = Image.fromarray(ensure_rgb(image))
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    radius = max(5, round(min(width, height) / 280))

    def font(size: int = 16) -> ImageFont.ImageFont:
        for candidate in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "arial.ttf"):
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    def label(text: str, xy: tuple[float, float], color: tuple[int, int, int]) -> None:
        x, y = xy
        fnt = font(15)
        for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            draw.text((x + ox, y + oy), text, font=fnt, fill=(0, 0, 0))
        draw.text((x, y), text, font=fnt, fill=color)

    def dashed(a: tuple[float, float], b: tuple[float, float], color: tuple[int, int, int], line_width: int = 3) -> None:
        dist = _distance(a, b)
        if dist <= 1e-9:
            return
        ux, uy = (b[0] - a[0]) / dist, (b[1] - a[1]) / dist
        step = max(10, round(min(width, height) / 100))
        pos = 0.0
        on = True
        while pos < dist:
            nxt = min(dist, pos + step)
            if on:
                draw.line(
                    (a[0] + ux * pos, a[1] + uy * pos, a[0] + ux * nxt, a[1] + uy * nxt),
                    fill=color,
                    width=line_width,
                )
            on = not on
            pos = nxt

    axis_points = _axis_points_from_metadata(metadata)
    if len(axis_points) >= 2:
        for a, b in zip(axis_points, axis_points[1:]):
            dashed(a, b, (0, 170, 230), max(3, radius // 2))
        label("body_midline_axis", (axis_points[0][0] + radius + 4, axis_points[0][1] + radius + 4), (0, 170, 230))

    line_specs = [
        ("body_depth_upper", "body_depth_lower", (0, 140, 255), "body_depth"),
        ("peduncle_depth_upper", "peduncle_depth_lower", (0, 210, 190), "peduncle_depth"),
        ("P1_snout_tip", "P5_caudal_base", (255, 220, 80), "SL"),
        ("P4_peduncle_anterior", "P5_caudal_base", (255, 210, 110), "peduncle_length"),
    ]
    for a_key, b_key, color, text in line_specs:
        a = _xy(formal_points.get(a_key))
        b = _xy(formal_points.get(b_key))
        if a and b:
            draw.line((a[0], a[1], b[0], b[1]), fill=color, width=max(3, radius // 2 + 1))
            label(text, ((a[0] + b[0]) / 2 + 6, (a[1] + b[1]) / 2 + 6), color)

    for key, point in formal_points.items():
        xy = _xy(point)
        if xy is None:
            continue
        if key.startswith("body_depth") or key.startswith("peduncle_depth"):
            color = (0, 140, 255) if key.startswith("body_depth") else (0, 210, 190)
        else:
            color = (255, 245, 160)
        draw.ellipse((xy[0] - radius, xy[1] - radius, xy[0] + radius, xy[1] + radius), fill=color, outline=(30, 30, 30), width=2)
        label(FORMAL_POINT_LABELS.get(key, key), (xy[0] + radius + 3, xy[1] - radius - 3), color)

    p7v = _xy(
        {
            "x": measurements.get("P7V_compressed_virtual_x"),
            "y": measurements.get("P7V_compressed_virtual_y"),
        }
    )
    if p7v is not None:
        r = radius + 8
        star_points = []
        for idx in range(10):
            rr = r if idx % 2 == 0 else r * 0.45
            angle = -math.pi / 2 + idx * math.pi / 5
            star_points.append((p7v[0] + math.cos(angle) * rr, p7v[1] + math.sin(angle) * rr))
        draw.polygon(star_points, fill=(160, 85, 255), outline=(255, 255, 255))
        label("P7V", (p7v[0] + r + 4, p7v[1] - r - 4), (160, 85, 255))

    title = " | ".join(part for part in (safe_name(image_name), specimen_id) if part)
    if title:
        fnt = font(18)
        bbox = draw.textbbox((0, 0), title, font=fnt)
        draw.rectangle((8, 8, bbox[2] + 18, bbox[3] + 18), fill=(0, 0, 0))
        draw.text((13, 12), title, font=fnt, fill=(255, 255, 255))
    return np.asarray(canvas)


def formal_editor_payload(
    image: np.ndarray,
    formal_points: Mapping[str, Any],
    measurements: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    max_width: int = 1100,
) -> tuple[Image.Image, float, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build scaled image and overlay objects for the draggable formal editor."""
    pil = Image.fromarray(ensure_rgb(image))
    original_width, _original_height = pil.size
    if original_width > max_width:
        display_width = int(max_width)
        display_height = round(pil.height * display_width / original_width)
        display = pil.resize((display_width, display_height), Image.Resampling.LANCZOS)
        scale = original_width / display_width
    else:
        display = pil
        scale = 1.0

    def scale_point(point: Any) -> tuple[float, float] | None:
        xy = _xy(point)
        if xy is None:
            return None
        return xy[0] / scale, xy[1] / scale

    point_payloads: list[dict[str, Any]] = []
    for name, point in formal_points.items():
        xy = scale_point(point)
        if xy is None:
            continue
        is_depth = name.startswith("body_depth") or name.startswith("peduncle_depth")
        point_payloads.append(
            {
                "name": name,
                "label": FORMAL_POINT_LABELS.get(name, name),
                "x": xy[0],
                "y": xy[1],
                "color": "#008cff" if name.startswith("body_depth") else ("#00d2be" if name.startswith("peduncle_depth") else "#fff2a0"),
                "outline": "#202020",
                "radius": 5 if is_depth else 4,
                "hover_radius": 8,
                "hit_radius": 16,
                "draggable": True,
            }
        )

    lines = [
        {"a": "body_depth_upper", "b": "body_depth_lower", "color": "#00cfff", "width": 4, "label": "body_depth"},
        {"a": "peduncle_depth_upper", "b": "peduncle_depth_lower", "color": "#00e0bf", "width": 4, "label": "peduncle_depth"},
        {"a": "P1_snout_tip", "b": "P5_caudal_base", "color": "#f4d35e", "width": 2, "label": "SL"},
        {"a": "P4_peduncle_anterior", "b": "P5_caudal_base", "color": "#f7c76b", "width": 2, "label": "peduncle_length"},
    ]
    polylines: list[dict[str, Any]] = []
    axis_points = _axis_points_from_metadata(metadata)
    if len(axis_points) >= 2:
        polylines.append(
            {
                "points": [[x / scale, y / scale] for x, y in axis_points],
                "color": "#00aee6",
                "width": 3,
                "dash": True,
            }
        )
    stars: list[dict[str, Any]] = []
    p7v = _xy({"x": measurements.get("P7V_compressed_virtual_x"), "y": measurements.get("P7V_compressed_virtual_y")})
    if p7v is not None:
        stars.append(
            {
                "x": p7v[0] / scale,
                "y": p7v[1] / scale,
                "label": "P7V",
                "color": "#9b5cff",
                "outline": "#ffffff",
                "radius": 6,
            }
        )
    return display, scale, point_payloads, lines, polylines, stars


def update_formal_points_from_editor(
    formal_points: Mapping[str, Any],
    editor_value: Mapping[str, Any] | None,
    *,
    display_to_original_scale: float,
) -> tuple[dict[str, list[float]], dict[str, Any] | None]:
    """Apply one drag event returned by the formal editor."""
    updated = {
        key: [float(_xy(value)[0]), float(_xy(value)[1])]
        for key, value in formal_points.items()
        if _xy(value) is not None
    }
    if not isinstance(editor_value, Mapping):
        return updated, None
    if editor_value.get("event") == "apply_changes":
        edited_points = editor_value.get("points", [])
        changed_names: list[str] = []
        if isinstance(edited_points, list):
            for point in edited_points:
                if not isinstance(point, Mapping):
                    continue
                name = str(point.get("name") or "")
                if name not in updated:
                    continue
                try:
                    x = float(point.get("x")) * display_to_original_scale
                    y = float(point.get("y")) * display_to_original_scale
                except (TypeError, ValueError):
                    continue
                old = updated.get(name)
                if old is None or abs(old[0] - x) > 1e-6 or abs(old[1] - y) > 1e-6:
                    changed_names.append(name)
                updated[name] = [x, y]
        override = {
            "xy": [],
            "modified_by_user": True,
            "modified_at": datetime.now().isoformat(timespec="seconds"),
            "changed_points": changed_names,
            "measurements": editor_value.get("measurements", {}),
        }
        return updated, {"point_name": "applied_formal_editor_points", **override}
    if editor_value.get("event") != "point_dragged":
        return updated, None
    return updated, None


def simplified_measurement_row(
    source: Mapping[str, Any],
    *,
    image_name: str = "",
    specimen_id: str = "",
    weight_g: Any = "",
    notes: str = "",
) -> dict[str, Any]:
    """Map full internal measurement fields to the formal V1.0 export view."""
    row = dict(source)
    status = row.get("measurement_status")
    if not status:
        status = "需要复核" if bool(row.get("needs_review") or row.get("measurements_needs_review")) else "已生成"
    exported_weight = row.get("weight_g", "")
    try:
        if pd.isna(exported_weight):
            exported_weight = ""
    except (TypeError, ValueError):
        pass
    if exported_weight in ("", None) and weight_g not in ("", None):
        exported_weight = weight_g
    return {
        "image_name": safe_name(str(row.get("image_name") or image_name)),
        "specimen_id": str(row.get("specimen_id") or specimen_id or ""),
        "weight_g": exported_weight,
        "TL_compressed_virtual_mm": row.get("TL_compressed_virtual_mm", ""),
        "TL_open_projection_mm": row.get("TL_open_projection_mm", row.get("TL_final_mm", "")),
        "SL_mm": row.get("SL_mm", row.get("SL_final_mm", row.get("SL_straight_mm", ""))),
        "FL_mm": row.get("FL_mm", row.get("fork_length_mm", "")),
        "body_depth_mm": row.get("body_depth_mm", ""),
        "head_length_mm": row.get("head_length_mm", row.get("head_length_straight_mm", "")),
        "snout_length_mm": row.get("snout_length_mm", row.get("snout_length_straight_mm", "")),
        "caudal_peduncle_length_mm": row.get(
            "caudal_peduncle_length_mm",
            row.get("caudal_peduncle_length_straight_mm", ""),
        ),
        "caudal_peduncle_depth_mm": row.get("caudal_peduncle_depth_mm", ""),
        "measurement_status": status,
        "notes": str(row.get("notes") or notes or ""),
    }


def simplified_measurement_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return the formal export columns from any internal measurement table."""
    if df.empty:
        return pd.DataFrame(columns=SIMPLIFIED_EXPORT_COLUMNS)
    rows = [simplified_measurement_row(record) for record in df.to_dict(orient="records")]
    return pd.DataFrame(rows, columns=SIMPLIFIED_EXPORT_COLUMNS)


def available_measurement_tables(project_root: Path) -> dict[str, Path]:
    """Known read-only measurement tables for the Results page."""
    candidates = {
        "v0.6.4 正式分析数据": project_root / "results" / "batch_measurement_v0.6.4_stable" / "final_analysis_dataset.csv",
        "v0.6.4 批量测量表": project_root / "results" / "batch_measurement_v0.6.4_stable" / "batch_measurements.csv",
        "5.24 新批次确认数据": project_root / "results" / "realworld_review_5_24_v0.6.5" / "final_corrected_dataset.csv",
    }
    formal_measurements_dir = project_root / "results" / "formal_v1_user_outputs" / "measurements"
    if formal_measurements_dir.exists():
        for path in sorted(formal_measurements_dir.glob("*_measurement.csv"), reverse=True):
            candidates[f"正式 V1.0 单鱼输出 - {path.stem}"] = path
    return {label: path for label, path in candidates.items() if path.exists()}


def read_measurement_table(path: Path) -> pd.DataFrame:
    """Read a CSV table without mutating it."""
    return pd.read_csv(path)
