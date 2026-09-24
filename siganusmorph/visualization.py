"""Drawing helpers for review images."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .axis_utils import AXIS_MODE_SPLINE, axis_mode_label, chord_length_cubic_spline_points
from .config import BODY_AXIS_POINT_ORDER, KEYPOINT_BY_NAME, KEYPOINT_DEFS, MEASUREMENT_DEFS
from .image_utils import ensure_rgb


Point = Mapping[str, float]

LINE_COLORS = {
    "TL_straight_axis_mm": (230, 57, 70),
    "SL_straight_mm": (255, 183, 3),
    "body_depth_mm": (255, 183, 3),
    "head_length_straight_mm": (255, 183, 3),
    "snout_length_straight_mm": (255, 183, 3),
    "caudal_peduncle_length_straight_mm": (255, 183, 3),
    "caudal_peduncle_depth_mm": (255, 183, 3),
}
BODY_DEPTH_GEOMETRY_SOURCE_V066 = "body_midline_normal_trunk_boundary_max_width"
BODY_DEPTH_GEOMETRY_VERSION_V066 = "v0.6.9_morphometric_definition_fix"
PEDUNCLE_DEPTH_GEOMETRY_SOURCE_V068 = "P4_P5_axis_normal_min_width"
PEDUNCLE_DEPTH_GEOMETRY_VERSION_V068 = "v0.6.8_peduncle_axis_normals"


def _point_xy(point: Point) -> tuple[float, float]:
    return float(point["x"]), float(point["y"])


def _font(size: int = 16) -> ImageFont.ImageFont:
    candidates = (
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "arial.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    fill: tuple[int, int, int] = (255, 255, 255),
) -> None:
    x, y = xy
    font = _font(16)
    outline = (0, 0, 0)
    for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        draw.text((x + ox, y + oy), text, font=font, fill=outline)
    draw.text((x, y), text, font=font, fill=fill)


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    p1: tuple[float, float],
    p2: tuple[float, float],
    fill: tuple[int, int, int],
    width: int,
    dash_length: float,
) -> None:
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    distance = float((dx * dx + dy * dy) ** 0.5)
    if distance <= 0:
        return
    ux = dx / distance
    uy = dy / distance
    position = 0.0
    draw_dash = True
    while position < distance:
        next_position = min(position + dash_length, distance)
        if draw_dash:
            start = (x1 + ux * position, y1 + uy * position)
            end = (x1 + ux * next_position, y1 + uy * next_position)
            draw.line((start[0], start[1], end[0], end[1]), fill=fill, width=width)
        draw_dash = not draw_dash
        position = next_position


def _draw_star(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    radius: int,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
) -> None:
    x, y = xy
    points = [
        (x, y - radius),
        (x + radius * 0.32, y - radius * 0.32),
        (x + radius, y - radius * 0.28),
        (x + radius * 0.45, y + radius * 0.18),
        (x + radius * 0.62, y + radius),
        (x, y + radius * 0.55),
        (x - radius * 0.62, y + radius),
        (x - radius * 0.45, y + radius * 0.18),
        (x - radius, y - radius * 0.28),
        (x - radius * 0.32, y - radius * 0.32),
    ]
    draw.polygon(points, fill=fill, outline=outline)


def _derived_p7v(measurements: Mapping[str, Any] | None) -> Point | None:
    if not measurements:
        return None
    if measurements.get("p7v_valid") is False:
        return None
    derived_points = measurements.get("derived_points")
    if not isinstance(derived_points, Mapping):
        return None
    point = derived_points.get("P7V_caudal_fin_posterior_endpoint") or derived_points.get("caudal_fin_posterior_endpoint")
    return point if isinstance(point, Mapping) else None


def draw_keypoints_and_measurements(
    image: np.ndarray | Image.Image,
    keypoints: Mapping[str, Point],
    measurements: Mapping[str, Any] | None = None,
    image_name: str = "",
    specimen_id: str = "",
    scale_points: Sequence[Point] | None = None,
    axis_mode_selected: str = "",
) -> np.ndarray:
    """Draw keypoints, measurement lines, and optional scale points."""
    if isinstance(image, Image.Image):
        canvas = image.convert("RGB")
    else:
        canvas = Image.fromarray(ensure_rgb(image))

    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    radius = max(5, round(min(width, height) / 260))
    line_width = max(3, round(min(width, height) / 450))
    p7v = _derived_p7v(measurements)

    if "snout_tip" in keypoints and p7v is not None:
        x1, y1 = _point_xy(keypoints["snout_tip"])
        x2, y2 = _point_xy(p7v)
        color = LINE_COLORS["TL_straight_axis_mm"]
        draw.line((x1, y1, x2, y2), fill=color, width=line_width)
        mid = ((x1 + x2) / 2 + 6, (y1 + y2) / 2 + 6)
        if measurements and measurements.get("TL_straight_axis_mm") is not None:
            text = f"TL axis {float(measurements['TL_straight_axis_mm']):.1f} mm"
        else:
            text = "TL axis"
        _draw_label(draw, mid, text, fill=color)

    for measurement_key, label, point_a, point_b in MEASUREMENT_DEFS:
        if point_a not in keypoints or point_b not in keypoints:
            continue
        x1, y1 = _point_xy(keypoints[point_a])
        x2, y2 = _point_xy(keypoints[point_b])
        color = LINE_COLORS.get(measurement_key, (255, 255, 255))
        draw.line((x1, y1, x2, y2), fill=color, width=line_width)
        mid = ((x1 + x2) / 2 + 6, (y1 + y2) / 2 + 6)
        if measurements and measurement_key in measurements:
            text = f"{label} {measurements[measurement_key]:.1f} mm"
        else:
            text = label
        _draw_label(draw, mid, text, fill=color)

    axis_points = [keypoints[name] for name in BODY_AXIS_POINT_ORDER if name in keypoints]
    if len(axis_points) >= 2:
        axis_color = (0, 102, 255)
        dash_length = max(12, round(min(width, height) / 90))
        for point_a, point_b in zip(axis_points, axis_points[1:]):
            _draw_dashed_line(
                draw,
                _point_xy(point_a),
                _point_xy(point_b),
                fill=axis_color,
                width=line_width,
                dash_length=dash_length,
            )
        active_axis_mode = axis_mode_selected or str((measurements or {}).get("axis_mode_selected", ""))
        if active_axis_mode == AXIS_MODE_SPLINE and len(axis_points) >= 3:
            spline_points = chord_length_cubic_spline_points([_point_xy(point) for point in axis_points])
            draw.line(spline_points, fill=(0, 220, 220), width=line_width + 1)

    if "caudal_base_midpoint" in keypoints and p7v is not None:
        caudal_color = (245, 130, 32)
        dash_length = max(12, round(min(width, height) / 90))
        _draw_dashed_line(
            draw,
            _point_xy(keypoints["caudal_base_midpoint"]),
            _point_xy(p7v),
            fill=caudal_color,
            width=line_width,
            dash_length=dash_length,
        )

    selected_tip = str((measurements or {}).get("caudal_tip_selected", ""))
    for definition in KEYPOINT_DEFS:
        point = keypoints.get(definition.name)
        if point is None:
            continue
        x, y = _point_xy(point)
        fill = (230, 57, 70) if definition.code.startswith("P") else (46, 204, 113)
        if (selected_tip == "upper" and definition.name == "caudal_fin_upper_tip") or (
            selected_tip == "lower" and definition.name == "caudal_fin_lower_tip"
        ):
            draw.ellipse((x - radius - 5, y - radius - 5, x + radius + 5, y + radius + 5), outline=(255, 245, 0), width=4)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=(255, 255, 255), width=2)
        _draw_label(draw, (x + radius + 2, y - radius - 2), definition.code, fill=fill)

    if p7v is not None:
        x, y = _point_xy(p7v)
        _draw_star(draw, (x, y), radius + 8, fill=(255, 220, 40), outline=(120, 70, 160))
        _draw_label(draw, (x + radius + 12, y - radius - 6), "P7V", fill=(255, 220, 40))

    if scale_points:
        for index, point in enumerate(scale_points, start=1):
            x, y = _point_xy(point)
            draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=(255, 245, 157), outline=(0, 0, 0), width=2)
            _draw_label(draw, (x + radius + 2, y - radius - 2), f"S{index}", fill=(255, 245, 157))
        if len(scale_points) == 2:
            x1, y1 = _point_xy(scale_points[0])
            x2, y2 = _point_xy(scale_points[1])
            draw.line((x1, y1, x2, y2), fill=(255, 245, 157), width=line_width)

    title_parts = [part for part in (image_name, f"Specimen {specimen_id}" if specimen_id else "") if part]
    if title_parts:
        title = " | ".join(title_parts)
        title_font = _font(18)
        bbox = draw.textbbox((0, 0), title, font=title_font)
        draw.rectangle((8, 8, bbox[2] + 18, bbox[3] + 18), fill=(0, 0, 0))
        draw.text((13, 12), title, font=title_font, fill=(255, 255, 255))

    axis_mode_text = axis_mode_selected or str((measurements or {}).get("axis_mode_selected", ""))
    if axis_mode_text:
        metric_parts = [f"Axis mode: {axis_mode_label(axis_mode_text)}"]
        if measurements:
            sl_final = measurements.get("SL_final_mm")
            tl_final = measurements.get("TL_final_mm")
            curvature = measurements.get("curvature_index_spline" if axis_mode_text == AXIS_MODE_SPLINE else "curvature_index_polyline")
            if sl_final is not None:
                metric_parts.append(f"SL final {float(sl_final):.1f} mm")
            if tl_final is not None and measurements.get("p7v_valid") is not False:
                metric_parts.append(f"TL final {float(tl_final):.1f} mm")
            caudal_tip = measurements.get("caudal_tip_selected")
            caudal_extension = measurements.get("caudal_extension_axis_mm")
            tl_curve = measurements.get("TL_curve_mm")
            if caudal_tip:
                metric_parts.append(f"caudal {caudal_tip}")
            if caudal_extension is not None and measurements.get("p7v_valid") is not False:
                metric_parts.append(f"tail ext {float(caudal_extension):.1f} mm")
            if tl_curve is not None and measurements.get("p7v_valid") is not False:
                metric_parts.append(f"TL curve {float(tl_curve):.1f} mm")
            if curvature is not None:
                metric_parts.append(f"curvature {float(curvature):.3f}")
        text = " | ".join(metric_parts)
        font = _font(16)
        bbox = draw.textbbox((0, 0), text, font=font)
        y0 = 34 if title_parts else 8
        draw.rectangle((8, y0, bbox[2] + 18, y0 + bbox[3] + 10), fill=(0, 0, 0))
        draw.text((13, y0 + 4), text, font=font, fill=(0, 220, 220))

    return np.asarray(canvas)


def keypoint_dict_from_list(points: Sequence[Mapping[str, float | str]]) -> dict[str, dict[str, float]]:
    """Convert app session keypoint list to a name-indexed dictionary."""
    return {
        str(point["name"]): {"x": float(point["x"]), "y": float(point["y"])}
        for point in points
        if str(point.get("name", "")) in KEYPOINT_BY_NAME
    }


def draw_enhanced_preannotation_overlay(
    image: np.ndarray | Image.Image,
    preannotation_metadata: Mapping[str, Any] | None,
) -> np.ndarray:
    """Draw raw model points, mask suggestions, and QC flags over an image."""
    if not preannotation_metadata:
        return np.asarray(Image.fromarray(ensure_rgb(image))) if not isinstance(image, Image.Image) else np.asarray(image.convert("RGB"))
    if isinstance(image, Image.Image):
        canvas = image.convert("RGB")
    else:
        canvas = Image.fromarray(ensure_rgb(image))
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    radius = max(5, round(min(width, height) / 300))

    raw_points = preannotation_metadata.get("model_keypoints_raw", {})
    heatmap_points = preannotation_metadata.get("heatmap_keypoints", {})
    v01_heatmap_points = preannotation_metadata.get("v01_heatmap_keypoints", {})
    v05_heatmap_points = preannotation_metadata.get("v05_heatmap_keypoints", {})
    v034_points = preannotation_metadata.get("v034_keypoints", {})
    v041_points = preannotation_metadata.get("v041_hybrid_keypoints", {})
    v06_points = preannotation_metadata.get("v06_keypoints", {})
    hybrid_points = preannotation_metadata.get("hybrid_keypoints", {})
    suggestions = preannotation_metadata.get("mask_suggestions", {})
    geometric_suggestions = preannotation_metadata.get("geometric_suggestions", {})
    local_normal_suggestions = preannotation_metadata.get("local_normal_suggestions", {})
    local_normal = preannotation_metadata.get("local_normal_measurements", {})
    body_depth_geometry = preannotation_metadata.get("body_depth_geometry", {})
    peduncle_depth_geometry = preannotation_metadata.get("peduncle_depth_geometry", {})
    operculum_qc = preannotation_metadata.get("operculum_qc", {})
    curvature_qc = preannotation_metadata.get("curvature_qc", {})
    measurement_axes = preannotation_metadata.get("measurement_axes", {})
    corrected_points = preannotation_metadata.get("corrected_keypoints", {})
    qc = preannotation_metadata.get("qc_results", {})
    hybrid_qc = preannotation_metadata.get("hybrid_qc_results", {})
    tail_qc = preannotation_metadata.get("tail_geometry_qc", {})
    tail_qc_v065 = preannotation_metadata.get("tail_qc", {})
    p6_gap_derivation = preannotation_metadata.get("P6_gap_derivation", {})
    point_sources = preannotation_metadata.get("point_sources", {})
    comparison = preannotation_metadata.get("comparison_preannotation", {})
    if not isinstance(raw_points, Mapping):
        raw_points = {}
    if not isinstance(heatmap_points, Mapping):
        heatmap_points = {}
    if not isinstance(v01_heatmap_points, Mapping):
        v01_heatmap_points = {}
    if not isinstance(v05_heatmap_points, Mapping):
        v05_heatmap_points = {}
    if not isinstance(v034_points, Mapping):
        v034_points = {}
    if not isinstance(v041_points, Mapping):
        v041_points = {}
    if not isinstance(v06_points, Mapping):
        v06_points = {}
    if not isinstance(hybrid_points, Mapping):
        hybrid_points = {}
    if not isinstance(suggestions, Mapping):
        suggestions = {}
    if not isinstance(geometric_suggestions, Mapping):
        geometric_suggestions = {}
    if not isinstance(local_normal_suggestions, Mapping):
        local_normal_suggestions = {}
    if not isinstance(local_normal, Mapping):
        local_normal = {}
    if not isinstance(body_depth_geometry, Mapping):
        body_depth_geometry = {}
    if body_depth_geometry and not (
        body_depth_geometry.get("body_depth_geometry_version") == BODY_DEPTH_GEOMETRY_VERSION_V066
        or body_depth_geometry.get("body_depth_source") == BODY_DEPTH_GEOMETRY_SOURCE_V066
    ):
        body_depth_geometry = {}
    if not isinstance(peduncle_depth_geometry, Mapping):
        peduncle_depth_geometry = {}
    if peduncle_depth_geometry and not (
        peduncle_depth_geometry.get("peduncle_depth_geometry_version") == PEDUNCLE_DEPTH_GEOMETRY_VERSION_V068
        or peduncle_depth_geometry.get("peduncle_depth_source") == PEDUNCLE_DEPTH_GEOMETRY_SOURCE_V068
    ):
        peduncle_depth_geometry = {}
    if not isinstance(curvature_qc, Mapping):
        curvature_qc = {}
    if not isinstance(operculum_qc, Mapping):
        operculum_qc = {}
    if not isinstance(measurement_axes, Mapping):
        measurement_axes = {}
    if not isinstance(corrected_points, Mapping):
        corrected_points = {}
    if not isinstance(qc, Mapping):
        qc = {}
    if not isinstance(hybrid_qc, Mapping):
        hybrid_qc = {}
    if not isinstance(tail_qc, Mapping):
        tail_qc = {}
    if not isinstance(tail_qc_v065, Mapping):
        tail_qc_v065 = {}
    if not isinstance(p6_gap_derivation, Mapping):
        p6_gap_derivation = {}
    if not isinstance(point_sources, Mapping):
        point_sources = {}
    if not isinstance(comparison, Mapping):
        comparison = {}
    flagged = qc.get("flagged_keypoints", {})
    if not isinstance(flagged, Mapping):
        flagged = {}
    show_heatmap = bool(preannotation_metadata.get("show_heatmap_points", False))
    show_v034 = bool(preannotation_metadata.get("show_v034_points", False))
    show_suggestions = bool(preannotation_metadata.get("show_mask_suggestions", False))
    show_qc = bool(preannotation_metadata.get("show_qc_warnings", True))
    show_v01_heatmap = bool(preannotation_metadata.get("show_v01_heatmap_points", False))
    show_v05_heatmap = bool(preannotation_metadata.get("show_v05_heatmap_points", False))
    show_v041 = bool(preannotation_metadata.get("show_v041_points", False))
    show_v06_selected = bool(preannotation_metadata.get("show_v06_selected_points", False))
    show_point_source_labels = bool(preannotation_metadata.get("show_point_source_labels", False))
    show_local_normal = bool(preannotation_metadata.get("show_local_normal_measurement_suggestions", True))
    show_width_profiles = bool(preannotation_metadata.get("show_width_profiles", True))
    show_curvature_qc = bool(preannotation_metadata.get("show_curvature_qc", True))
    show_model_axis = bool(preannotation_metadata.get("show_model_axis", False))
    show_body_midline_axis = bool(preannotation_metadata.get("show_body_midline_axis", False))
    show_geometric_c_points = bool(preannotation_metadata.get("show_geometric_c_points", True))
    show_body_midline_contour = bool(preannotation_metadata.get("show_body_midline_contour", True))
    show_dual_axis = bool(preannotation_metadata.get("show_dual_axis_comparison", True))
    show_selected_axis_only = bool(preannotation_metadata.get("show_selected_measurement_axis_only", False))
    show_p6_geometry_qc = bool(preannotation_metadata.get("show_p6_geometry_qc", False))
    show_p6_fallback = bool(preannotation_metadata.get("show_p6_fallback_suggestion", False))
    show_geometric_body_depth = bool(preannotation_metadata.get("show_geometric_body_depth", False))
    show_body_depth_width_profile = bool(preannotation_metadata.get("show_body_depth_width_profile", False))
    show_body_depth_comparison_labels = bool(preannotation_metadata.get("show_body_depth_comparison_labels", False))
    show_geometric_peduncle_depth = bool(preannotation_metadata.get("show_geometric_peduncle_depth", False))
    show_peduncle_depth_width_profile = bool(preannotation_metadata.get("show_peduncle_depth_width_profile", False))
    show_peduncle_depth_comparison_labels = bool(preannotation_metadata.get("show_peduncle_depth_comparison_labels", False))
    show_fin_suppression_regions = bool(preannotation_metadata.get("show_fin_suppression_regions", False))
    show_body_depth_boundary_contour = bool(preannotation_metadata.get("show_body_depth_boundary_contour", False))
    show_body_trunk_contours = bool(preannotation_metadata.get("show_body_trunk_contours", False))
    show_p3_operculum_suggestion = bool(preannotation_metadata.get("show_p3_operculum_suggestion", False))
    show_tail_fork_gap_region = bool(preannotation_metadata.get("show_tail_fork_gap_region", False))
    show_current_body_depth_line = bool(preannotation_metadata.get("show_current_body_depth_line", False))
    show_current_peduncle_depth_line = bool(preannotation_metadata.get("show_current_peduncle_depth_line", False))
    show_compressed_tail_p7v = bool(preannotation_metadata.get("show_compressed_tail_p7v", False))
    show_open_projection_p7v = bool(preannotation_metadata.get("show_open_projection_p7v", False))
    show_tail_folding_arc_reference = bool(preannotation_metadata.get("show_tail_folding_arc_reference", False))
    show_tl_comparison_labels = bool(preannotation_metadata.get("show_tl_comparison_labels", False))
    advanced_overlay_mode = bool(preannotation_metadata.get("advanced_overlay_mode", False))
    missing_overlay_messages: list[str] = []

    def draw_point(payload: Any, label: str, fill: tuple[int, int, int], outline: tuple[int, int, int]) -> None:
        if not isinstance(payload, (list, tuple)) or len(payload) < 2:
            return
        x, y = float(payload[0]), float(payload[1])
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=2)
        _draw_label(draw, (x + radius + 2, y - radius - 2), label, fill=fill)

    def draw_hollow(payload: Any, label: str, outline: tuple[int, int, int], width_px: int = 3) -> None:
        if not isinstance(payload, (list, tuple)) or len(payload) < 2:
            return
        x, y = float(payload[0]), float(payload[1])
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=outline, width=width_px)
        _draw_label(draw, (x + radius + 2, y - radius - 2), label, fill=outline)

    def xy_tuple(payload: Any) -> tuple[float, float] | None:
        if isinstance(payload, Mapping):
            payload = (payload.get("x"), payload.get("y"))
        if isinstance(payload, (list, tuple)) and len(payload) >= 2 and payload[0] is not None and payload[1] is not None:
            try:
                return float(payload[0]), float(payload[1])
            except (TypeError, ValueError):
                return None
        return None

    def current_working_point(key: str) -> Any:
        return (
            corrected_points.get(key)
            or v06_points.get(key)
            or hybrid_points.get(key)
            or v041_points.get(key)
            or raw_points.get(key)
        )

    def current_body_depth_mm() -> float | None:
        for container in (
            preannotation_metadata,
            preannotation_metadata.get("measurements", {}) if isinstance(preannotation_metadata.get("measurements", {}), Mapping) else {},
            preannotation_metadata.get("measurement_axes", {}) if isinstance(preannotation_metadata.get("measurement_axes", {}), Mapping) else {},
        ):
            if not isinstance(container, Mapping):
                continue
            for key in ("body_depth_mm", "body_depth_selected_mm", "body_depth_current_mm"):
                value = container.get(key)
                if value not in ("", None):
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        pass
        p8 = xy_tuple(current_working_point("P8_body_depth_dorsal"))
        p9 = xy_tuple(current_working_point("P9_body_depth_ventral"))
        if p8 and p9:
            try:
                mm_per_pixel = float(preannotation_metadata.get("mm_per_pixel", 0.1) or 0.1)
            except (TypeError, ValueError):
                mm_per_pixel = 0.1
            return math.hypot(p8[0] - p9[0], p8[1] - p9[1]) * mm_per_pixel
        return None

    def current_peduncle_depth_mm() -> float | None:
        for container in (
            preannotation_metadata,
            preannotation_metadata.get("measurements", {}) if isinstance(preannotation_metadata.get("measurements", {}), Mapping) else {},
            preannotation_metadata.get("measurement_axes", {}) if isinstance(preannotation_metadata.get("measurement_axes", {}), Mapping) else {},
        ):
            if not isinstance(container, Mapping):
                continue
            for key in ("caudal_peduncle_depth_mm", "peduncle_depth_selected_mm", "peduncle_depth_current_mm"):
                value = container.get(key)
                if value not in ("", None):
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        pass
        p10 = xy_tuple(current_working_point("P10_peduncle_depth_dorsal"))
        p11 = xy_tuple(current_working_point("P11_peduncle_depth_ventral"))
        if p10 and p11:
            try:
                mm_per_pixel = float(preannotation_metadata.get("mm_per_pixel", 0.1) or 0.1)
            except (TypeError, ValueError):
                mm_per_pixel = 0.1
            return math.hypot(p10[0] - p11[0], p10[1] - p11[1]) * mm_per_pixel
        return None

    def draw_current_section_line(a_key: str, b_key: str, label: str, color: tuple[int, int, int]) -> None:
        p_a = xy_tuple(current_working_point(a_key))
        p_b = xy_tuple(current_working_point(b_key))
        if p_a and p_b:
            draw.line((p_a[0], p_a[1], p_b[0], p_b[1]), fill=color, width=max(3, radius // 2 + 1))
            _draw_label(draw, ((p_a[0] + p_b[0]) / 2 + radius + 3, (p_a[1] + p_b[1]) / 2 + radius + 3), label, fill=color)
        else:
            missing_overlay_messages.append(f"No current {label} endpoints available.")

    if show_current_body_depth_line:
        draw_current_section_line("P8_body_depth_dorsal", "P9_body_depth_ventral", "current P8-P9", (255, 190, 0))

    if show_current_peduncle_depth_line:
        draw_current_section_line("P10_peduncle_depth_dorsal", "P11_peduncle_depth_ventral", "current P10-P11", (255, 170, 60))

    if show_v01_heatmap:
        for key, point in v01_heatmap_points.items():
            code = str(key).split("_", 1)[0]
            draw_hollow(point, f"{code}v01", outline=(160, 90, 255), width_px=3)

    if show_v05_heatmap:
        for key, point in v05_heatmap_points.items():
            code = str(key).split("_", 1)[0]
            draw_hollow(point, f"{code}v05", outline=(255, 80, 180), width_px=3)

    if show_v041:
        draw_payload = v041_points or hybrid_points or v034_points
        for key, point in draw_payload.items():
            code = str(key).split("_", 1)[0]
            draw_hollow(point, f"{code}v41", outline=(255, 140, 0), width_px=3)

    if show_v06_selected:
        draw_payload = corrected_points or v06_points or hybrid_points
        for key, point in draw_payload.items():
            code = str(key).split("_", 1)[0]
            source = str(point_sources.get(key, ""))
            label = f"{code}:{source}" if show_point_source_labels and source else code
            draw_point(point, label, fill=(230, 35, 45), outline=(255, 255, 255))

    if show_v034:
        draw_payload = v034_points or raw_points
        for key, point in draw_payload.items():
            code = str(key).split("_", 1)[0]
            draw_hollow(point, f"{code}v34", outline=(255, 140, 0), width_px=3)
    elif raw_points and not heatmap_points:
        for key, point in raw_points.items():
            code = str(key).split("_", 1)[0]
            draw_point(point, f"{code}m", fill=(255, 140, 0), outline=(30, 30, 30))

    if show_heatmap:
        for key, point in heatmap_points.items():
            code = str(key).split("_", 1)[0]
            draw_hollow(point, f"{code}h", outline=(210, 80, 255), width_px=3)

    combined_suggestions = dict(suggestions)
    combined_suggestions.update(geometric_suggestions)
    combined_suggestions.update(local_normal_suggestions)
    if show_suggestions:
        for key, point in combined_suggestions.items():
            code = str(key).split("_", 1)[0]
            draw_point(point, f"{code}s", fill=(0, 220, 220), outline=(0, 60, 80))

    def draw_refined_section(payload: Any, upper_name: str, lower_name: str, label: str) -> None:
        if not isinstance(payload, Mapping):
            return
        upper = payload.get(upper_name)
        lower = payload.get(lower_name)
        if not (
            isinstance(upper, (list, tuple))
            and isinstance(lower, (list, tuple))
            and len(upper) >= 2
            and len(lower) >= 2
        ):
            return
        p1 = (float(upper[0]), float(upper[1]))
        p2 = (float(lower[0]), float(lower[1]))
        if show_width_profiles:
            draw.line((p1[0], p1[1], p2[0], p2[1]), fill=(0, 210, 255), width=max(3, radius // 2 + 1))
        draw_point(upper, label.split("/")[0], fill=(0, 210, 255), outline=(0, 60, 90))
        draw_point(lower, label.split("/")[1], fill=(0, 210, 255), outline=(0, 60, 90))
        center = payload.get("section_center")
        if isinstance(center, (list, tuple)) and len(center) >= 2:
            _draw_label(draw, (float(center[0]) + radius + 4, float(center[1]) + radius + 4), label, fill=(0, 210, 255))

    if show_local_normal:
        body_local = local_normal.get("body_depth") if isinstance(local_normal.get("body_depth"), Mapping) else {}
        ped_local = local_normal.get("peduncle_depth") if isinstance(local_normal.get("peduncle_depth"), Mapping) else {}
        if not (show_geometric_body_depth and body_depth_geometry):
            draw_refined_section(body_local, "P8_refined", "P9_refined", "P8n/P9n")
        draw_refined_section(ped_local, "P10_refined", "P11_refined", "P10n/P11n")
        axis_curve = body_local.get("axis_curve") if isinstance(body_local, Mapping) else {}
        sampled = axis_curve.get("sampled_points", []) if isinstance(axis_curve, Mapping) else []
        if show_width_profiles and isinstance(sampled, list) and len(sampled) >= 2:
            for p_a, p_b in zip(sampled[::8], sampled[8::8]):
                if isinstance(p_a, (list, tuple)) and isinstance(p_b, (list, tuple)) and len(p_a) >= 2 and len(p_b) >= 2:
                    _draw_dashed_line(
                        draw,
                        (float(p_a[0]), float(p_a[1])),
                        (float(p_b[0]), float(p_b[1])),
                        fill=(0, 102, 255),
                        width=max(2, radius // 3),
                        dash_length=max(8, round(min(width, height) / 120)),
                    )

    body_depth_requested = (
        show_geometric_body_depth
        or show_body_depth_width_profile
        or show_body_depth_comparison_labels
        or show_body_depth_boundary_contour
        or show_body_trunk_contours
        or show_fin_suppression_regions
    )
    if body_depth_requested and not body_depth_geometry:
        missing_overlay_messages.append("No body_depth_geometry available.")

    if body_depth_requested and body_depth_geometry:
        def draw_polyline(points: Any, color: tuple[int, int, int], width_px: int = 2) -> None:
            if not isinstance(points, list) or len(points) < 2:
                return
            pts: list[tuple[float, float]] = []
            for point in points:
                xy = xy_tuple(point)
                if xy is not None:
                    pts.append(xy)
            if len(pts) >= 2:
                draw.line(pts, fill=color, width=width_px)

        if show_body_depth_boundary_contour:
            draw_polyline(body_depth_geometry.get("dorsal_body_depth_contour"), (115, 215, 255), max(2, radius // 3))
            draw_polyline(body_depth_geometry.get("ventral_body_depth_contour"), (115, 215, 255), max(2, radius // 3))
            if not body_depth_geometry.get("dorsal_body_depth_contour") and not body_depth_geometry.get("ventral_body_depth_contour"):
                missing_overlay_messages.append("No body-depth outer boundary contour available.")
        if show_body_trunk_contours:
            draw_polyline(body_depth_geometry.get("dorsal_trunk_contour"), (90, 235, 255), max(2, radius // 3))
            draw_polyline(body_depth_geometry.get("ventral_trunk_contour"), (90, 235, 255), max(2, radius // 3))
            if not body_depth_geometry.get("dorsal_trunk_contour") and not body_depth_geometry.get("ventral_trunk_contour"):
                missing_overlay_messages.append("No body trunk contours available.")
        if show_fin_suppression_regions:
            regions = body_depth_geometry.get("fin_suppression_regions", []) or []
            if not regions:
                missing_overlay_messages.append("No fin_suppression_regions available.")
            for region in body_depth_geometry.get("fin_suppression_regions", []) or []:
                if not isinstance(region, Mapping):
                    continue
                upper = xy_tuple(region.get("upper"))
                lower = xy_tuple(region.get("lower"))
                if upper and lower:
                    draw.line((upper[0], upper[1], lower[0], lower[1]), fill=(255, 220, 0), width=max(3, radius // 2))
        p8_geo = xy_tuple(body_depth_geometry.get("P8_geometric"))
        p9_geo = xy_tuple(body_depth_geometry.get("P9_geometric"))
        if show_geometric_body_depth and (not p8_geo or not p9_geo):
            missing_overlay_messages.append("No geometric P8/P9 data available.")
        if p8_geo:
            draw_point(p8_geo, "P8_geo", fill=(0, 135, 255), outline=(0, 40, 110))
        if p9_geo:
            draw_point(p9_geo, "P9_geo", fill=(0, 135, 255), outline=(0, 40, 110))
        p8_current = xy_tuple(current_working_point("P8_body_depth_dorsal"))
        p9_current = xy_tuple(current_working_point("P9_body_depth_ventral"))
        if p8_geo and p8_current:
            _draw_dashed_line(
                draw,
                p8_current,
                p8_geo,
                fill=(115, 215, 255),
                width=max(2, radius // 3),
                dash_length=max(6, round(min(width, height) / 160)),
            )
        if p9_geo and p9_current:
            _draw_dashed_line(
                draw,
                p9_current,
                p9_geo,
                fill=(115, 215, 255),
                width=max(2, radius // 3),
                dash_length=max(6, round(min(width, height) / 160)),
            )
        if show_body_depth_width_profile:
            if p8_geo and p9_geo:
                draw.line((p8_geo[0], p8_geo[1], p9_geo[0], p9_geo[1]), fill=(0, 230, 255), width=max(4, radius // 2 + 2))
        if show_body_depth_width_profile:
            selected = body_depth_geometry.get("body_depth_selected_section", {})
            center = body_depth_geometry.get("section_center")
            if isinstance(selected, Mapping) and not isinstance(center, (list, tuple)):
                center = selected.get("section_center")
            if isinstance(center, (list, tuple)) and len(center) >= 2:
                _draw_label(
                    draw,
                    (float(center[0]) + radius + 6, float(center[1]) + radius + 6),
                    "body-depth geometric max width",
                    fill=(0, 210, 255),
                )
        if show_body_depth_comparison_labels:
            geo_mm = body_depth_geometry.get("body_depth_geometric_mm")
            try:
                geo_mm_float = float(geo_mm) if geo_mm not in ("", None) else None
            except (TypeError, ValueError):
                geo_mm_float = None
            current_mm = current_body_depth_mm()
            diff = abs(current_mm - geo_mm_float) if current_mm is not None and geo_mm_float is not None else None
            qc_pass = body_depth_geometry.get("body_depth_geometric_qc_pass", "")
            reason = str(body_depth_geometry.get("body_depth_geometric_review_reason", "") or "")
            lines = []
            if current_mm is not None:
                lines.append(f"current body depth: {current_mm:.2f} mm")
            if geo_mm_float is not None:
                lines.append(f"geometric body depth: {geo_mm_float:.2f} mm")
            if diff is not None:
                lines.append(f"diff: {diff:.2f} mm")
            if qc_pass not in ("", None):
                lines.append(f"geo QC: {qc_pass}")
            if reason:
                lines.append(str(reason)[:72])
            if lines:
                _draw_label(draw, (18, 130), " | ".join(lines), fill=(0, 210, 255))

    peduncle_requested = show_geometric_peduncle_depth or show_peduncle_depth_width_profile or show_peduncle_depth_comparison_labels
    if peduncle_requested and not peduncle_depth_geometry:
        missing_overlay_messages.append("No peduncle_depth_geometry available.")

    if peduncle_requested and peduncle_depth_geometry:
        p10_geo = xy_tuple(peduncle_depth_geometry.get("P10_geometric"))
        p11_geo = xy_tuple(peduncle_depth_geometry.get("P11_geometric"))
        if show_geometric_peduncle_depth and (not p10_geo or not p11_geo):
            missing_overlay_messages.append("No geometric P10/P11 data available.")
        if p10_geo:
            draw_point(p10_geo, "P10_geo", fill=(0, 220, 180), outline=(0, 70, 70))
        if p11_geo:
            draw_point(p11_geo, "P11_geo", fill=(0, 220, 180), outline=(0, 70, 70))
        p10_current = xy_tuple(current_working_point("P10_peduncle_depth_dorsal"))
        p11_current = xy_tuple(current_working_point("P11_peduncle_depth_ventral"))
        if p10_geo and p10_current:
            _draw_dashed_line(
                draw,
                p10_current,
                p10_geo,
                fill=(105, 235, 200),
                width=max(2, radius // 3),
                dash_length=max(6, round(min(width, height) / 160)),
            )
        if p11_geo and p11_current:
            _draw_dashed_line(
                draw,
                p11_current,
                p11_geo,
                fill=(105, 235, 200),
                width=max(2, radius // 3),
                dash_length=max(6, round(min(width, height) / 160)),
            )
        if show_peduncle_depth_width_profile and p10_geo and p11_geo:
            draw.line((p10_geo[0], p10_geo[1], p11_geo[0], p11_geo[1]), fill=(0, 245, 190), width=max(4, radius // 2 + 2))
            selected = peduncle_depth_geometry.get("peduncle_depth_selected_section", {})
            center = selected.get("section_center") if isinstance(selected, Mapping) else None
            if isinstance(center, (list, tuple)) and len(center) >= 2:
                _draw_label(
                    draw,
                    (float(center[0]) + radius + 6, float(center[1]) + radius + 6),
                    "peduncle geometric min width",
                    fill=(0, 245, 190),
                )
        if show_peduncle_depth_comparison_labels:
            geo_mm = peduncle_depth_geometry.get("peduncle_depth_geometric_mm")
            try:
                geo_mm_float = float(geo_mm) if geo_mm not in ("", None) else None
            except (TypeError, ValueError):
                geo_mm_float = None
            current_mm = current_peduncle_depth_mm()
            diff = abs(current_mm - geo_mm_float) if current_mm is not None and geo_mm_float is not None else None
            qc_pass = peduncle_depth_geometry.get("peduncle_depth_geometric_qc_pass", "")
            reason = str(peduncle_depth_geometry.get("peduncle_depth_geometric_review_reason", "") or "")
            lines = []
            if current_mm is not None:
                lines.append(f"current peduncle depth: {current_mm:.2f} mm")
            if geo_mm_float is not None:
                lines.append(f"geometric peduncle depth: {geo_mm_float:.2f} mm")
            if diff is not None:
                lines.append(f"diff: {diff:.2f} mm")
            if qc_pass not in ("", None):
                lines.append(f"geo QC: {qc_pass}")
            if reason:
                lines.append(str(reason)[:72])
            if lines:
                _draw_label(draw, (18, 156), " | ".join(lines), fill=(0, 245, 190))
        if peduncle_depth_geometry.get("peduncle_depth_review_suggested"):
            point = p10_geo or p11_geo
            if point:
                draw.ellipse((point[0] - radius - 8, point[1] - radius - 8, point[0] + radius + 8, point[1] + radius + 8), outline=(255, 230, 0), width=4)

    if (show_p3_operculum_suggestion or (show_qc and bool(operculum_qc.get("P3_needs_review", False)))) and not operculum_qc:
        missing_overlay_messages.append("No P3 operculum suggestion available.")

    if show_p3_operculum_suggestion or (show_qc and bool(operculum_qc.get("P3_needs_review", False))):
        p3_suggestion = xy_tuple(operculum_qc.get("P3_operculum_edge_suggestion"))
        p3_current = xy_tuple(current_working_point("P3_operculum_posterior"))
        if p3_suggestion:
            draw_point(p3_suggestion, "P3_edge", fill=(255, 80, 210), outline=(90, 0, 80))
        if p3_current and p3_suggestion:
            _draw_dashed_line(
                draw,
                p3_current,
                p3_suggestion,
                fill=(255, 150, 230),
                width=max(2, radius // 3),
                dash_length=max(6, round(min(width, height) / 160)),
            )
        if p3_current and bool(operculum_qc.get("P3_needs_review", False)):
            draw.ellipse((p3_current[0] - radius - 8, p3_current[1] - radius - 8, p3_current[0] + radius + 8, p3_current[1] + radius + 8), outline=(255, 230, 0), width=4)
            reason = str(operculum_qc.get("P3_review_reason", "") or "P3 operculum QC warning")
            _draw_label(draw, (p3_current[0] + radius + 8, p3_current[1] + radius + 8), reason[:60], fill=(255, 230, 0))
        if show_p3_operculum_suggestion and not p3_suggestion:
            missing_overlay_messages.append("No P3 operculum suggestion available.")

    def draw_axis_payload(axis_payload: Any, color: tuple[int, int, int], width_px: int, label: str = "") -> None:
        if not isinstance(axis_payload, Mapping):
            return
        points = axis_payload.get("axis_points", [])
        if not isinstance(points, list) or len(points) < 2:
            return
        pts = []
        for point in points:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                pts.append((float(point[0]), float(point[1])))
        if len(pts) < 2:
            return
        for a, b in zip(pts, pts[1:]):
            _draw_dashed_line(
                draw,
                a,
                b,
                fill=color,
                width=width_px,
                dash_length=max(8, round(min(width, height) / 110)),
            )
        if label:
            _draw_label(draw, (pts[0][0] + radius + 4, pts[0][1] + radius + 4), label, fill=color)

    def axis_points_from_payload(axis_payload: Any) -> list[tuple[float, float]]:
        if not isinstance(axis_payload, Mapping):
            return []
        points = axis_payload.get("axis_points", [])
        if not isinstance(points, list):
            return []
        parsed: list[tuple[float, float]] = []
        for point in points:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                parsed.append((float(point[0]), float(point[1])))
        return parsed

    def draw_sparse_points(points: Any, color: tuple[int, int, int], step: int = 10, radius_px: int = 2) -> None:
        if not isinstance(points, list):
            return
        for point in points[:: max(1, step)]:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                x, y = float(point[0]), float(point[1])
                draw.ellipse((x - radius_px, y - radius_px, x + radius_px, y + radius_px), fill=color)

    def draw_polyline_points(points: Any, color: tuple[int, int, int], width_px: int = 2) -> None:
        if not isinstance(points, list):
            return
        parsed = [
            (float(point[0]), float(point[1]))
            for point in points
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
        if len(parsed) >= 2:
            draw.line(parsed, fill=color, width=width_px)

    axis_requested = (
        show_model_axis
        or show_body_midline_axis
        or show_geometric_c_points
        or show_body_midline_contour
        or show_dual_axis
        or show_selected_axis_only
    )
    if axis_requested and not measurement_axes:
        missing_overlay_messages.append("No measurement_axes/model_axis available.")

    if measurement_axes:
        model_axis = measurement_axes.get("model_axis", {}) if isinstance(measurement_axes.get("model_axis", {}), Mapping) else {}
        body_axis = measurement_axes.get("body_midline_axis", {}) if isinstance(measurement_axes.get("body_midline_axis", {}), Mapping) else {}
        selected_axis = str(measurement_axes.get("selected_measurement_axis", "model_axis") or "model_axis")
        requested_axis = str(measurement_axes.get("measurement_axis_mode_request", "") or "")
        comparison_axis = measurement_axes.get("dual_axis_comparison", {}) if isinstance(measurement_axes.get("dual_axis_comparison", {}), Mapping) else {}
        body_points = body_axis.get("body_midline_points", []) if isinstance(body_axis, Mapping) else []
        if show_body_midline_contour and isinstance(body_axis, Mapping):
            draw_polyline_points(body_axis.get("dorsal_body_contour", []), (90, 210, 255), width_px=max(2, radius // 3))
            draw_polyline_points(body_axis.get("ventral_body_contour", []), (90, 210, 255), width_px=max(2, radius // 3))
            draw_sparse_points(body_points, (0, 105, 255), step=6, radius_px=max(2, radius // 5))
        if show_selected_axis_only:
            if selected_axis == "body_midline_axis":
                draw_axis_payload(body_axis, (0, 105, 255), max(5, radius // 2 + 2), "selected body axis")
            elif selected_axis == "model_axis":
                draw_axis_payload(model_axis, (255, 140, 0), max(5, radius // 2 + 2), "selected model axis")
        else:
            if show_model_axis:
                model_width = max(3, radius // 2)
                if selected_axis == "model_axis":
                    model_width = max(6, radius // 2 + 3)
                draw_axis_payload(model_axis, (255, 140, 0), model_width, "model axis" if selected_axis != "model_axis" else "selected model axis")
                if not model_axis:
                    missing_overlay_messages.append("No model_axis available.")
            if show_body_midline_axis:
                body_width = max(3, radius // 2)
                if selected_axis == "body_midline_axis":
                    body_width = max(6, radius // 2 + 3)
                draw_axis_payload(body_axis, (0, 105, 255), body_width, "body midline axis" if selected_axis != "body_midline_axis" else "selected body axis")
                if not body_axis:
                    missing_overlay_messages.append("No body_midline_axis available.")
            if selected_axis == "body_midline_axis" and not show_body_midline_axis:
                draw_axis_payload(body_axis, (0, 105, 255), max(6, radius // 2 + 3), "selected")
            elif selected_axis == "model_axis" and not show_model_axis:
                draw_axis_payload(model_axis, (255, 140, 0), max(6, radius // 2 + 3), "selected")
        if show_geometric_c_points:
            axis_pts = axis_points_from_payload(body_axis)
            if not axis_pts:
                missing_overlay_messages.append("No geometric C points/body_midline_axis available.")
            for index, label in ((1, "C1g"), (2, "C2g"), (3, "C3g"), (5, "C4g")):
                if index < len(axis_pts):
                    x, y = axis_pts[index]
                    draw.ellipse(
                        (x - radius - 1, y - radius - 1, x + radius + 1, y + radius + 1),
                        fill=(0, 105, 255),
                        outline=(255, 255, 255),
                        width=2,
                    )
                    _draw_label(draw, (x + radius + 3, y - radius - 3), label, fill=(0, 105, 255))
        if show_dual_axis and comparison_axis:
            y_text = 104
            selected = measurement_axes.get("selected_measurement_axis", "")
            tl_diff = comparison_axis.get("TL_curve_diff_mm", "")
            sl_diff = comparison_axis.get("SL_curve_diff_mm", "")
            disagreement = bool(comparison_axis.get("dual_axis_disagreement", False))
            review_reason = str(measurement_axes.get("measurement_axis_review_reason", "") or "")
            label = f"axis mode: {requested_axis} | selected: {selected} | TL diff {tl_diff} | SL diff {sl_diff}"
            fill = (255, 220, 0) if disagreement else (120, 210, 255)
            _draw_label(draw, (18, y_text), label, fill=fill)
            if disagreement or measurement_axes.get("measurements_needs_review"):
                warning = review_reason or "dual-axis measurement needs review"
                _draw_label(draw, (18, y_text + 24), warning, fill=(255, 80, 40) if measurement_axes.get("measurements_needs_review") else (255, 220, 0))
        elif show_dual_axis:
            missing_overlay_messages.append("No dual-axis comparison available.")

    comparison_requested = (
        show_heatmap
        or show_v034
        or show_suggestions
        or show_v01_heatmap
        or show_v05_heatmap
        or show_v041
        or show_v06_selected
    )

    # When comparing an already-confirmed image, show the v0.4 hybrid draft in
    # red without replacing the visible corrected labels.
    if comparison:
        comp_heatmap = comparison.get("heatmap_keypoints", {}) if isinstance(comparison.get("heatmap_keypoints", {}), Mapping) else {}
        comp_v01 = comparison.get("v01_heatmap_keypoints", {}) if isinstance(comparison.get("v01_heatmap_keypoints", {}), Mapping) else {}
        comp_v05 = comparison.get("v05_heatmap_keypoints", {}) if isinstance(comparison.get("v05_heatmap_keypoints", {}), Mapping) else {}
        comp_v034 = comparison.get("v034_keypoints", {}) if isinstance(comparison.get("v034_keypoints", {}), Mapping) else {}
        comp_v041 = comparison.get("v041_hybrid_keypoints", {}) if isinstance(comparison.get("v041_hybrid_keypoints", {}), Mapping) else {}
        comp_v06 = comparison.get("v06_keypoints", {}) if isinstance(comparison.get("v06_keypoints", {}), Mapping) else {}
        comp_sources = comparison.get("point_sources", {}) if isinstance(comparison.get("point_sources", {}), Mapping) else {}
        comp_suggestions = comparison.get("mask_suggestions", {}) if isinstance(comparison.get("mask_suggestions", {}), Mapping) else {}
        comp_geometric = comparison.get("geometric_suggestions", {}) if isinstance(comparison.get("geometric_suggestions", {}), Mapping) else {}
        comp_hybrid = comparison.get("hybrid_keypoints", {}) if isinstance(comparison.get("hybrid_keypoints", {}), Mapping) else comparison.get("corrected_keypoints", {})
        if show_v01_heatmap:
            for key, point in comp_v01.items():
                draw_hollow(point, f"{str(key).split('_', 1)[0]}v01", outline=(160, 90, 255), width_px=3)
        if show_v05_heatmap:
            for key, point in comp_v05.items():
                draw_hollow(point, f"{str(key).split('_', 1)[0]}v05", outline=(255, 80, 180), width_px=3)
        if show_v041:
            for key, point in (comp_v041 or comp_hybrid).items():
                draw_hollow(point, f"{str(key).split('_', 1)[0]}v41", outline=(255, 140, 0), width_px=3)
        if show_v06_selected:
            for key, point in (comp_v06 or comp_hybrid).items():
                code = str(key).split("_", 1)[0]
                source = str(comp_sources.get(key, ""))
                label = f"{code}:{source}" if show_point_source_labels and source else f"{code}v6"
                draw_hollow(point, label, outline=(245, 45, 55), width_px=4)
        if show_v034:
            for key, point in comp_v034.items():
                draw_hollow(point, f"{str(key).split('_', 1)[0]}v34", outline=(255, 140, 0), width_px=3)
        if show_heatmap:
            for key, point in comp_heatmap.items():
                draw_hollow(point, f"{str(key).split('_', 1)[0]}h", outline=(210, 80, 255), width_px=3)
        if show_suggestions:
            comp_s = dict(comp_suggestions)
            comp_s.update(comp_geometric)
            comp_local = comparison.get("local_normal_suggestions", {}) if isinstance(comparison.get("local_normal_suggestions", {}), Mapping) else {}
            comp_s.update(comp_local)
            for key, point in comp_s.items():
                draw_point(point, f"{str(key).split('_', 1)[0]}s", fill=(0, 220, 220), outline=(0, 60, 80))
        if isinstance(comp_hybrid, Mapping):
            for key, point in comp_hybrid.items():
                draw_hollow(point, f"{str(key).split('_', 1)[0]}hy", outline=(245, 45, 55), width_px=4)
    elif comparison_requested:
        if show_heatmap or show_v01_heatmap or show_v05_heatmap:
            missing_overlay_messages.append("No heatmap debug points available.")
        if show_v034 or show_v041 or show_v06_selected:
            missing_overlay_messages.append("No comparison preannotation points available.")
        if show_suggestions:
            missing_overlay_messages.append("No mask/geometric suggestions available.")

    tail_start = tail_qc.get("tail_axis_start")
    tail_end = tail_qc.get("tail_axis_end")
    if isinstance(tail_start, (list, tuple)) and isinstance(tail_end, (list, tuple)) and len(tail_start) >= 2 and len(tail_end) >= 2:
        _draw_dashed_line(
            draw,
            (float(tail_start[0]), float(tail_start[1])),
            (float(tail_end[0]), float(tail_end[1])),
            fill=(245, 130, 32),
            width=max(3, radius // 2),
            dash_length=max(10, round(min(width, height) / 100)),
        )
        _draw_label(draw, (float(tail_end[0]) + radius + 4, float(tail_end[1]) + radius + 4), "tail axis", fill=(245, 130, 32))

    if (show_tail_fork_gap_region or show_p6_fallback) and not (p6_gap_derivation or tail_qc_v065):
        missing_overlay_messages.append("No P6_gap_derivation available.")

    if show_tail_fork_gap_region and p6_gap_derivation:
        centerline = p6_gap_derivation.get("tail_gap_centerline", [])
        pts = [xy_tuple(point) for point in centerline] if isinstance(centerline, list) else []
        pts = [point for point in pts if point is not None]
        if len(pts) >= 2:
            draw.line(pts, fill=(120, 235, 255), width=max(3, radius // 2))
        tip = p6_gap_derivation.get("tail_gap_tip") or p6_gap_derivation.get("P6_geometric")
        draw_point(tip, "gap", fill=(120, 235, 255), outline=(0, 70, 90))
        if not pts and xy_tuple(tip) is None:
            missing_overlay_messages.append("No tail fork gap region available.")

    if show_p6_fallback and tail_qc_v065:
        fallback = (
            p6_gap_derivation.get("P6_geometric")
            or p6_gap_derivation.get("tail_gap_tip")
            or tail_qc_v065.get("P6_geometric")
            or tail_qc_v065.get("P6_fallback")
            or tail_qc_v065.get("P6_fallback_suggestion")
        )
        if fallback is None and tail_qc_v065.get("P6_fallback_x") not in ("", None):
            fallback = [tail_qc_v065.get("P6_fallback_x"), tail_qc_v065.get("P6_fallback_y")]
        draw_point(fallback, "P6_gap", fill=(0, 230, 210), outline=(0, 60, 80))
        if xy_tuple(fallback) is None:
            missing_overlay_messages.append("No P6 fallback suggestion available.")
        current_p6 = corrected_points.get("P6_caudal_fork_midpoint") or hybrid_points.get("P6_caudal_fork_midpoint") or v06_points.get("P6_caudal_fork_midpoint")
        fallback_xy = xy_tuple(fallback)
        current_xy = xy_tuple(current_p6)
        if fallback_xy and current_xy:
            _draw_dashed_line(
                draw,
                current_xy,
                fallback_xy,
                fill=(0, 220, 220),
                width=max(2, radius // 3),
                dash_length=max(6, round(min(width, height) / 160)),
            )
    elif show_p6_fallback:
        missing_overlay_messages.append("No P6 fallback suggestion available.")

    if show_p6_geometry_qc and tail_qc_v065 and tail_qc_v065.get("P6_geometry_qc_pass") is False:
        p6_point = corrected_points.get("P6_caudal_fork_midpoint") or hybrid_points.get("P6_caudal_fork_midpoint") or v06_points.get("P6_caudal_fork_midpoint")
        if isinstance(p6_point, (list, tuple)) and len(p6_point) >= 2:
            x, y = float(p6_point[0]), float(p6_point[1])
            draw.ellipse((x - radius - 10, y - radius - 10, x + radius + 10, y + radius + 10), outline=(255, 230, 0), width=5)
            reason = str(tail_qc_v065.get("P6_review_reason", "P6 geometry failed"))
            _draw_label(draw, (x + radius + 10, y + radius + 10), reason, fill=(255, 230, 0))
    elif show_p6_geometry_qc and not tail_qc_v065:
        missing_overlay_messages.append("No P6 geometry QC data available.")

    if point_sources.get("P7U_caudal_fin_upper_tip") == "mask_tail_rule" or point_sources.get("P7L_caudal_fin_lower_tip") == "mask_tail_rule":
        p7u = corrected_points.get("P7U_caudal_fin_upper_tip") or combined_suggestions.get("P7U_caudal_fin_upper_tip")
        p7l = corrected_points.get("P7L_caudal_fin_lower_tip") or combined_suggestions.get("P7L_caudal_fin_lower_tip")
        anchor = p7u if isinstance(p7u, (list, tuple)) else p7l
        if isinstance(anchor, (list, tuple)) and len(anchor) >= 2:
            _draw_label(draw, (float(anchor[0]) + radius + 8, float(anchor[1]) - radius * 3), "P7U/P7L from mask rule", fill=(0, 220, 220))

    if show_qc:
        for key in flagged:
            point = corrected_points.get(key) or hybrid_points.get(key) or combined_suggestions.get(key) or raw_points.get(key)
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                x, y = float(point[0]), float(point[1])
                draw.ellipse((x - radius - 8, y - radius - 8, x + radius + 8, y + radius + 8), outline=(255, 230, 0), width=5)

    p7v_valid = qc.get("p7v_valid", hybrid_qc.get("p7v_valid", None))
    p7v_payload = hybrid_qc.get("p7v", {}) if isinstance(hybrid_qc.get("p7v", {}), Mapping) else {}
    if p7v_valid is False:
        p7v = qc.get("p7v_suggestion") or tail_qc.get("p7v_suggestion") or p7v_payload.get("point")
        if isinstance(p7v, Mapping):
            p7v = (p7v.get("x"), p7v.get("y"))
        if isinstance(p7v, (list, tuple)) and len(p7v) >= 2:
            x, y = float(p7v[0]), float(p7v[1])
            size = radius + 12
            draw.line((x - size, y - size, x + size, y + size), fill=(255, 0, 0), width=5)
            draw.line((x - size, y + size, x + size, y - size), fill=(255, 0, 0), width=5)
            _draw_label(draw, (x + size + 2, y - size), "invalid P7V", fill=(255, 0, 0))
    else:
        p7v = qc.get("p7v_suggestion") or tail_qc.get("p7v_suggestion") or p7v_payload.get("point")
        if isinstance(p7v, Mapping):
            p7v = (p7v.get("x"), p7v.get("y"))
        if isinstance(p7v, (list, tuple)) and len(p7v) >= 2 and p7v[0] is not None and p7v[1] is not None:
            x, y = float(p7v[0]), float(p7v[1])
            _draw_star(draw, (x, y), radius + 8, fill=(210, 90, 255), outline=(70, 20, 110))
            _draw_label(draw, (x + radius + 10, y - radius), "P7V", fill=(210, 90, 255))

    if show_curvature_qc and curvature_qc:
        level = str(curvature_qc.get("curvature_qc_level", "") or "")
        index = curvature_qc.get("curvature_index")
        if level and level not in {"straight_or_mild", "unknown"}:
            label = f"curvature {level}"
            if index is not None:
                try:
                    label += f" ({float(index):.3f})"
                except (TypeError, ValueError):
                    pass
            fill = (255, 60, 60) if level == "high" else (255, 210, 0)
            _draw_label(draw, (18, 78), label, fill=fill)

    tl_payload = preannotation_metadata.get("compressed_tail_tl", {})
    if not isinstance(tl_payload, Mapping):
        tl_payload = {}
    if not tl_payload:
        tl_payload = preannotation_metadata
    compressed_requested = show_compressed_tail_p7v or show_open_projection_p7v or show_tail_folding_arc_reference or show_tl_comparison_labels
    if compressed_requested:
        p_open = xy_tuple(tl_payload.get("P7V_open_projection") or (
            tl_payload.get("P7V_open_projection_x"),
            tl_payload.get("P7V_open_projection_y"),
        ))
        p_comp = xy_tuple(tl_payload.get("P7V_compressed_virtual") or (
            tl_payload.get("P7V_compressed_virtual_x"),
            tl_payload.get("P7V_compressed_virtual_y"),
        ))
        p5 = xy_tuple(corrected_points.get("P5_caudal_base_midpoint") or corrected_points.get("caudal_base_midpoint"))
        p7u = xy_tuple(corrected_points.get("P7U_caudal_fin_upper_tip") or corrected_points.get("caudal_fin_upper_tip"))
        p7l = xy_tuple(corrected_points.get("P7L_caudal_fin_lower_tip") or corrected_points.get("caudal_fin_lower_tip"))
        if show_open_projection_p7v:
            if p_open is not None:
                _draw_star(draw, p_open, radius + 7, fill=(255, 178, 80), outline=(105, 55, 0))
                _draw_label(draw, (p_open[0] + radius + 10, p_open[1] - radius), "P7V_open", fill=(255, 178, 80))
            else:
                missing_overlay_messages.append("No P7V_open_projection available.")
        if show_compressed_tail_p7v:
            if p_comp is not None:
                _draw_star(draw, p_comp, radius + 8, fill=(80, 245, 100), outline=(0, 95, 20))
                _draw_label(draw, (p_comp[0] + radius + 10, p_comp[1] - radius), "P7V_comp", fill=(80, 245, 100))
            else:
                missing_overlay_messages.append("No P7V_compressed_virtual available.")
        if show_tail_folding_arc_reference:
            if p5 is not None:
                for tip, color in ((p7u, (120, 220, 120)), (p7l, (120, 220, 120))):
                    if tip is not None:
                        draw.line((p5[0], p5[1], tip[0], tip[1]), fill=color, width=max(2, radius // 3))
                if p_comp is not None:
                    draw.line((p5[0], p5[1], p_comp[0], p_comp[1]), fill=(80, 245, 100), width=max(3, radius // 2))
            else:
                missing_overlay_messages.append("No tail-lobe radius reference available.")
        if show_tl_comparison_labels:
            if tl_payload:
                open_tl = tl_payload.get("TL_open_projection_mm", "")
                comp_tl = tl_payload.get("TL_compressed_virtual_mm", "")
                diff = tl_payload.get("TL_difference_mm", "")
                pct = tl_payload.get("TL_difference_percent", "")
                upper_angle = tl_payload.get("upper_lobe_angle_to_axis_deg", tl_payload.get("upper_lobe_angle_deg", ""))
                lower_angle = tl_payload.get("lower_lobe_angle_to_axis_deg", tl_payload.get("lower_lobe_angle_deg", ""))
                inter_angle = tl_payload.get("inter_lobe_open_angle_deg", tl_payload.get("tail_open_angle_deg", ""))
                valid_qc = tl_payload.get("compressed_tail_tl_valid_qc_pass", tl_payload.get("compressed_tail_tl_qc_pass", ""))
                diff_flag = tl_payload.get("compressed_vs_projection_difference_flag", "")
                label = f"TL open={open_tl} comp={comp_tl} diff={diff}mm ({pct}%) U/L={upper_angle}/{lower_angle} inter={inter_angle} valid={valid_qc} diff_flag={diff_flag}"
                _draw_label(draw, (18, 102), label, fill=(80, 245, 100) if valid_qc in {True, 'True', 'true'} else (255, 220, 0))
            else:
                missing_overlay_messages.append("No TL comparison data available.")

    if (
        advanced_overlay_mode
        and show_qc
        and not flagged
        and not qc
        and not hybrid_qc
        and not tail_qc_v065
        and not operculum_qc
    ):
        missing_overlay_messages.append("No QC warning data available.")

    if missing_overlay_messages:
        deduped_messages = []
        for message in missing_overlay_messages:
            if message not in deduped_messages:
                deduped_messages.append(message)
        font_missing = _font(14)
        x_msg = 18
        y_msg = 108
        visible_messages = deduped_messages[:6]
        text_lines = ["Overlay data missing:"] + visible_messages
        if len(deduped_messages) > len(visible_messages):
            text_lines.append(f"... {len(deduped_messages) - len(visible_messages)} more")
        widths = [draw.textbbox((0, 0), line, font=font_missing)[2] for line in text_lines]
        line_h = max(16, draw.textbbox((0, 0), "Ag", font=font_missing)[3] + 4)
        box_w = min(max(widths) + 18, max(220, width - 36))
        box_h = line_h * len(text_lines) + 10
        draw.rectangle((x_msg - 6, y_msg - 6, x_msg - 6 + box_w, y_msg - 6 + box_h), fill=(0, 0, 0), outline=(255, 210, 0), width=2)
        for idx, line in enumerate(text_lines):
            fill = (255, 230, 80) if idx == 0 else (255, 255, 255)
            draw.text((x_msg, y_msg + idx * line_h), line[:90], font=font_missing, fill=fill)

    if v06_points or v01_heatmap_points or v05_heatmap_points:
        legend = "purple=v0.1 | pink=v0.5 | orange=v0.4.1 | red=v0.6/current | cyan=suggestion | yellow=QC | star=P7V"
    else:
        legend = "purple=heatmap | orange=v0.3.4 | cyan=suggestion | red=hybrid compare | yellow=QC | star=P7V"
    font = _font(15)
    bbox = draw.textbbox((0, 0), legend, font=font)
    y0 = max(8, height - bbox[3] - 18)
    draw.rectangle((8, y0, bbox[2] + 18, y0 + bbox[3] + 10), fill=(0, 0, 0))
    draw.text((13, y0 + 4), legend, font=font, fill=(255, 255, 255))
    return np.asarray(canvas)
