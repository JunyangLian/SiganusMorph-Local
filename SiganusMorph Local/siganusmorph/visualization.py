"""Drawing helpers for review images."""

from __future__ import annotations

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
            if tl_final is not None:
                metric_parts.append(f"TL final {float(tl_final):.1f} mm")
            caudal_tip = measurements.get("caudal_tip_selected")
            caudal_extension = measurements.get("caudal_extension_axis_mm")
            tl_curve = measurements.get("TL_curve_mm")
            if caudal_tip:
                metric_parts.append(f"caudal {caudal_tip}")
            if caudal_extension is not None:
                metric_parts.append(f"tail ext {float(caudal_extension):.1f} mm")
            if tl_curve is not None:
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
