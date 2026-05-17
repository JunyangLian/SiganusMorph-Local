"""Distance and morphometric calculations."""

from __future__ import annotations

from math import hypot
from typing import Any, Mapping

from .axis_utils import (
    AXIS_MODE_POLYLINE,
    AXIS_MODE_SPLINE,
    body_axis_points_from_keypoints,
    derive_caudal_posterior_endpoint,
    normalize_axis_mode,
    polyline_length,
    spline_length,
)
from .config import BODY_AXIS_POINT_ORDER, CAUDAL_TIP_CANDIDATE_NAMES, DERIVED_POINT_DEFS, KEYPOINT_DEFS, MEASUREMENT_DEFS, RESULT_COLUMNS


PointLike = Mapping[str, float] | tuple[float, float] | list[float]


def _xy(point: PointLike) -> tuple[float, float]:
    if isinstance(point, Mapping):
        return float(point["x"]), float(point["y"])
    return float(point[0]), float(point[1])


def euclidean_distance(p1: PointLike, p2: PointLike) -> float:
    """Return the Euclidean distance in pixels between two points."""
    x1, y1 = _xy(p1)
    x2, y2 = _xy(p2)
    return hypot(x2 - x1, y2 - y1)


def calculate_measurements(
    keypoints: Mapping[str, PointLike],
    mm_per_pixel: float,
    axis_mode_selected: str = "auto",
) -> dict[str, Any]:
    """Calculate straight-line and axis-based morphometrics from keypoints."""
    if mm_per_pixel <= 0:
        raise ValueError("mm_per_pixel must be greater than 0.")

    required_names = {point_name for _, _, a, b in MEASUREMENT_DEFS for point_name in (a, b)}
    required_names.update(BODY_AXIS_POINT_ORDER)
    required_names.update(CAUDAL_TIP_CANDIDATE_NAMES)
    required_names.add("caudal_fork_midpoint")
    missing = sorted(required_names - set(keypoints))
    if missing:
        raise KeyError(f"Missing required keypoints: {', '.join(missing)}")

    measurements: dict[str, float | None] = {}
    for column, _label, point_a, point_b in MEASUREMENT_DEFS:
        distance_px = euclidean_distance(keypoints[point_a], keypoints[point_b])
        measurements[column] = distance_px * mm_per_pixel

    body_axis_points = body_axis_points_from_keypoints(keypoints)
    caudal_endpoint = derive_caudal_posterior_endpoint(keypoints)
    derived_endpoint = caudal_endpoint["point"]
    sl_straight = measurements.get("SL_straight_mm")
    caudal_extension_axis = float(caudal_endpoint["extension_px"]) * mm_per_pixel
    tl_straight_axis = euclidean_distance(keypoints["snout_tip"], derived_endpoint) * mm_per_pixel
    sl_polyline = polyline_length(body_axis_points) * mm_per_pixel if len(body_axis_points) >= 2 else None
    sl_spline = spline_length(body_axis_points, sample_count=200) * mm_per_pixel if len(body_axis_points) >= 3 else sl_polyline
    tl_polyline = (
        float(sl_polyline) + caudal_extension_axis
        if sl_polyline is not None
        else None
    )
    tl_spline = (
        float(sl_spline) + caudal_extension_axis
        if sl_spline is not None
        else tl_polyline
    )
    peduncle_axis_points = [
        _xy(keypoints["peduncle_start_midpoint"]),
        _xy(keypoints["peduncle_axis_point"]),
        _xy(keypoints["caudal_base_midpoint"]),
    ]
    caudal_peduncle_length_axis = polyline_length(peduncle_axis_points) * mm_per_pixel
    curvature_polyline = (
        float(sl_polyline) / float(sl_straight)
        if sl_polyline is not None and sl_straight and float(sl_straight) > 0
        else None
    )
    curvature_spline = (
        float(sl_spline) / float(sl_straight)
        if sl_spline is not None and sl_straight and float(sl_straight) > 0
        else None
    )
    final_axis_mode = normalize_axis_mode(axis_mode_selected, curvature_polyline)
    if final_axis_mode == AXIS_MODE_SPLINE:
        sl_final = sl_spline
        tl_final = tl_spline
        selected_curvature = curvature_spline
    else:
        final_axis_mode = AXIS_MODE_POLYLINE
        sl_final = sl_polyline
        tl_final = tl_polyline
        selected_curvature = curvature_polyline

    measurements.update(
        {
            "axis_mode_selected": final_axis_mode,
            "derived_points": {
                "P7V_caudal_fin_posterior_endpoint": derived_endpoint,
            },
            "caudal_tip_selected": caudal_endpoint["selected"],
            "caudal_tip_projection_upper_px": caudal_endpoint["proj_upper_px"],
            "caudal_tip_projection_lower_px": caudal_endpoint["proj_lower_px"],
            "tail_axis_source": caudal_endpoint["tail_axis_source"],
            "caudal_extension_axis_mm": caudal_extension_axis,
            "TL_straight_axis_mm": tl_straight_axis,
            "SL_axis_polyline_mm": sl_polyline,
            "SL_axis_spline_mm": sl_spline,
            "TL_axis_polyline_mm": tl_polyline,
            "TL_axis_spline_mm": tl_spline,
            "caudal_peduncle_length_axis_mm": caudal_peduncle_length_axis,
            "body_depth_axis_corrected_mm": None,
            "head_length_axis_mm": None,
            "snout_length_axis_mm": None,
            "caudal_peduncle_depth_axis_corrected_mm": None,
            "curvature_index_polyline": curvature_polyline,
            "curvature_index_spline": curvature_spline,
            "SL_final_mm": sl_final,
            "TL_final_mm": tl_final,
            "SL_curve_mm": sl_final,
            "TL_curve_mm": tl_final,
            "curvature_index": selected_curvature,
        }
    )

    return measurements


def build_result_row(
    image_name: str,
    specimen_id: str,
    scale_method: str,
    mm_per_pixel: float,
    keypoints: Mapping[str, PointLike],
    measurements: Mapping[str, Any],
    needs_review: bool = False,
    notes: str = "",
    source_type: str = "",
) -> dict[str, Any]:
    """Flatten metadata, keypoints, and measurements into one output row."""
    row: dict[str, Any] = {
        "image_name": image_name,
        "specimen_id": specimen_id,
        "source_type": source_type,
        "scale_method": scale_method,
        "mm_per_pixel": round(float(mm_per_pixel), 8),
        "axis_mode_selected": "",
        "body_depth_axis_corrected_mm": "",
        "head_length_axis_mm": "",
        "snout_length_axis_mm": "",
        "caudal_peduncle_depth_axis_corrected_mm": "",
        "SL_curve_mm": "",
        "TL_curve_mm": "",
        "curvature_index": "",
        "needs_review": bool(needs_review),
        "notes": notes,
    }

    for definition in KEYPOINT_DEFS:
        point = keypoints.get(definition.name)
        prefix = f"{definition.code}_{definition.name}"
        if point is None:
            row[f"{prefix}_x"] = ""
            row[f"{prefix}_y"] = ""
            continue
        x, y = _xy(point)
        row[f"{prefix}_x"] = round(x, 3)
        row[f"{prefix}_y"] = round(y, 3)

    derived_points = measurements.get("derived_points", {})
    if isinstance(derived_points, Mapping):
        for definition in DERIVED_POINT_DEFS:
            prefix = f"{definition.code}_{definition.name}"
            point = derived_points.get(prefix) or derived_points.get(definition.name)
            if point is None:
                row[f"{prefix}_x"] = ""
                row[f"{prefix}_y"] = ""
                continue
            x, y = _xy(point)
            row[f"{prefix}_x"] = round(x, 3)
            row[f"{prefix}_y"] = round(y, 3)

    for key, value in measurements.items():
        if value is None:
            row[key] = ""
        elif isinstance(value, Mapping):
            continue
        elif isinstance(value, str):
            row[key] = value
        elif isinstance(value, bool):
            row[key] = bool(value)
        else:
            row[key] = round(float(value), 3)

    for column in RESULT_COLUMNS:
        row.setdefault(column, "")

    return {column: row[column] for column in RESULT_COLUMNS}
