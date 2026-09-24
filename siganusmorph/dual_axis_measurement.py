"""Dual-axis measurement comparison utilities.

This module compares the existing model/keypoint-derived body axis with the
body-contour midline axis from v0.6.2.  It is intentionally read-only: callers
can benchmark or display the alternate measurement axis without changing saved
manual annotations or model defaults.
"""

from __future__ import annotations

from math import acos, degrees, hypot
from typing import Any, Mapping, Sequence

import numpy as np

from .body_contour_midline import estimate_body_contour_midline_points
from .hybrid_point_selector import derive_hybrid_p7v
from .local_normal_measurement import (
    estimate_body_depth_by_local_normals,
    estimate_peduncle_depth_by_local_normals,
)
from .segmentation import segment_fish_from_blue_board


BODY_AXIS_KEYS = (
    "P1_snout_tip",
    "C1_head_axis_point",
    "C2_trunk_axis_point",
    "C3_posterior_trunk_axis_point",
    "P4_peduncle_start_midpoint",
    "C4_peduncle_axis_point",
    "P5_caudal_base_midpoint",
)

DEFAULT_CONFIG = {
    "mm_per_pixel": 0.1,
    "axis_sample_count": 220,
    "axis_jump_angle_deg": 35.0,
    "TL_DIFF_REVIEW_MM": 5.0,
    "SL_DIFF_REVIEW_MM": 5.0,
    "CURVATURE_DIFF_REVIEW": 0.03,
    "BODY_SMOOTHNESS_IMPROVEMENT_RATIO": 0.85,
}

MEASUREMENT_AXIS_MODEL = "model_axis"
MEASUREMENT_AXIS_BODY_MIDLINE = "body_midline_axis"
MEASUREMENT_AXIS_AUTO_QC = "auto_qc_gated"
MEASUREMENT_AXIS_MANUAL_REVIEW = "manual_review_required"


def _merge_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_CONFIG)
    if config:
        merged.update(dict(config))
    return merged


def _xy(value: Any) -> tuple[float, float] | None:
    if isinstance(value, Mapping):
        if "x" in value and "y" in value:
            return float(value["x"]), float(value["y"])
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return None


def _as_list(point: tuple[float, float] | None) -> list[float] | None:
    return None if point is None else [float(point[0]), float(point[1])]


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return hypot(b[0] - a[0], b[1] - a[1])


def _polyline_length(points: Sequence[tuple[float, float]]) -> float:
    return sum(_distance(a, b) for a, b in zip(points, points[1:]))


def _cumdist(points: Sequence[tuple[float, float]]) -> np.ndarray:
    if not points:
        return np.asarray([], dtype=float)
    values = [0.0]
    for a, b in zip(points, points[1:]):
        values.append(values[-1] + _distance(a, b))
    return np.asarray(values, dtype=float)


def _resample_polyline(points: Sequence[tuple[float, float]], sample_count: int) -> list[tuple[float, float]]:
    if len(points) <= 1:
        return list(points)
    cumulative = _cumdist(points)
    total = float(cumulative[-1])
    if total <= 0:
        return list(points)
    targets = np.linspace(0.0, total, max(2, int(sample_count)))
    out: list[tuple[float, float]] = []
    segment = 0
    for target in targets:
        while segment < len(cumulative) - 2 and cumulative[segment + 1] < target:
            segment += 1
        start_s = cumulative[segment]
        end_s = cumulative[segment + 1]
        if end_s <= start_s:
            out.append(points[segment])
            continue
        ratio = float((target - start_s) / (end_s - start_s))
        x = points[segment][0] + ratio * (points[segment + 1][0] - points[segment][0])
        y = points[segment][1] + ratio * (points[segment + 1][1] - points[segment][1])
        out.append((float(x), float(y)))
    return out


def _bend_angles(points: Sequence[tuple[float, float]]) -> list[float]:
    angles: list[float] = []
    for a, b, c in zip(points, points[1:], points[2:]):
        v1 = (b[0] - a[0], b[1] - a[1])
        v2 = (c[0] - b[0], c[1] - b[1])
        l1 = hypot(*v1)
        l2 = hypot(*v2)
        if l1 <= 1e-9 or l2 <= 1e-9:
            continue
        dot = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)))
        angles.append(degrees(acos(dot)))
    return angles


def _max_deviation(points: Sequence[tuple[float, float]], start: tuple[float, float], end: tuple[float, float]) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    denom = hypot(dx, dy)
    if denom <= 1e-9:
        return 0.0
    return max(abs(dy * p[0] - dx * p[1] + end[0] * start[1] - end[1] * start[0]) / denom for p in points)


def _axis_points_from_keypoints(keypoints: Mapping[str, Any]) -> list[tuple[float, float]] | None:
    points: list[tuple[float, float]] = []
    for key in BODY_AXIS_KEYS:
        point = _xy(keypoints.get(key))
        if point is None:
            return None
        points.append(point)
    return points


def _axis_curve(points: Sequence[tuple[float, float]], cfg: Mapping[str, Any]) -> dict[str, Any]:
    sampled = _resample_polyline(points, int(cfg.get("axis_sample_count", DEFAULT_CONFIG["axis_sample_count"])))
    s_values = _cumdist(sampled)
    return {
        "axis_curve_type": "polyline",
        "axis_curve_quality": "good" if len(sampled) >= 2 else "failed",
        "control_points": [[float(x), float(y)] for x, y in points],
        "sampled_points": [[float(x), float(y)] for x, y in sampled],
        "s_values": [float(v) for v in s_values],
        "length_px": float(s_values[-1]) if len(s_values) else 0.0,
        "review_reason": "",
    }


def _curvature_level(curvature_index: float | None) -> str:
    if curvature_index is None or not np.isfinite(curvature_index):
        return "unknown"
    if curvature_index <= 1.03:
        return "straight_or_mild"
    if curvature_index <= 1.08:
        return "moderate"
    return "high"


def _body_axis_keypoints(base: Mapping[str, Any], body_midline_result: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    replacements = {
        "C1_head_axis_point": body_midline_result.get("C1_geometric"),
        "C2_trunk_axis_point": body_midline_result.get("C2_geometric"),
        "C3_posterior_trunk_axis_point": body_midline_result.get("C3_geometric"),
        "C4_peduncle_axis_point": body_midline_result.get("C4_geometric"),
    }
    for key, point in replacements.items():
        xy = _xy(point)
        if xy is not None:
            out[key] = [xy[0], xy[1]]
    return out


def _measure_axis(
    fish_mask: Any,
    keypoints: Mapping[str, Any],
    image_shape: Sequence[int],
    cfg: Mapping[str, Any],
    *,
    source_name: str,
) -> dict[str, Any]:
    mm_per_pixel = float(cfg.get("mm_per_pixel", DEFAULT_CONFIG["mm_per_pixel"]))
    points = _axis_points_from_keypoints(keypoints)
    if points is None:
        return {
            "source": source_name,
            "axis_curve": {},
            "TL_curve_mm": np.nan,
            "SL_curve_mm": np.nan,
            "curvature_index": np.nan,
            "max_axis_deviation_mm": np.nan,
            "axis_bend_angle_deg": np.nan,
            "axis_smoothness_score": np.nan,
            "P7V_valid": False,
            "body_depth_local_normal_mm": np.nan,
            "peduncle_depth_local_normal_mm": np.nan,
            "measurements_needs_review": True,
            "curvature_qc_level": "unknown",
            "review_reason": "missing_axis_points",
        }
    p1 = points[0]
    p5 = points[-1]
    curve = _axis_curve(points, cfg)
    sl_curve_px = _polyline_length(points)
    sl_straight_px = _distance(p1, p5)
    curvature_index = sl_curve_px / sl_straight_px if sl_straight_px > 0 else np.nan
    angles = _bend_angles(points)
    max_bend = max(angles) if angles else 0.0
    mean_bend = float(np.mean(angles)) if angles else 0.0
    jump_count = int(sum(a > float(cfg.get("axis_jump_angle_deg", 35.0)) for a in angles))
    smoothness = mean_bend + 0.35 * max_bend + 10.0 * jump_count
    p7v = derive_hybrid_p7v(keypoints)
    p7v_valid = bool(p7v.get("point")) and not bool(p7v.get("p7v_invalid_reason"))
    extension_px = float(p7v.get("extension_px", np.nan)) if p7v_valid else np.nan
    tl_curve_mm = (sl_curve_px + extension_px) * mm_per_pixel if p7v_valid else np.nan

    body_depth = estimate_body_depth_by_local_normals(
        fish_mask,
        curve,
        keypoints,
        image_shape,
        config=cfg,
    )
    peduncle_depth = estimate_peduncle_depth_by_local_normals(
        fish_mask,
        curve,
        keypoints,
        image_shape,
        config=cfg,
    )
    level = _curvature_level(float(curvature_index) if np.isfinite(curvature_index) else None)
    needs_review = bool(
        level == "high"
        or jump_count > 0
        or not p7v_valid
        or not bool(body_depth.get("qc_pass", False))
        or not bool(peduncle_depth.get("qc_pass", False))
    )
    reasons = []
    if level == "high":
        reasons.append("high_curvature")
    if jump_count > 0:
        reasons.append("axis_jump")
    if not p7v_valid:
        reasons.append("invalid_p7v")
    if not bool(body_depth.get("qc_pass", False)):
        reasons.append("body_depth_qc_failed")
    if not bool(peduncle_depth.get("qc_pass", False)):
        reasons.append("peduncle_depth_qc_failed")

    return {
        "source": source_name,
        "axis_curve": curve,
        "axis_points": [[float(x), float(y)] for x, y in points],
        "derived_points": {"P7V_caudal_fin_posterior_endpoint": p7v},
        "TL_curve_mm": float(tl_curve_mm) if np.isfinite(tl_curve_mm) else np.nan,
        "SL_curve_mm": float(sl_curve_px * mm_per_pixel),
        "curvature_index": float(curvature_index) if np.isfinite(curvature_index) else np.nan,
        "max_axis_deviation_mm": float(_max_deviation(points, p1, p5) * mm_per_pixel),
        "axis_bend_angle_deg": float(max_bend),
        "mean_axis_bend_angle_deg": float(mean_bend),
        "axis_jump_count": int(jump_count),
        "axis_smoothness_score": float(smoothness),
        "P7V_valid": bool(p7v_valid),
        "body_depth_local_normal_mm": float(body_depth.get("width_mm")) if body_depth.get("width_mm") is not None else np.nan,
        "peduncle_depth_local_normal_mm": float(peduncle_depth.get("width_mm")) if peduncle_depth.get("width_mm") is not None else np.nan,
        "body_depth_local_normal": body_depth,
        "peduncle_depth_local_normal": peduncle_depth,
        "measurements_needs_review": needs_review,
        "curvature_qc_level": level,
        "review_reason": ";".join(reasons),
    }


def _safe_diff(a: Any, b: Any) -> float:
    try:
        aa = float(a)
        bb = float(b)
    except Exception:
        return np.nan
    if not np.isfinite(aa) or not np.isfinite(bb):
        return np.nan
    return aa - bb


def compare_dual_axis_measurements(
    fish_mask: Any,
    keypoints: Mapping[str, Any],
    body_midline_result: Mapping[str, Any],
    image_shape: Sequence[int],
    scale_info: Mapping[str, Any] | None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare model-axis and body-midline-axis measurements.

    The returned structure is suitable for CSV flattening and for embedding in
    future preannotation metadata.  It does not mutate `keypoints`.
    """

    cfg = _merge_config(config)
    if scale_info and "mm_per_pixel" in scale_info:
        cfg["mm_per_pixel"] = float(scale_info["mm_per_pixel"])

    model_keypoints = dict(keypoints)
    body_keypoints = _body_axis_keypoints(keypoints, body_midline_result)
    model_axis = _measure_axis(fish_mask, model_keypoints, image_shape, cfg, source_name="model_axis")
    body_axis = _measure_axis(fish_mask, body_keypoints, image_shape, cfg, source_name="body_midline_axis")

    tl_diff = _safe_diff(body_axis.get("TL_curve_mm"), model_axis.get("TL_curve_mm"))
    sl_diff = _safe_diff(body_axis.get("SL_curve_mm"), model_axis.get("SL_curve_mm"))
    curvature_diff = _safe_diff(body_axis.get("curvature_index"), model_axis.get("curvature_index"))
    body_qc = body_midline_result.get("body_midline_qc", {}) if isinstance(body_midline_result, Mapping) else {}
    body_qc_pass = bool(body_qc.get("body_midline_qc_pass", False))
    dual_axis_disagreement = bool(
        (np.isfinite(tl_diff) and abs(tl_diff) > float(cfg["TL_DIFF_REVIEW_MM"]))
        or (np.isfinite(sl_diff) and abs(sl_diff) > float(cfg["SL_DIFF_REVIEW_MM"]))
        or (np.isfinite(curvature_diff) and abs(curvature_diff) > float(cfg["CURVATURE_DIFF_REVIEW"]))
    )

    reasons: list[str] = []
    if not body_qc_pass:
        reasons.append("body_midline_qc_failed")
    if dual_axis_disagreement:
        reasons.append("dual_axis_disagreement")
    if body_axis.get("curvature_qc_level") == "high" or model_axis.get("curvature_qc_level") == "high":
        reasons.append("high_curvature")

    if "high_curvature" in reasons:
        recommended = "manual_review_required"
    elif not body_qc_pass:
        recommended = "model_axis"
    elif dual_axis_disagreement:
        recommended = "model_axis"
    elif float(body_axis.get("axis_smoothness_score", np.inf)) < float(model_axis.get("axis_smoothness_score", np.inf)) * float(
        cfg["BODY_SMOOTHNESS_IMPROVEMENT_RATIO"]
    ):
        recommended = "body_midline_axis"
        reasons.append("body_midline_axis_smoother")
    else:
        recommended = "model_axis"
        reasons.append("model_axis_stable_or_similar")

    comparison = {
        "TL_curve_diff_mm": tl_diff,
        "SL_curve_diff_mm": sl_diff,
        "curvature_index_diff": curvature_diff,
        "dual_axis_disagreement": dual_axis_disagreement,
        "body_midline_qc_pass": body_qc_pass,
        "body_midline_qc_reason": body_qc.get("body_midline_qc_reason", ""),
    }
    return {
        "model_axis_measurements": model_axis,
        "body_midline_axis_measurements": body_axis,
        "dual_axis_comparison": comparison,
        "axis_recommendation": {
            "recommended_measurement_axis": recommended,
            "recommendation_reason": ";".join(dict.fromkeys(reasons)),
            "measurements_needs_review": bool(
                dual_axis_disagreement
                or model_axis.get("measurements_needs_review", False)
                or body_axis.get("measurements_needs_review", False)
            ),
        },
        "qc_results": {
            "model_axis_review": model_axis.get("measurements_needs_review", True),
            "body_midline_axis_review": body_axis.get("measurements_needs_review", True),
            "dual_axis_disagreement": dual_axis_disagreement,
            "body_midline_qc_pass": body_qc_pass,
        },
    }


def select_measurement_axis(
    model_axis_measurements: Mapping[str, Any],
    body_midline_axis_measurements: Mapping[str, Any],
    dual_axis_comparison: Mapping[str, Any],
    curvature_qc: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Choose the measurement axis by the v0.6.4 QC-gated rule."""

    curvature_qc = curvature_qc or {}
    model_p7v_valid = bool(model_axis_measurements.get("P7V_valid", False))
    body_p7v_valid = bool(body_midline_axis_measurements.get("P7V_valid", False))
    high_curvature = (
        str(curvature_qc.get("curvature_qc_level", "")).lower() == "high"
        or str(model_axis_measurements.get("curvature_qc_level", "")).lower() == "high"
        or str(body_midline_axis_measurements.get("curvature_qc_level", "")).lower() == "high"
    )
    if high_curvature or not (model_p7v_valid and body_p7v_valid):
        return {
            "selected_measurement_axis": MEASUREMENT_AXIS_MANUAL_REVIEW,
            "measurements_needs_review": True,
            "measurement_axis_review_reason": "high_curvature_or_invalid_p7v",
        }
    body_qc_pass = bool(
        dual_axis_comparison.get(
            "body_midline_qc_pass",
            not bool(body_midline_axis_measurements.get("measurements_needs_review", True)),
        )
    )
    if body_qc_pass and not bool(dual_axis_comparison.get("dual_axis_disagreement", True)):
        return {
            "selected_measurement_axis": MEASUREMENT_AXIS_BODY_MIDLINE,
            "measurements_needs_review": False,
            "measurement_axis_review_reason": "body_midline_axis_qc_pass",
        }
    return {
        "selected_measurement_axis": MEASUREMENT_AXIS_MODEL,
        "measurements_needs_review": bool(model_axis_measurements.get("measurements_needs_review", False)),
        "measurement_axis_review_reason": "body_midline_qc_failed_or_dual_axis_disagreement",
    }


def _clean_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _clean_mapping(payload: Any) -> Any:
    if isinstance(payload, np.ndarray):
        return None
    if isinstance(payload, Mapping):
        return {str(key): _clean_mapping(value) for key, value in payload.items() if value is not None and not isinstance(value, np.ndarray)}
    if isinstance(payload, (list, tuple)):
        return [_clean_mapping(value) for value in payload]
    return _clean_scalar(payload)


def _axis_payload(measurements: Mapping[str, Any], *, include_body_contours: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "axis_points": measurements.get("axis_points", []),
        "TL_curve_mm": measurements.get("TL_curve_mm"),
        "SL_curve_mm": measurements.get("SL_curve_mm"),
        "curvature_index": measurements.get("curvature_index"),
        "axis_smoothness": measurements.get("axis_smoothness_score"),
        "axis_bend_angle_deg": measurements.get("axis_bend_angle_deg"),
        "max_axis_deviation_mm": measurements.get("max_axis_deviation_mm"),
        "body_depth_local_normal_mm": measurements.get("body_depth_local_normal_mm"),
        "peduncle_depth_local_normal_mm": measurements.get("peduncle_depth_local_normal_mm"),
        "P7V_valid": measurements.get("P7V_valid"),
        "qc_pass": not bool(measurements.get("measurements_needs_review", True)),
        "review_reason": measurements.get("review_reason", ""),
    }
    if include_body_contours:
        payload.update(
            {
                "body_midline_points": include_body_contours.get("body_midline_points", []),
                "dorsal_body_contour": include_body_contours.get("dorsal_body_contour", []),
                "ventral_body_contour": include_body_contours.get("ventral_body_contour", []),
                "body_midline_qc": include_body_contours.get("body_midline_qc", {}),
            }
        )
    return _clean_mapping(payload)


def _row_fields(measurement_axes: Mapping[str, Any]) -> dict[str, Any]:
    model = measurement_axes.get("model_axis", {}) if isinstance(measurement_axes.get("model_axis", {}), Mapping) else {}
    body = measurement_axes.get("body_midline_axis", {}) if isinstance(measurement_axes.get("body_midline_axis", {}), Mapping) else {}
    selected_axis = str(measurement_axes.get("selected_measurement_axis", MEASUREMENT_AXIS_MODEL))
    if selected_axis == MEASUREMENT_AXIS_BODY_MIDLINE:
        selected_source = body
    elif selected_axis == MEASUREMENT_AXIS_MODEL:
        selected_source = model
    else:
        selected_source = {}
    comparison = measurement_axes.get("dual_axis_comparison", {}) if isinstance(measurement_axes.get("dual_axis_comparison", {}), Mapping) else {}
    return {
        "selected_measurement_axis": selected_axis,
        "TL_curve_model_axis_mm": model.get("TL_curve_mm"),
        "SL_curve_model_axis_mm": model.get("SL_curve_mm"),
        "curvature_index_model_axis": model.get("curvature_index"),
        "axis_smoothness_model_axis": model.get("axis_smoothness"),
        "body_depth_model_axis_mm": model.get("body_depth_local_normal_mm"),
        "peduncle_depth_model_axis_mm": model.get("peduncle_depth_local_normal_mm"),
        "TL_curve_body_midline_axis_mm": body.get("TL_curve_mm"),
        "SL_curve_body_midline_axis_mm": body.get("SL_curve_mm"),
        "curvature_index_body_midline_axis": body.get("curvature_index"),
        "axis_smoothness_body_midline_axis": body.get("axis_smoothness"),
        "body_depth_body_midline_axis_mm": body.get("body_depth_local_normal_mm"),
        "peduncle_depth_body_midline_axis_mm": body.get("peduncle_depth_local_normal_mm"),
        "TL_curve_selected_mm": selected_source.get("TL_curve_mm"),
        "SL_curve_selected_mm": selected_source.get("SL_curve_mm"),
        "curvature_index_selected": selected_source.get("curvature_index"),
        "body_depth_selected_mm": selected_source.get("body_depth_local_normal_mm"),
        "peduncle_depth_selected_mm": selected_source.get("peduncle_depth_local_normal_mm"),
        "TL_curve_axis_diff_mm": comparison.get("TL_curve_diff_mm"),
        "SL_curve_axis_diff_mm": comparison.get("SL_curve_diff_mm"),
        "curvature_index_axis_diff": comparison.get("curvature_index_diff"),
        "dual_axis_disagreement": comparison.get("dual_axis_disagreement", False),
        "measurements_needs_review": measurement_axes.get("measurements_needs_review", False),
        "measurement_axis_review_reason": measurement_axes.get("measurement_axis_review_reason", ""),
    }


def compute_measurement_axis_metadata(
    warped_image: Any,
    full_keypoints: Mapping[str, Any],
    *,
    mm_per_pixel: float = 0.1,
    measurement_axis_mode: str = MEASUREMENT_AXIS_MODEL,
    fish_mask: Any = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute model/body-midline axes and selected measurement-axis fields.

    This function never mutates keypoints.  It is safe to call from UI save or
    preview code because geometric C points are stored only under
    ``measurement_axes`` metadata.
    """

    cfg = _merge_config(config)
    cfg["mm_per_pixel"] = float(mm_per_pixel)
    if fish_mask is None:
        fish_mask, _bbox, segmentation_quality = segment_fish_from_blue_board(warped_image)
    else:
        segmentation_quality = {"segmentation_success": True, "segmentation_quality": "provided"}
    body_result = estimate_body_contour_midline_points(
        fish_mask,
        full_keypoints,
        getattr(warped_image, "shape", (0, 0)),
        config={"mm_per_pixel": float(mm_per_pixel)},
    )
    dual = compare_dual_axis_measurements(
        fish_mask,
        full_keypoints,
        body_result,
        getattr(warped_image, "shape", (0, 0)),
        {"mm_per_pixel": float(mm_per_pixel)},
        config=cfg,
    )
    model = dual["model_axis_measurements"]
    body = dual["body_midline_axis_measurements"]
    comparison = dual["dual_axis_comparison"]
    auto_selection = select_measurement_axis(model, body, comparison, {})
    requested = str(measurement_axis_mode or MEASUREMENT_AXIS_MODEL)
    if requested == MEASUREMENT_AXIS_BODY_MIDLINE:
        selection = {
            "selected_measurement_axis": MEASUREMENT_AXIS_BODY_MIDLINE,
            "measurements_needs_review": bool(body.get("measurements_needs_review", False)) or bool(comparison.get("dual_axis_disagreement", False)),
            "measurement_axis_review_reason": "user_selected_body_midline_axis",
        }
    elif requested == MEASUREMENT_AXIS_AUTO_QC:
        selection = auto_selection
    else:
        selection = {
            "selected_measurement_axis": MEASUREMENT_AXIS_MODEL,
            "measurements_needs_review": bool(model.get("measurements_needs_review", False)),
            "measurement_axis_review_reason": "user_selected_model_axis",
        }
    measurement_axes = {
        "measurement_axis_mode_request": requested,
        "model_axis": _axis_payload(model),
        "body_midline_axis": _axis_payload(body, include_body_contours=body_result),
        "dual_axis_comparison": _clean_mapping(
            {
                **comparison,
                "recommended_measurement_axis": auto_selection.get("selected_measurement_axis"),
                "auto_qc_review_reason": auto_selection.get("measurement_axis_review_reason", ""),
            }
        ),
        "selected_measurement_axis": selection["selected_measurement_axis"],
        "measurements_needs_review": bool(selection["measurements_needs_review"]),
        "measurement_axis_review_reason": selection["measurement_axis_review_reason"],
        "segmentation_quality": _clean_mapping(segmentation_quality),
    }
    return {
        "measurement_axes": _clean_mapping(measurement_axes),
        "measurement_axis_row_fields": _row_fields(measurement_axes),
        "dual_axis_result": _clean_mapping(dual),
        "body_midline_result": _clean_mapping(
            {
                "C1_geometric": body_result.get("C1_geometric"),
                "C2_geometric": body_result.get("C2_geometric"),
                "C3_geometric": body_result.get("C3_geometric"),
                "C4_geometric": body_result.get("C4_geometric"),
                "body_midline_qc": body_result.get("body_midline_qc", {}),
            }
        ),
    }
