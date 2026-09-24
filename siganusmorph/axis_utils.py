"""Axis fitting helpers for curved-body measurements."""

from __future__ import annotations

from math import hypot
from typing import Mapping, Sequence

from .config import BODY_AXIS_POINT_ORDER, CAUDAL_FIN_AXIS_POINT_ORDER, CAUDAL_TIP_CANDIDATE_NAMES


PointLike = Mapping[str, float] | tuple[float, float] | list[float]

AXIS_MODE_AUTO = "auto"
AXIS_MODE_POLYLINE = "polyline"
AXIS_MODE_SPLINE = "spline"


def point_xy(point: PointLike) -> tuple[float, float]:
    """Return an ``(x, y)`` tuple for a point-like object."""
    if isinstance(point, Mapping):
        return float(point["x"]), float(point["y"])
    return float(point[0]), float(point[1])


def body_axis_points_from_keypoints(keypoints: Mapping[str, PointLike]) -> list[tuple[float, float]]:
    """Return body-axis points in the fixed P1-C1-C2-C3-P4-C4-P5 order."""
    return [point_xy(keypoints[name]) for name in BODY_AXIS_POINT_ORDER if name in keypoints]


def caudal_fin_axis_points_from_keypoints(keypoints: Mapping[str, PointLike]) -> list[tuple[float, float]]:
    """Return caudal-fin reference points in the fixed P5-P6 order."""
    return [point_xy(keypoints[name]) for name in CAUDAL_FIN_AXIS_POINT_ORDER if name in keypoints]


def axis_points_from_keypoints(keypoints: Mapping[str, PointLike]) -> list[tuple[float, float]]:
    """Backward-compatible alias for body-axis points."""
    return body_axis_points_from_keypoints(keypoints)


def polyline_length(points: Sequence[tuple[float, float]]) -> float:
    """Return the length of a polyline in pixels."""
    if len(points) < 2:
        return 0.0
    return sum(hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in zip(points, points[1:]))


def unit_vector(start: tuple[float, float], end: tuple[float, float]) -> tuple[float, float] | None:
    """Return the unit vector from start to end, or None for a zero-length vector."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = hypot(dx, dy)
    if length <= 1e-9:
        return None
    return dx / length, dy / length


def dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Return the dot product of two 2D vectors."""
    return a[0] * b[0] + a[1] * b[1]


def derive_caudal_posterior_endpoint(keypoints: Mapping[str, PointLike]) -> dict[str, object]:
    """Derive P7V from upper/lower caudal tips projected onto the tail axis."""
    p5 = point_xy(keypoints["caudal_base_midpoint"])
    p6 = point_xy(keypoints["caudal_fork_midpoint"])
    c4 = point_xy(keypoints["peduncle_axis_point"])
    upper = point_xy(keypoints["caudal_fin_upper_tip"])
    lower = point_xy(keypoints["caudal_fin_lower_tip"])

    tail_axis = unit_vector(p5, p6)
    tail_axis_source = "P5_to_P6"
    if tail_axis is None:
        tail_axis = unit_vector(c4, p5)
        tail_axis_source = "C4_to_P5"
    if tail_axis is None:
        raise ValueError("Cannot derive caudal endpoint because tail-axis reference points overlap.")

    proj_upper = dot((upper[0] - p5[0], upper[1] - p5[1]), tail_axis)
    proj_lower = dot((lower[0] - p5[0], lower[1] - p5[1]), tail_axis)
    if proj_upper >= proj_lower:
        selected = "upper"
        extension = proj_upper
    else:
        selected = "lower"
        extension = proj_lower
    extension = max(0.0, extension)
    endpoint = (p5[0] + extension * tail_axis[0], p5[1] + extension * tail_axis[1])
    return {
        "point": {"x": endpoint[0], "y": endpoint[1]},
        "tail_axis": tail_axis,
        "tail_axis_source": tail_axis_source,
        "proj_upper_px": proj_upper,
        "proj_lower_px": proj_lower,
        "extension_px": extension,
        "selected": selected,
        "candidates": {
            "upper": {"x": upper[0], "y": upper[1]},
            "lower": {"x": lower[0], "y": lower[1]},
        },
        "candidate_names": CAUDAL_TIP_CANDIDATE_NAMES,
    }


def _dedupe_consecutive_points(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    cleaned: list[tuple[float, float]] = []
    for point in points:
        if not cleaned or hypot(point[0] - cleaned[-1][0], point[1] - cleaned[-1][1]) > 1e-9:
            cleaned.append(point)
    return cleaned


def _natural_cubic_coefficients(t: Sequence[float], values: Sequence[float]) -> list[tuple[float, float, float, float]]:
    n = len(values)
    a = [float(value) for value in values]
    h = [t[i + 1] - t[i] for i in range(n - 1)]
    alpha = [0.0] * n
    for i in range(1, n - 1):
        alpha[i] = (3.0 / h[i]) * (a[i + 1] - a[i]) - (3.0 / h[i - 1]) * (a[i] - a[i - 1])

    l = [1.0] + [0.0] * (n - 1)
    mu = [0.0] * n
    z = [0.0] * n
    for i in range(1, n - 1):
        l[i] = 2.0 * (t[i + 1] - t[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]
    l[n - 1] = 1.0

    b = [0.0] * (n - 1)
    c = [0.0] * n
    d = [0.0] * (n - 1)
    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (a[j + 1] - a[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0
        d[j] = (c[j + 1] - c[j]) / (3.0 * h[j])
    return [(a[i], b[i], c[i], d[i]) for i in range(n - 1)]


def _eval_natural_cubic(
    t: Sequence[float],
    coefficients: Sequence[tuple[float, float, float, float]],
    value_t: float,
) -> float:
    if value_t <= t[0]:
        index = 0
    elif value_t >= t[-1]:
        index = len(coefficients) - 1
    else:
        index = 0
        for i in range(len(coefficients)):
            if t[i] <= value_t <= t[i + 1]:
                index = i
                break
    a, b, c, d = coefficients[index]
    dt = value_t - t[index]
    return a + b * dt + c * dt * dt + d * dt * dt * dt


def chord_length_cubic_spline_points(
    points: Sequence[tuple[float, float]],
    sample_count: int = 200,
) -> list[tuple[float, float]]:
    """Sample a natural cubic spline with chord-length parameterization."""
    points = _dedupe_consecutive_points(points)
    if len(points) < 3:
        return list(points)

    distances = [0.0]
    for start, end in zip(points, points[1:]):
        distances.append(distances[-1] + hypot(end[0] - start[0], end[1] - start[1]))
    total = distances[-1]
    if total <= 0:
        return list(points)

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_coefficients = _natural_cubic_coefficients(distances, xs)
    y_coefficients = _natural_cubic_coefficients(distances, ys)
    sample_count = max(len(points), int(sample_count))
    sampled: list[tuple[float, float]] = []
    for index in range(sample_count):
        value_t = total * index / (sample_count - 1)
        sampled.append(
            (
                _eval_natural_cubic(distances, x_coefficients, value_t),
                _eval_natural_cubic(distances, y_coefficients, value_t),
            )
        )
    return sampled


def catmull_rom_spline_points(
    points: Sequence[tuple[float, float]],
    samples_per_segment: int = 24,
) -> list[tuple[float, float]]:
    """Backward-compatible alias for the chord-length cubic spline sampler."""
    sample_count = max(len(points), (len(points) - 1) * max(4, int(samples_per_segment)) + 1)
    return chord_length_cubic_spline_points(points, sample_count=sample_count)


def spline_length(points: Sequence[tuple[float, float]], sample_count: int = 200) -> float:
    """Return sampled chord-length cubic spline length in pixels."""
    return polyline_length(chord_length_cubic_spline_points(points, sample_count=sample_count))


def recommend_axis_mode(curvature_index_polyline: float | None) -> str:
    """Recommend an axis mode from the polyline curvature index."""
    if curvature_index_polyline is None:
        return AXIS_MODE_POLYLINE
    return AXIS_MODE_POLYLINE if curvature_index_polyline <= 1.02 else AXIS_MODE_SPLINE


def normalize_axis_mode(axis_mode: str | None, curvature_index_polyline: float | None = None) -> str:
    """Normalize user-facing or internal axis mode values."""
    if axis_mode in {AXIS_MODE_POLYLINE, "polyline_axis", "折线中轴线（polyline axis）", "折线中轴线"}:
        return AXIS_MODE_POLYLINE
    if axis_mode in {AXIS_MODE_SPLINE, "spline_axis", "平滑曲线中轴线（spline axis）", "平滑曲线中轴线"}:
        return AXIS_MODE_SPLINE
    return recommend_axis_mode(curvature_index_polyline)


def axis_mode_label(axis_mode: str | None) -> str:
    """Return a compact Chinese label for an axis mode."""
    if axis_mode == AXIS_MODE_SPLINE:
        return "平滑曲线中轴线"
    if axis_mode == AXIS_MODE_POLYLINE:
        return "折线中轴线"
    return "自动推荐"
