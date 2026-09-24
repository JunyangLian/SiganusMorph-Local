from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph.body_contour_midline import estimate_body_contour_midline_points
from siganusmorph.hybrid_point_selector import derive_hybrid_p7v
from siganusmorph.image_utils import load_image_file
from siganusmorph.keypointwise_hybrid_selector import KEYPOINT_NAMES
from siganusmorph.segmentation import segment_fish_from_blue_board


MM_PER_PIXEL = 0.1
OUT = PROJECT_ROOT / "results" / "model_eval" / "v0.6.2_body_contour_midline"
VIS_OUT = OUT / "visual_comparisons"
SAME_SET = PROJECT_ROOT / "results" / "model_eval" / "v0.6_keypointwise_hybrid_selector" / "same_set_comparison"
STRICT_MANIFEST = SAME_SET / "strict_union_manifest.csv"
PRED_DIR = SAME_SET / "predictions"
C_KEYS = [
    "C1_head_axis_point",
    "C2_trunk_axis_point",
    "C3_posterior_trunk_axis_point",
    "C4_peduncle_axis_point",
]
BODY_AXIS_KEYS = [
    "P1_snout_tip",
    "C1_head_axis_point",
    "C2_trunk_axis_point",
    "C3_posterior_trunk_axis_point",
    "P4_peduncle_start_midpoint",
    "C4_peduncle_axis_point",
    "P5_caudal_base_midpoint",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def xy_from(value: Any) -> tuple[float, float] | None:
    if isinstance(value, Mapping) and "x" in value and "y" in value:
        return float(value["x"]), float(value["y"])
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return None


def corrected_keypoints(path: Path) -> dict[str, list[float]]:
    payload = load_json(path)
    points = payload.get("corrected_keypoints") or payload.get("keypoints") or {}
    out: dict[str, list[float]] = {}
    for key in KEYPOINT_NAMES:
        xy = xy_from(points.get(key))
        if xy is not None:
            out[key] = [xy[0], xy[1]]
    return out


def prediction_map(path: Path) -> dict[str, dict[str, list[float]]]:
    df = pd.read_csv(path)
    out: dict[str, dict[str, list[float]]] = {}
    for row in df.to_dict("records"):
        out.setdefault(str(row["image_name"]), {})[str(row["keypoint_name"])] = [float(row["x"]), float(row["y"])]
    return out


def dist_mm(a: Any, b: Any) -> float:
    aa = xy_from(a)
    bb = xy_from(b)
    if aa is None or bb is None:
        return float("nan")
    return math.hypot(aa[0] - bb[0], aa[1] - bb[1]) * MM_PER_PIXEL


def polyline_length(points: list[tuple[float, float]]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))


def bend_angles(points: list[tuple[float, float]]) -> list[float]:
    angles: list[float] = []
    for a, b, c in zip(points, points[1:], points[2:]):
        v1 = (b[0] - a[0], b[1] - a[1])
        v2 = (c[0] - b[0], c[1] - b[1])
        l1 = math.hypot(*v1)
        l2 = math.hypot(*v2)
        if l1 <= 1e-9 or l2 <= 1e-9:
            continue
        dot = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)))
        angles.append(math.degrees(math.acos(dot)))
    return angles


def axis_metrics(keypoints: Mapping[str, Any]) -> dict[str, Any]:
    axis = []
    for key in BODY_AXIS_KEYS:
        xy = xy_from(keypoints.get(key))
        if xy is None:
            return {
                "axis_smoothness_score": np.nan,
                "max_axis_bend_angle_deg": np.nan,
                "mean_axis_bend_angle_deg": np.nan,
                "axis_jump_count": np.nan,
                "curvature_index": np.nan,
                "TL_curve_mm": np.nan,
                "SL_curve_mm": np.nan,
                "P7V_valid": False,
                "measurements_needs_review": True,
                "axis_quality_flag": "missing_axis_point",
            }
        axis.append(xy)
    p1 = axis[0]
    p5 = axis[-1]
    sl_curve_px = polyline_length(axis)
    sl_straight_px = math.hypot(p5[0] - p1[0], p5[1] - p1[1])
    curvature = sl_curve_px / sl_straight_px if sl_straight_px > 0 else np.nan
    angles = bend_angles(axis)
    max_angle = max(angles) if angles else 0.0
    mean_angle = float(np.mean(angles)) if angles else 0.0
    jump_count = int(sum(angle > 35.0 for angle in angles))
    p7v = derive_hybrid_p7v(keypoints)
    p7v_valid = bool(p7v.get("point")) and not bool(p7v.get("p7v_invalid_reason"))
    extension = float(p7v.get("extension_px", np.nan)) if p7v_valid else np.nan
    tl_curve_mm = (sl_curve_px + extension) * MM_PER_PIXEL if p7v_valid else np.nan
    needs_review = bool(jump_count > 0 or (not p7v_valid) or (not np.isfinite(curvature)) or curvature > 1.08)
    return {
        "axis_smoothness_score": mean_angle + 0.35 * max_angle + 10.0 * jump_count,
        "max_axis_bend_angle_deg": max_angle,
        "mean_axis_bend_angle_deg": mean_angle,
        "axis_jump_count": jump_count,
        "curvature_index": curvature,
        "SL_curve_mm": sl_curve_px * MM_PER_PIXEL,
        "TL_curve_mm": tl_curve_mm,
        "P7V_valid": p7v_valid,
        "measurements_needs_review": needs_review,
        "axis_quality_flag": "review" if needs_review else "ok",
    }


def with_body_c_points(base: dict[str, list[float]], body: Mapping[str, Any]) -> dict[str, list[float]]:
    out = dict(base)
    mapping = {
        "C1_head_axis_point": body.get("C1_geometric"),
        "C2_trunk_axis_point": body.get("C2_geometric"),
        "C3_posterior_trunk_axis_point": body.get("C3_geometric"),
        "C4_peduncle_axis_point": body.get("C4_geometric"),
    }
    for key, point in mapping.items():
        xy = xy_from(point)
        if xy is not None:
            out[key] = [xy[0], xy[1]]
    return out


def draw_visual(record: Mapping[str, Any], gt: Mapping[str, Any], v041: Mapping[str, Any], v06: Mapping[str, Any], body: Mapping[str, Any]) -> None:
    warped_path = resolve_path(record.get("warped_image_path"))
    if warped_path is None or not warped_path.exists():
        return
    image = Image.open(warped_path).convert("RGB")
    scale = min(1.0, 1800 / image.width)
    display = image.resize((int(image.width * scale), int(image.height * scale))) if scale < 1 else image.copy()
    draw = ImageDraw.Draw(display, "RGBA")
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font = ImageFont.load_default()

    def sxy(point: Any) -> tuple[float, float] | None:
        xy = xy_from(point)
        return None if xy is None else (xy[0] * scale, xy[1] * scale)

    def dot(point: Any, fill: tuple[int, int, int, int], label: str, r: int = 5, outline: tuple[int, int, int, int] = (0, 0, 0, 255)) -> None:
        xy = sxy(point)
        if xy is None:
            return
        x, y = xy
        draw.ellipse([x - r, y - r, x + r, y + r], fill=fill, outline=outline, width=2)
        draw.text((x + r + 2, y - r), label, fill=fill, font=font)

    def poly(points: list[list[float]], color: tuple[int, int, int, int], width: int = 2) -> None:
        pts = [(p[0] * scale, p[1] * scale) for p in points if len(p) >= 2]
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=width)

    poly(body.get("dorsal_body_contour", []), (70, 210, 255, 210), 2)
    poly(body.get("ventral_body_contour", []), (70, 210, 255, 210), 2)
    mid = body.get("body_midline_points", [])
    if isinstance(mid, list) and len(mid) >= 2:
        pts = [(p[0] * scale, p[1] * scale) for p in mid[::3] if len(p) >= 2]
        for a, b in zip(pts, pts[1:]):
            draw.line([a, b], fill=(0, 90, 255, 180), width=2)
    for key, label in [(P4 := "P4_peduncle_start_midpoint", "P4"), (P5 := "P5_caudal_base_midpoint", "P5")]:
        dot(v06.get(key), (255, 255, 255, 220), label, r=4)
    for key in C_KEYS:
        code = key.split("_", 1)[0]
        dot(gt.get(key), (0, 220, 60, 240), code + "gt", 4)
        dot(v041.get(key), (255, 150, 30, 220), code + "41", 4)
        dot(v06.get(key), (230, 30, 40, 220), code + "6", 4)
        dot(
            body.get(
                key.replace("_head_axis_point", "_geometric")
                .replace("_trunk_axis_point", "_geometric")
                .replace("_posterior_trunk_axis_point", "_geometric")
                .replace("_peduncle_axis_point", "_geometric")
            ),
            (0, 120, 255, 220),
            code + "g",
            6,
        )

    qc = body.get("body_midline_qc", {}) if isinstance(body.get("body_midline_qc", {}), Mapping) else {}
    draw.rectangle([0, 0, 760, 62], fill=(0, 0, 0, 220))
    draw.text((8, 6), f"{record['image_name']} body contour midline", fill=(255, 255, 255, 255), font=font)
    draw.text((8, 32), f"QC: {qc.get('body_midline_qc_pass')} {qc.get('body_midline_qc_reason','')}", fill=(70, 210, 255, 255), font=font)
    VIS_OUT.mkdir(parents=True, exist_ok=True)
    display.save(VIS_OUT / f"{Path(str(record['image_name'])).stem}_body_midline_compare.png")


def build_body_results(manifest: pd.DataFrame, v06_map: dict[str, dict[str, list[float]]]) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    body_results: dict[str, dict[str, Any]] = {}
    qc_rows = []
    for index, record in enumerate(manifest.to_dict("records"), start=1):
        image_name = str(record["image_name"])
        warped_path = resolve_path(record.get("warped_image_path"))
        if warped_path is None or not warped_path.exists():
            continue
        image = load_image_file(warped_path)
        fish_mask, fish_bbox, quality = segment_fish_from_blue_board(image)
        result = estimate_body_contour_midline_points(
            fish_mask,
            v06_map.get(image_name, {}),
            image.shape,
            config={"mm_per_pixel": MM_PER_PIXEL},
        )
        body_results[image_name] = result
        qc = result.get("body_midline_qc", {}) if isinstance(result.get("body_midline_qc", {}), Mapping) else {}
        qc_rows.append(
            {
                "image_name": image_name,
                "specimen_id": record.get("specimen_id", ""),
                "subset_source": record.get("subset_source", ""),
                "included_in_v05_test": record.get("included_in_v05_test", ""),
                "included_in_realworld_review": record.get("included_in_realworld_review", ""),
                "segmentation_quality": quality,
                "body_midline_qc_pass": qc.get("body_midline_qc_pass", ""),
                "body_midline_qc_reason": qc.get("body_midline_qc_reason", ""),
                "fin_spike_removed_count": qc.get("fin_spike_removed_count", ""),
                "dorsal_contour_quality": qc.get("dorsal_contour_quality", ""),
                "ventral_contour_quality": qc.get("ventral_contour_quality", ""),
                "body_midline_smoothness": qc.get("body_midline_smoothness", ""),
                "C1_source": qc.get("C1_source", ""),
                "C2_source": qc.get("C2_source", ""),
                "C3_source": qc.get("C3_source", ""),
                "C4_source": qc.get("C4_source", ""),
            }
        )
        print(f"[{index}/{len(manifest)}] body midline: {image_name}")
    return body_results, pd.DataFrame(qc_rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(STRICT_MANIFEST)
    v041_df = pd.read_csv(PRED_DIR / "v041_automatic_predictions.csv")
    v06_df = pd.read_csv(PRED_DIR / "v06_predictions.csv")
    v041_map = prediction_to_image_map(v041_df)
    v06_map = prediction_to_image_map(v06_df)
    gt_map: dict[str, dict[str, list[float]]] = {}
    for record in manifest.to_dict("records"):
        gt_map[str(record["image_name"])] = corrected_keypoints(resolve_path(record["corrected_json_path"]) or Path(record["corrected_json_path"]))

    body_results, qc_summary = build_body_results(manifest, v06_map)
    qc_summary.to_csv(OUT / "body_midline_qc_summary.csv", index=False, encoding="utf-8-sig")

    error_rows = []
    by_image_rows = []
    stability_rows = []
    axis_rows = []
    for record in manifest.to_dict("records"):
        image_name = str(record["image_name"])
        gt = gt_map.get(image_name, {})
        v041 = v041_map.get(image_name, {})
        v06 = v06_map.get(image_name, {})
        body = body_results.get(image_name, {})
        body_points = {
            "C1_head_axis_point": body.get("C1_geometric"),
            "C2_trunk_axis_point": body.get("C2_geometric"),
            "C3_posterior_trunk_axis_point": body.get("C3_geometric"),
            "C4_peduncle_axis_point": body.get("C4_geometric"),
        }
        draw_visual(record, gt, v041, v06, {**body, **body_points})
        for key in C_KEYS:
            row = {
                "image_name": image_name,
                "specimen_id": record.get("specimen_id", ""),
                "subset_source": record.get("subset_source", ""),
                "included_in_v05_test": bool(record.get("included_in_v05_test", False)),
                "included_in_realworld_review": bool(record.get("included_in_realworld_review", False)),
                "keypoint_name": key,
                "v041_error_mm": dist_mm(v041.get(key), gt.get(key)),
                "v06_error_mm": dist_mm(v06.get(key), gt.get(key)),
                "body_midline_error_mm": dist_mm(body_points.get(key), gt.get(key)),
            }
            error_rows.append(row)
        by_image_rows.append(
            {
                "image_name": image_name,
                "specimen_id": record.get("specimen_id", ""),
                "subset_source": record.get("subset_source", ""),
                "included_in_v05_test": bool(record.get("included_in_v05_test", False)),
                "included_in_realworld_review": bool(record.get("included_in_realworld_review", False)),
                "C1_v041_error_mm": dist_mm(v041.get(C1 := "C1_head_axis_point"), gt.get(C1)),
                "C1_v06_error_mm": dist_mm(v06.get(C1), gt.get(C1)),
                "C1_body_midline_error_mm": dist_mm(body_points.get(C1), gt.get(C1)),
                "C2_v041_error_mm": dist_mm(v041.get(C2 := "C2_trunk_axis_point"), gt.get(C2)),
                "C2_v06_error_mm": dist_mm(v06.get(C2), gt.get(C2)),
                "C2_body_midline_error_mm": dist_mm(body_points.get(C2), gt.get(C2)),
                "C3_v041_error_mm": dist_mm(v041.get(C3 := "C3_posterior_trunk_axis_point"), gt.get(C3)),
                "C3_v06_error_mm": dist_mm(v06.get(C3), gt.get(C3)),
                "C3_body_midline_error_mm": dist_mm(body_points.get(C3), gt.get(C3)),
                "C4_v041_error_mm": dist_mm(v041.get(C4 := "C4_peduncle_axis_point"), gt.get(C4)),
                "C4_v06_error_mm": dist_mm(v06.get(C4), gt.get(C4)),
                "C4_body_midline_error_mm": dist_mm(body_points.get(C4), gt.get(C4)),
                "body_midline_qc_pass": (body.get("body_midline_qc") or {}).get("body_midline_qc_pass", ""),
                "body_midline_qc_reason": (body.get("body_midline_qc") or {}).get("body_midline_qc_reason", ""),
            }
        )
        source_keypoints = {
            "v041": v041,
            "v06": v06,
            "body_midline": with_body_c_points(v06, body),
        }
        for source, points in source_keypoints.items():
            metrics = axis_metrics(points)
            stability_rows.append(
                {
                    "image_name": image_name,
                    "specimen_id": record.get("specimen_id", ""),
                    "source": source,
                    "TL_curve_mm": metrics["TL_curve_mm"],
                    "SL_curve_mm": metrics["SL_curve_mm"],
                    "curvature_index": metrics["curvature_index"],
                    "P7V_valid": metrics["P7V_valid"],
                    "measurements_needs_review": metrics["measurements_needs_review"],
                    "axis_quality_flag": metrics["axis_quality_flag"],
                }
            )
            axis_rows.append(
                {
                    "image_name": image_name,
                    "specimen_id": record.get("specimen_id", ""),
                    "source": source,
                    **metrics,
                }
            )

    error_df = pd.DataFrame(error_rows)
    by_image = pd.DataFrame(by_image_rows)
    by_image.to_csv(OUT / "body_midline_error_by_image.csv", index=False, encoding="utf-8-sig")
    source_rows = []
    for key in C_KEYS:
        subset = error_df[error_df["keypoint_name"] == key]
        body = subset["body_midline_error_mm"]
        v06 = subset["v06_error_mm"]
        v041 = subset["v041_error_mm"]
        source_rows.append(
            {
                "keypoint_name": key,
                "num_images": int(len(subset)),
                "v041_median_error_mm": float(v041.median()),
                "v06_median_error_mm": float(v06.median()),
                "body_midline_median_error_mm": float(body.median()),
                "v041_mean_error_mm": float(v041.mean()),
                "v06_mean_error_mm": float(v06.mean()),
                "body_midline_mean_error_mm": float(body.mean()),
                "body_midline_better_count": int(((body < v06) & (body < v041)).sum()),
                "v06_better_count": int(((v06 < body) & (v06 < v041)).sum()),
                "recommend_use_body_midline": bool(body.median() <= min(v06.median(), v041.median())),
                "notes": "manual C points are subjective; combine with axis-quality metrics",
            }
        )
    source_comparison = pd.DataFrame(source_rows)
    source_comparison.to_csv(OUT / "body_midline_source_comparison.csv", index=False, encoding="utf-8-sig")

    subset_rows = []
    subset_defs = {
        "v05_test_subset": error_df["included_in_v05_test"] == True,
        "realworld_review_confirmed_subset": error_df["included_in_realworld_review"] == True,
    }
    for subset_name, mask in subset_defs.items():
        subset_df = error_df[mask]
        for key in C_KEYS:
            subset = subset_df[subset_df["keypoint_name"] == key]
            if subset.empty:
                continue
            subset_rows.append(
                {
                    "subset": subset_name,
                    "keypoint_name": key,
                    "num_images": int(len(subset)),
                    "v041_median_error_mm": float(subset["v041_error_mm"].median()),
                    "v06_median_error_mm": float(subset["v06_error_mm"].median()),
                    "body_midline_median_error_mm": float(subset["body_midline_error_mm"].median()),
                    "v041_mean_error_mm": float(subset["v041_error_mm"].mean()),
                    "v06_mean_error_mm": float(subset["v06_error_mm"].mean()),
                    "body_midline_mean_error_mm": float(subset["body_midline_error_mm"].mean()),
                    "body_midline_better_count": int(
                        ((subset["body_midline_error_mm"] < subset["v06_error_mm"]) & (subset["body_midline_error_mm"] < subset["v041_error_mm"])).sum()
                    ),
                }
            )
    pd.DataFrame(subset_rows).to_csv(OUT / "body_midline_source_comparison_by_subset.csv", index=False, encoding="utf-8-sig")

    axis_detail = pd.DataFrame(axis_rows)
    axis_summary_rows = []
    for source, group in axis_detail.groupby("source"):
        axis_summary_rows.append(
            {
                "source": source,
                "num_images": int(group["image_name"].nunique()),
                "axis_smoothness_score": float(group["axis_smoothness_score"].mean()),
                "max_axis_bend_angle_deg": float(group["max_axis_bend_angle_deg"].max()),
                "mean_axis_bend_angle_deg": float(group["mean_axis_bend_angle_deg"].mean()),
                "axis_jump_count": int(group["axis_jump_count"].sum()),
                "curvature_index_mean": float(group["curvature_index"].mean()),
                "curvature_index_sd": float(group["curvature_index"].std()),
                "TL_curve_mean_mm": float(group["TL_curve_mm"].mean()),
                "TL_curve_sd_mm": float(group["TL_curve_mm"].std()),
                "P7V_valid_rate": float(group["P7V_valid"].mean()),
                "notes": "lower smoothness score means smoother axis",
            }
        )
    axis_summary = pd.DataFrame(axis_summary_rows)
    axis_summary.to_csv(OUT / "axis_quality_comparison.csv", index=False, encoding="utf-8-sig")

    stability = pd.DataFrame(stability_rows)
    stability.to_csv(OUT / "curve_measurement_stability.csv", index=False, encoding="utf-8-sig")
    write_readme(source_comparison, axis_summary, stability, len(manifest))

    def median_for(key: str, col: str) -> float:
        row = source_comparison[source_comparison["keypoint_name"] == key]
        return float(row[col].iloc[0]) if not row.empty else float("nan")

    print("Finished v0.6.2 body contour midline evaluation.")
    for key in C_KEYS:
        print(
            f"{key}: v041 / v06 / body_midline median = "
            f"{median_for(key, 'v041_median_error_mm'):.3f} / "
            f"{median_for(key, 'v06_median_error_mm'):.3f} / "
            f"{median_for(key, 'body_midline_median_error_mm'):.3f}"
        )
    print("Axis quality:")
    for row in axis_summary.to_dict("records"):
        print(f"{row['source']}: smoothness={row['axis_smoothness_score']:.3f}, P7V={row['P7V_valid_rate']:.3f}")
    body_recs = source_comparison["recommend_use_body_midline"].fillna(False)
    print("Recommendation:")
    print(f"Use body contour midline for C1-C3: {'yes' if bool(body_recs.iloc[:3].all()) else 'no'}")
    print(f"Use P4-P5 midpoint for C4: {'yes' if bool(body_recs.iloc[3]) else 'no'}")
    print(f"Use geometry C points as axis source: {'yes' if axis_summary.sort_values('axis_smoothness_score').iloc[0]['source'] == 'body_midline' else 'no'}")
    print(f"Integrate into experimental mode: {'yes' if bool(body_recs.any()) else 'no'}")
    print("Consider removing C1-C4 from future model training targets: review recommended")


def prediction_to_image_map(frame: pd.DataFrame) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {}
    for row in frame.to_dict("records"):
        out.setdefault(str(row["image_name"]), {})[str(row["keypoint_name"])] = [float(row["x"]), float(row["y"])]
    return out


def write_readme(source_comparison: pd.DataFrame, axis_summary: pd.DataFrame, stability: pd.DataFrame, num_images: int) -> None:
    def fmt(value: Any) -> str:
        try:
            return f"{float(value):.3f}"
        except Exception:
            return ""

    lines = ["# v0.6.2 Body Contour Midline Evaluation", ""]
    lines.append("No model was trained. No corrected labels or Streamlit defaults were modified.")
    lines.append("")
    lines.append(f"Images evaluated: {num_images}")
    lines.append("")
    lines.append("## C-point Error Against Manual Corrected Labels")
    lines.append("")
    lines.append("| keypoint | v041 median | v06 median | body midline median | recommend body midline |")
    lines.append("|---|---:|---:|---:|---|")
    for row in source_comparison.to_dict("records"):
        lines.append(
            f"| {row['keypoint_name']} | {fmt(row['v041_median_error_mm'])} | {fmt(row['v06_median_error_mm'])} | "
            f"{fmt(row['body_midline_median_error_mm'])} | {row['recommend_use_body_midline']} |"
        )
    lines.append("")
    lines.append("Manual C1-C4 labels are subjective, so these errors are diagnostic rather than absolute truth.")
    lines.append("")
    lines.append("## Axis Quality")
    lines.append("")
    lines.append("| source | smoothness score | max bend deg | mean bend deg | jumps | curvature mean | TL sd | P7V valid |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in axis_summary.to_dict("records"):
        lines.append(
            f"| {row['source']} | {fmt(row['axis_smoothness_score'])} | {fmt(row['max_axis_bend_angle_deg'])} | "
            f"{fmt(row['mean_axis_bend_angle_deg'])} | {row['axis_jump_count']} | {fmt(row['curvature_index_mean'])} | "
            f"{fmt(row['TL_curve_sd_mm'])} | {fmt(row['P7V_valid_rate'])} |"
        )
    best_axis = axis_summary.sort_values("axis_smoothness_score").iloc[0]["source"] if not axis_summary.empty else ""
    c2_row = source_comparison[source_comparison["keypoint_name"] == "C2_trunk_axis_point"]
    c4_row = source_comparison[source_comparison["keypoint_name"] == "C4_peduncle_axis_point"]
    lines.append("")
    lines.append("## Answers")
    lines.append("")
    lines.append(f"1. body contour midline improves C1-C4: {'partly' if source_comparison['recommend_use_body_midline'].any() else 'no'}")
    lines.append(
        "2. C2 improvement: "
        + (
            "yes"
            if not c2_row.empty and bool(c2_row.iloc[0]["recommend_use_body_midline"])
            else "no"
        )
    )
    lines.append(
        "3. C4=P4-P5 midpoint improvement: "
        + (
            "yes"
            if not c4_row.empty and bool(c4_row.iloc[0]["recommend_use_body_midline"])
            else "no"
        )
    )
    lines.append(f"4. smoothest axis source: {best_axis}")
    lines.append("5. TL/SL/curvature stability is in `curve_measurement_stability.csv`; use the axis table for aggregate TL_curve SD.")
    lines.append("6. de-finned contour stability is reported by `body_midline_qc_summary.csv`.")
    lines.append("7. this first version is intended for straight or mildly curved fish; high-curvature fish should remain suggestion/QC only.")
    lines.append(
        "8. integrate into v0.6 experimental: "
        + ("yes, as optional axis-source/QC" if source_comparison["recommend_use_body_midline"].any() else "not yet")
    )
    lines.append(
        "9. removing C1-C4 from future model training targets: worth considering, but only after visual review and stronger curved-fish handling."
    )
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    for name in [
        "body_midline_source_comparison.csv",
        "body_midline_source_comparison_by_subset.csv",
        "body_midline_error_by_image.csv",
        "body_midline_qc_summary.csv",
        "axis_quality_comparison.csv",
        "curve_measurement_stability.csv",
        "visual_comparisons/",
    ]:
        lines.append(f"- `{name}`")
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
