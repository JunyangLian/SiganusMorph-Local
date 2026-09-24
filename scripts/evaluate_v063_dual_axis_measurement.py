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
from siganusmorph.dual_axis_measurement import compare_dual_axis_measurements
from siganusmorph.image_utils import load_image_file
from siganusmorph.keypointwise_hybrid_selector import KEYPOINT_NAMES
from siganusmorph.segmentation import segment_fish_from_blue_board


MM_PER_PIXEL = 0.1
OUT = PROJECT_ROOT / "results" / "model_eval" / "v0.6.3_dual_axis_measurement"
VIS_OUT = OUT / "visual_comparisons"
SAME_SET = PROJECT_ROOT / "results" / "model_eval" / "v0.6_keypointwise_hybrid_selector" / "same_set_comparison"
STRICT_MANIFEST = SAME_SET / "strict_union_manifest.csv"
PRED_DIR = SAME_SET / "predictions"
V06_PREDICTIONS = PRED_DIR / "v06_predictions.csv"


def resolve_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def xy_from(value: Any) -> tuple[float, float] | None:
    if isinstance(value, Mapping):
        if "x" in value and "y" in value:
            return float(value["x"]), float(value["y"])
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return None


def prediction_to_image_map(frame: pd.DataFrame) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {}
    for row in frame.to_dict("records"):
        out.setdefault(str(row["image_name"]), {})[str(row["keypoint_name"])] = [float(row["x"]), float(row["y"])]
    return out


def flatten_measurement(prefix: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_TL_curve_mm": payload.get("TL_curve_mm", np.nan),
        f"{prefix}_SL_curve_mm": payload.get("SL_curve_mm", np.nan),
        f"{prefix}_curvature_index": payload.get("curvature_index", np.nan),
        f"{prefix}_axis_smoothness": payload.get("axis_smoothness_score", np.nan),
        f"{prefix}_axis_bend_angle_deg": payload.get("axis_bend_angle_deg", np.nan),
        f"{prefix}_max_axis_deviation_mm": payload.get("max_axis_deviation_mm", np.nan),
        f"{prefix}_body_depth_local_normal_mm": payload.get("body_depth_local_normal_mm", np.nan),
        f"{prefix}_peduncle_depth_local_normal_mm": payload.get("peduncle_depth_local_normal_mm", np.nan),
        f"{prefix}_P7V_valid": payload.get("P7V_valid", False),
        f"{prefix}_measurements_needs_review": payload.get("measurements_needs_review", True),
        f"{prefix}_curvature_qc_level": payload.get("curvature_qc_level", ""),
        f"{prefix}_review_reason": payload.get("review_reason", ""),
    }


def axis_points(payload: Mapping[str, Any]) -> list[list[float]]:
    points = payload.get("axis_points", [])
    return [[float(p[0]), float(p[1])] for p in points if isinstance(p, (list, tuple)) and len(p) >= 2]


def finite_values(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def summarize_metric(frame: pd.DataFrame, metric: str) -> dict[str, Any]:
    model_col = f"model_{metric}"
    body_col = f"body_midline_{metric}"
    model_values = finite_values(frame[model_col]) if model_col in frame else np.asarray([], dtype=float)
    body_values = finite_values(frame[body_col]) if body_col in frame else np.asarray([], dtype=float)
    diff_values = np.abs(body_values - model_values) if len(body_values) == len(model_values) and len(body_values) else np.asarray([], dtype=float)
    recommendation = "body_midline_axis_if_qc_pass" if metric == "axis_smoothness" and len(body_values) and np.nanmean(body_values) < np.nanmean(model_values) else "compare_per_image"
    return {
        "metric": metric,
        "model_axis_mean": float(np.nanmean(model_values)) if len(model_values) else np.nan,
        "body_midline_axis_mean": float(np.nanmean(body_values)) if len(body_values) else np.nan,
        "model_axis_sd": float(np.nanstd(model_values, ddof=1)) if len(model_values) > 1 else np.nan,
        "body_midline_axis_sd": float(np.nanstd(body_values, ddof=1)) if len(body_values) > 1 else np.nan,
        "mean_abs_difference": float(np.nanmean(diff_values)) if len(diff_values) else np.nan,
        "recommendation": recommendation,
    }


def draw_visual(record: Mapping[str, Any], keypoints: Mapping[str, Any], body_result: Mapping[str, Any], dual: Mapping[str, Any]) -> None:
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

    def line(points: list[list[float]], color: tuple[int, int, int, int], width: int = 3) -> None:
        pts = [(float(p[0]) * scale, float(p[1]) * scale) for p in points if len(p) >= 2]
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=width)

    def dot(point: Any, color: tuple[int, int, int, int], label: str, r: int = 5) -> None:
        xy = sxy(point)
        if xy is None:
            return
        x, y = xy
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=(0, 0, 0, 230), width=2)
        draw.text((x + r + 2, y - r), label, fill=color, font=font)

    model_axis = dual.get("model_axis_measurements", {})
    body_axis = dual.get("body_midline_axis_measurements", {})
    line(axis_points(model_axis), (255, 145, 35, 230), 4)
    line(axis_points(body_axis), (0, 115, 255, 235), 4)
    line(body_result.get("body_midline_points", [])[::4] if isinstance(body_result.get("body_midline_points"), list) else [], (0, 90, 255, 120), 2)
    line(body_result.get("dorsal_body_contour", []), (80, 210, 255, 160), 2)
    line(body_result.get("ventral_body_contour", []), (80, 210, 255, 160), 2)

    for key, label in [
        ("P4_peduncle_start_midpoint", "P4"),
        ("P5_caudal_base_midpoint", "P5"),
        ("P6_caudal_fork_midpoint", "P6"),
        ("P7U_caudal_fin_upper_tip", "P7U"),
        ("P7L_caudal_fin_lower_tip", "P7L"),
    ]:
        dot(keypoints.get(key), (250, 250, 250, 235), label, 4)
    for key, label in [
        ("C1_geometric", "C1g"),
        ("C2_geometric", "C2g"),
        ("C3_geometric", "C3g"),
        ("C4_geometric", "C4g"),
    ]:
        dot(body_result.get(key), (0, 115, 255, 230), label, 5)

    model_p7v = ((model_axis.get("derived_points") or {}).get("P7V_caudal_fin_posterior_endpoint") or {}).get("point")
    if model_p7v:
        dot(model_p7v, (155, 50, 255, 240), "P7V", 7)

    cmp_payload = dual.get("dual_axis_comparison", {})
    rec = dual.get("axis_recommendation", {})
    text_lines = [
        f"{record.get('image_name')} dual-axis measurement",
        f"model smooth={float(model_axis.get('axis_smoothness_score', np.nan)):.2f}  body smooth={float(body_axis.get('axis_smoothness_score', np.nan)):.2f}",
        f"TL diff={float(cmp_payload.get('TL_curve_diff_mm', np.nan)):.2f} mm  SL diff={float(cmp_payload.get('SL_curve_diff_mm', np.nan)):.2f} mm",
        f"recommend={rec.get('recommended_measurement_axis')}  {rec.get('recommendation_reason','')}",
    ]
    box_h = 22 * len(text_lines) + 8
    draw.rectangle([0, 0, 980, box_h], fill=(0, 0, 0, 220))
    for i, line_text in enumerate(text_lines):
        fill = (255, 255, 255, 255) if i == 0 else (220, 235, 255, 255)
        draw.text((8, 6 + i * 22), line_text, fill=fill, font=font)

    VIS_OUT.mkdir(parents=True, exist_ok=True)
    display.save(VIS_OUT / f"{Path(str(record['image_name'])).stem}_dual_axis_compare.png")


def write_readme(comparison: pd.DataFrame, summary: pd.DataFrame, num_images: int) -> None:
    def fmt(value: Any) -> str:
        try:
            return f"{float(value):.3f}"
        except Exception:
            return ""

    model_smooth = comparison["model_axis_smoothness"].mean()
    body_smooth = comparison["body_midline_axis_smoothness"].mean()
    tl_abs = comparison["TL_curve_diff_mm"].abs()
    sl_abs = comparison["SL_curve_diff_mm"].abs()
    disagreement = int(comparison["dual_axis_disagreement"].fillna(False).sum())
    body_rec = int((comparison["recommended_measurement_axis"] == "body_midline_axis").sum())
    model_rec = int((comparison["recommended_measurement_axis"] == "model_axis").sum())
    manual_rec = int((comparison["recommended_measurement_axis"] == "manual_review_required").sum())

    largest = comparison.assign(abs_tl=comparison["TL_curve_diff_mm"].abs(), abs_sl=comparison["SL_curve_diff_mm"].abs())
    largest = largest.sort_values(["abs_tl", "abs_sl"], ascending=False).head(8)

    lines = ["# v0.6.3 Dual-Axis Measurement Comparison", ""]
    lines.append("No model was trained. Corrected labels and Streamlit defaults were not modified.")
    lines.append("")
    lines.append(f"Images evaluated: {num_images}")
    lines.append("")
    lines.append("## Main Result")
    lines.append("")
    lines.append(f"- Mean model-axis smoothness: `{fmt(model_smooth)}`")
    lines.append(f"- Mean body-midline-axis smoothness: `{fmt(body_smooth)}`")
    lines.append(f"- TL_curve absolute difference, mean / median / max: `{fmt(tl_abs.mean())}` / `{fmt(tl_abs.median())}` / `{fmt(tl_abs.max())}` mm")
    lines.append(f"- SL_curve absolute difference, mean / median / max: `{fmt(sl_abs.mean())}` / `{fmt(sl_abs.median())}` / `{fmt(sl_abs.max())}` mm")
    lines.append(f"- Dual-axis disagreement cases: `{disagreement}`")
    lines.append(f"- Recommendation counts: body_midline_axis `{body_rec}`, model_axis `{model_rec}`, manual_review_required `{manual_rec}`")
    lines.append("")
    lines.append("## Summary By Metric")
    lines.append("")
    lines.append("| metric | model mean | body-midline mean | model SD | body SD | mean abs diff |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in summary.to_dict("records"):
        lines.append(
            f"| {row['metric']} | {fmt(row['model_axis_mean'])} | {fmt(row['body_midline_axis_mean'])} | "
            f"{fmt(row['model_axis_sd'])} | {fmt(row['body_midline_axis_sd'])} | {fmt(row['mean_abs_difference'])} |"
        )
    lines.append("")
    lines.append("## Largest TL/SL Axis Differences")
    lines.append("")
    lines.append("| image | specimen | subset | TL diff | SL diff | recommended axis | reason |")
    lines.append("|---|---|---|---:|---:|---|---|")
    for row in largest.to_dict("records"):
        lines.append(
            f"| {row['image_name']} | {row['specimen_id']} | {row['subset_source']} | "
            f"{fmt(row['TL_curve_diff_mm'])} | {fmt(row['SL_curve_diff_mm'])} | "
            f"{row['recommended_measurement_axis']} | {row['recommendation_reason']} |"
        )
    lines.append("")
    lines.append("## Answers")
    lines.append("")
    lines.append(f"1. body_midline_axis is smoother than model_axis: {'yes' if body_smooth < model_smooth else 'no'}")
    lines.append(
        "2. TL_curve / SL_curve stability: body_midline_axis is promising when QC passes, but disagreement cases should remain review-gated."
    )
    lines.append(
        f"3. Typical TL/SL differences: median TL `{fmt(tl_abs.median())}` mm, median SL `{fmt(sl_abs.median())}` mm."
    )
    lines.append("4. Largest disagreement images are listed above and visualized in `visual_comparisons/`.")
    lines.append(
        "5. Default measurement-axis recommendation: do not replace globally yet; use body_midline_axis where it is smoother and agrees with model axis."
    )
    lines.append(
        "6. Keep C1-C4 as keypoints for now, but treat them as subjective axis helpers rather than hard anatomy points."
    )
    lines.append(
        "7. Future work: consider removing C1-C4 from heatmap training targets only after a dedicated geometry-axis workflow is validated on curved specimens."
    )
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    for name in [
        "dual_axis_measurement_comparison.csv",
        "dual_axis_summary.csv",
        "visual_comparisons/",
    ]:
        lines.append(f"- `{name}`")
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    VIS_OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(STRICT_MANIFEST)
    v06_predictions = prediction_to_image_map(pd.read_csv(V06_PREDICTIONS))

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(manifest.to_dict("records"), start=1):
        image_name = str(record["image_name"])
        keypoints = v06_predictions.get(image_name, {})
        warped_path = resolve_path(record.get("warped_image_path"))
        if not keypoints or warped_path is None or not warped_path.exists():
            continue
        image = load_image_file(warped_path)
        fish_mask, fish_bbox, segmentation_quality = segment_fish_from_blue_board(image)
        body_result = estimate_body_contour_midline_points(
            fish_mask,
            keypoints,
            image.shape,
            config={"mm_per_pixel": MM_PER_PIXEL},
        )
        dual = compare_dual_axis_measurements(
            fish_mask,
            keypoints,
            body_result,
            image.shape,
            {"mm_per_pixel": MM_PER_PIXEL},
            config={"mm_per_pixel": MM_PER_PIXEL},
        )
        model_axis = dual["model_axis_measurements"]
        body_axis = dual["body_midline_axis_measurements"]
        cmp_payload = dual["dual_axis_comparison"]
        rec = dual["axis_recommendation"]
        row = {
            "image_name": image_name,
            "specimen_id": record.get("specimen_id", ""),
            "subset_source": record.get("subset_source", ""),
            "included_in_v05_test": bool(record.get("included_in_v05_test", False)),
            "included_in_realworld_review": bool(record.get("included_in_realworld_review", False)),
            "segmentation_quality": segmentation_quality,
            "fish_bbox": json.dumps(fish_bbox),
            **flatten_measurement("model", model_axis),
            **flatten_measurement("body_midline", body_axis),
            "TL_curve_diff_mm": cmp_payload.get("TL_curve_diff_mm", np.nan),
            "SL_curve_diff_mm": cmp_payload.get("SL_curve_diff_mm", np.nan),
            "curvature_index_diff": cmp_payload.get("curvature_index_diff", np.nan),
            "dual_axis_disagreement": cmp_payload.get("dual_axis_disagreement", False),
            "recommended_measurement_axis": rec.get("recommended_measurement_axis", ""),
            "recommendation_reason": rec.get("recommendation_reason", ""),
        }
        rows.append(row)
        draw_visual(record, keypoints, body_result, dual)
        print(f"[{index}/{len(manifest)}] dual-axis: {image_name}")

    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "dual_axis_measurement_comparison.csv", index=False, encoding="utf-8-sig")

    metrics = [
        "TL_curve_mm",
        "SL_curve_mm",
        "curvature_index",
        "axis_smoothness",
        "axis_bend_angle_deg",
        "body_depth_local_normal_mm",
        "peduncle_depth_local_normal_mm",
    ]
    summary = pd.DataFrame([summarize_metric(comparison, metric) for metric in metrics])
    summary.to_csv(OUT / "dual_axis_summary.csv", index=False, encoding="utf-8-sig")
    write_readme(comparison, summary, len(comparison))

    model_smooth = comparison["model_axis_smoothness"].mean()
    body_smooth = comparison["body_midline_axis_smoothness"].mean()
    tl_abs = comparison["TL_curve_diff_mm"].abs()
    sl_abs = comparison["SL_curve_diff_mm"].abs()
    disagreements = int(comparison["dual_axis_disagreement"].fillna(False).sum())
    use_body = int((comparison["recommended_measurement_axis"] == "body_midline_axis").sum())
    keep_c = "yes"
    remove_c = "not yet"

    print("Finished v0.6.3 dual-axis measurement comparison.")
    print("Axis smoothness:")
    print(f"model_axis: {model_smooth:.3f}")
    print(f"body_midline_axis: {body_smooth:.3f}")
    print("TL_curve difference:")
    print(f"mean / median / max: {tl_abs.mean():.3f} / {tl_abs.median():.3f} / {tl_abs.max():.3f} mm")
    print("SL_curve difference:")
    print(f"mean / median / max: {sl_abs.mean():.3f} / {sl_abs.median():.3f} / {sl_abs.max():.3f} mm")
    print("Dual-axis disagreement cases:")
    print(disagreements)
    print("Recommendation:")
    print(f"Use body_midline_axis as default measurement axis: {'yes' if use_body > len(comparison) / 2 else 'no'}")
    print(f"Keep C1-C4 as predicted keypoints: {keep_c}")
    print(f"Consider removing C1-C4 from future training targets: {remove_c}")


if __name__ == "__main__":
    main()

