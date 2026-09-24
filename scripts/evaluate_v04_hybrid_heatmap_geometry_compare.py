"""Evaluate v0.4 hybrid heatmap-geometry preannotation on the 30-image sample.

This script does not train models, overwrite corrected labels, or modify the
Streamlit preannotation workflow. It only runs a benchmark draft that combines
heatmap-U-Net predictions with the existing v0.3.4 geometry/QC pipeline.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_v031_preannotation_compare import (  # noqa: E402
    DEFAULT_MM_PER_PIXEL,
    KEYPOINT_KEYS,
    collect_available_records,
    distance_px,
    records_from_sample_manifest,
    safe_measurements,
    xy,
)
from siganusmorph.heatmap_preannotation import default_heatmap_model, predict_warped_keypoints_heatmap_unet  # noqa: E402
from siganusmorph.hybrid_point_selector import select_hybrid_keypoints  # noqa: E402
from siganusmorph.image_utils import ensure_rgb, load_image_file  # noqa: E402
from siganusmorph.preannotation import default_preannotation_model, preannotate_warped_image  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "results" / "model_eval" / "v0.4_hybrid_heatmap_geometry_compare_30"
V034_DIR = PROJECT_ROOT / "results" / "model_eval" / "v0.3.4_local_structure_refinement_compare_30"
V033_DIR = PROJECT_ROOT / "results" / "model_eval" / "v0.3.3_geometry_refinement_compare_30"
EXTERNAL_SUMMARY = PROJECT_ROOT / "results" / "model_eval" / "external_benchmark" / "benchmark_summary_v0.1.csv"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def font(size: int = 20) -> ImageFont.ImageFont:
    for candidate in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _scale_point(point: Any, scale: float) -> tuple[float, float] | None:
    p = xy(point)
    if p is None:
        return None
    return p[0] * scale, p[1] * scale


def _draw_label(draw: ImageDraw.ImageDraw, point: tuple[float, float], text: str, fill: tuple[int, int, int]) -> None:
    fnt = font(18)
    x, y = point
    for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        draw.text((x + ox, y + oy), text, font=fnt, fill=(0, 0, 0))
    draw.text((x, y), text, font=fnt, fill=fill)


def _draw_star(draw: ImageDraw.ImageDraw, point: tuple[float, float], radius: int, fill: tuple[int, int, int]) -> None:
    x0, y0 = point
    pts = []
    for i in range(10):
        angle = -math.pi / 2 + i * math.pi / 5
        r = radius if i % 2 == 0 else radius * 0.45
        pts.append((x0 + math.cos(angle) * r, y0 + math.sin(angle) * r))
    draw.polygon(pts, fill=fill, outline=(70, 20, 110))


def save_hybrid_visual(
    image: np.ndarray,
    output_path: Path,
    *,
    manual_points: Mapping[str, Any],
    heatmap_points: Mapping[str, Any],
    v034_points: Mapping[str, Any],
    suggestions: Mapping[str, Any],
    hybrid_points: Mapping[str, Any],
    p7v_info: Mapping[str, Any],
    image_summary: Mapping[str, Any],
    point_sources: Mapping[str, str],
) -> None:
    pil = Image.fromarray(ensure_rgb(image))
    scale = min(1.0, 2200 / max(1, pil.width))
    if scale < 1:
        pil = pil.resize((int(round(pil.width * scale)), int(round(pil.height * scale))), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(pil)
    radius = max(5, int(round(min(pil.size) / 260)))
    large_keys = set(str(image_summary.get("large_error_keypoints", "")).split(";"))

    # Suggestions: cyan.
    for key, point in suggestions.items():
        p = _scale_point(point, scale)
        if p is None:
            continue
        draw.ellipse((p[0] - radius, p[1] - radius, p[0] + radius, p[1] + radius), fill=(0, 220, 220), outline=(0, 70, 90), width=2)

    for key in KEYPOINT_KEYS:
        code = key.split("_", 1)[0]
        manual = _scale_point(manual_points.get(key), scale)
        heat = _scale_point(heatmap_points.get(key), scale)
        v034 = _scale_point(v034_points.get(key), scale)
        hybrid = _scale_point(hybrid_points.get(key), scale)
        if manual is not None and hybrid is not None:
            draw.line((manual[0], manual[1], hybrid[0], hybrid[1]), fill=(255, 255, 255), width=1)
        if manual is not None:
            draw.ellipse((manual[0] - radius, manual[1] - radius, manual[0] + radius, manual[1] + radius), fill=(60, 220, 90), outline=(10, 70, 20), width=2)
        if heat is not None:
            draw.ellipse((heat[0] - radius, heat[1] - radius, heat[0] + radius, heat[1] + radius), outline=(210, 80, 255), width=3)
        if v034 is not None:
            draw.ellipse((v034[0] - radius, v034[1] - radius, v034[0] + radius, v034[1] + radius), outline=(255, 145, 30), width=3)
        if hybrid is not None:
            draw.ellipse((hybrid[0] - radius, hybrid[1] - radius, hybrid[0] + radius, hybrid[1] + radius), fill=(245, 45, 55), outline=(90, 0, 0), width=2)
            _draw_label(draw, (hybrid[0] + radius + 2, hybrid[1] - radius - 2), code, (255, 90, 90))
        if key in large_keys:
            anchor = hybrid or manual
            if anchor is not None:
                draw.ellipse((anchor[0] - radius - 10, anchor[1] - radius - 10, anchor[0] + radius + 10, anchor[1] + radius + 10), outline=(255, 230, 0), width=5)

    p7v = p7v_info.get("point")
    p7v_scaled = _scale_point(p7v, scale)
    if p7v_scaled is not None:
        if p7v_info.get("p7v_valid") is False:
            size = radius + 12
            draw.line((p7v_scaled[0] - size, p7v_scaled[1] - size, p7v_scaled[0] + size, p7v_scaled[1] + size), fill=(255, 0, 0), width=5)
            draw.line((p7v_scaled[0] - size, p7v_scaled[1] + size, p7v_scaled[0] + size, p7v_scaled[1] - size), fill=(255, 0, 0), width=5)
            _draw_label(draw, (p7v_scaled[0] + size + 3, p7v_scaled[1] - size), "P7V invalid", (255, 0, 0))
        else:
            _draw_star(draw, p7v_scaled, radius + 9, (210, 90, 255))
            _draw_label(draw, (p7v_scaled[0] + radius + 10, p7v_scaled[1] - radius), "P7V", (210, 90, 255))

    source_counts = Counter(point_sources.values())
    source_text = "; ".join(f"{k}:{v}" for k, v in source_counts.most_common(4))
    lines = [
        f"{image_summary.get('image_name', '')} | {image_summary.get('specimen_id', '')} | {image_summary.get('split', '')}",
        f"mean {float(image_summary.get('mean_error_mm', 0) or 0):.2f} mm | median {float(image_summary.get('median_error_mm', 0) or 0):.2f} mm",
        f"large: {image_summary.get('large_error_keypoints', '') or 'none'}",
        f"sources: {source_text}",
        f"QC: {str(image_summary.get('review_reason', ''))[:120]}",
        "green=manual | purple=heatmap | orange=v0.3.4 | cyan=rule | red=hybrid",
    ]
    fnt = font(21)
    text_width = max(draw.textbbox((0, 0), line, font=fnt)[2] for line in lines)
    line_h = draw.textbbox((0, 0), "Ag", font=fnt)[3] + 8
    draw.rectangle((8, 8, min(pil.width - 8, text_width + 24), 16 + line_h * len(lines)), fill=(0, 0, 0))
    for i, line in enumerate(lines):
        draw.text((16, 12 + i * line_h), line, font=fnt, fill=(255, 255, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(output_path, optimize=True)


def summarize_by_point(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in summary_df.groupby("keypoint_name"):
        rows.append(
            {
                "keypoint_name": key,
                "num_images": len(group),
                "mean_error_px": group["error_px"].mean(),
                "median_error_px": group["error_px"].median(),
                "max_error_px": group["error_px"].max(),
                "mean_error_mm": group["error_mm"].mean(),
                "median_error_mm": group["error_mm"].median(),
                "max_error_mm": group["error_mm"].max(),
                "large_error_rate": group["is_large_error"].mean(),
                "most_common_point_source": Counter(group["auto_point_source"].fillna("").astype(str)).most_common(1)[0][0],
                "notes": "",
            }
        )
    return pd.DataFrame(rows).sort_values("median_error_mm", ascending=False)


def source_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    totals = summary_df.groupby("keypoint_name").size().to_dict()
    for (key, source), group in summary_df.groupby(["keypoint_name", "auto_point_source"]):
        total = max(1, totals.get(key, len(group)))
        rows.append(
            {
                "keypoint_name": key,
                "source_type": source,
                "count": len(group),
                "percentage": round(100 * len(group) / total, 2),
                "mean_error_mm": group["error_mm"].mean(),
                "median_error_mm": group["error_mm"].median(),
            }
        )
    return pd.DataFrame(rows).sort_values(["keypoint_name", "count"], ascending=[True, False])


def hybrid_source_comparison(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out = []
    for key, group in df.groupby("keypoint_name"):
        medians = {
            "v034": group["v034_error_mm"].median(),
            "heatmap": group["heatmap_error_mm"].median(),
            "mask_or_rule": group["mask_or_rule_error_mm"].median(),
            "hybrid": group["hybrid_error_mm"].median(),
        }
        valid_sources = {k: v for k, v in medians.items() if pd.notna(v)}
        best_source = min(valid_sources, key=valid_sources.get) if valid_sources else ""
        if key == "P1_snout_tip":
            recommend = "mask_left_boundary"
        elif key == "P8_body_depth_dorsal":
            recommend = "v0.3.4_enhanced_or_rule_QC"
        elif key in {"P9_body_depth_ventral", "P10_peduncle_depth_dorsal", "P11_peduncle_depth_ventral"}:
            recommend = "heatmap_with_QC" if medians["heatmap"] < medians["v034"] else "v0.3.4_enhanced_or_rule_QC"
        else:
            recommend = best_source
        out.append(
            {
                "keypoint_name": key,
                "num_images": len(group),
                "v034_median_error_mm": medians["v034"],
                "heatmap_median_error_mm": medians["heatmap"],
                "mask_or_rule_median_error_mm": medians["mask_or_rule"],
                "hybrid_median_error_mm": medians["hybrid"],
                "best_source": best_source,
                "recommend_default_source": recommend,
            }
        )
    return pd.DataFrame(out).sort_values("keypoint_name")


def _base_summary(path: Path) -> tuple[float, float, dict[str, float]]:
    if not path.exists():
        return np.nan, np.nan, {}
    df = pd.read_csv(path)
    return float(df["error_mm"].median()), float(df["error_mm"].mean()), df.groupby("keypoint_name")["error_mm"].median().to_dict()


def _heatmap_best_summary() -> tuple[float, float, dict[str, float]]:
    path = PROJECT_ROOT / "results" / "model_eval" / "external_benchmark" / "heatmap_unet_v0.1_best" / "keypoint_error_summary.csv"
    return _base_summary(path)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "visual_comparisons").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "debug_json").mkdir(parents=True, exist_ok=True)

    records = collect_available_records()
    manifest = V034_DIR / "sample_manifest.csv"
    if not manifest.exists():
        manifest = V033_DIR / "sample_manifest.csv"
    selected = records_from_sample_manifest(records, manifest)
    pd.DataFrame(
        [
            {
                "image_name": r["image_name"],
                "specimen_id": r["specimen_id"],
                "split": r["split"],
                "warped_image_path": str(r["warped_image_path"]),
                "corrected_json_path": str(r["corrected_json_path"]),
            }
            for r in selected
        ]
    ).to_csv(OUTPUT_DIR / "sample_manifest.csv", index=False, encoding="utf-8-sig")

    v034_model = default_preannotation_model(PROJECT_ROOT)
    heatmap_model = default_heatmap_model(PROJECT_ROOT)

    summary_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    source_compare_rows: list[dict[str, Any]] = []

    for index, record in enumerate(selected, start=1):
        image_name = str(record["image_name"])
        print(f"[{index}/{len(selected)}] evaluating hybrid {image_name}")
        payload = record["payload"]
        manual_points = payload.get("corrected_keypoints") or payload.get("keypoints") or {}
        mm_per_pixel = float(payload.get("mm_per_pixel") or DEFAULT_MM_PER_PIXEL)
        image = load_image_file(record["warped_image_path"])

        v034_result = preannotate_warped_image(image, image_name, PROJECT_ROOT, v034_model)
        v034_metadata = v034_result.get("metadata", {})
        v034_points = v034_metadata.get("corrected_keypoints") or v034_result.get("keypoints") or {}
        v034_model_raw = v034_metadata.get("model_keypoints_raw", {}) if isinstance(v034_metadata.get("model_keypoints_raw", {}), Mapping) else {}
        mask_suggestions = v034_metadata.get("mask_suggestions", {}) if isinstance(v034_metadata.get("mask_suggestions", {}), Mapping) else {}
        geometric_suggestions = v034_metadata.get("geometric_suggestions", {}) if isinstance(v034_metadata.get("geometric_suggestions", {}), Mapping) else {}
        local_suggestions = v034_metadata.get("local_structure_suggestions", {}) if isinstance(v034_metadata.get("local_structure_suggestions", {}), Mapping) else {}
        suggestions = dict(mask_suggestions)
        suggestions.update(geometric_suggestions)
        suggestions.update(local_suggestions)
        v034_qc = v034_metadata.get("qc_results", {}) if isinstance(v034_metadata.get("qc_results", {}), Mapping) else {}
        fish_mask = v034_result.get("fish_mask")
        fish_bbox = tuple(v034_result.get("fish_bbox")) if v034_result.get("fish_bbox") is not None else None

        heatmap_result = predict_warped_keypoints_heatmap_unet(
            image,
            image_name,
            PROJECT_ROOT,
            model_path=heatmap_model,
        )
        heatmap_points = heatmap_result["heatmap_keypoints_warped"]
        heatmap_conf = heatmap_result["heatmap_confidences"]

        hybrid_result = select_hybrid_keypoints(
            heatmap_points,
            heatmap_conf,
            v034_model_raw,
            mask_suggestions,
            {**geometric_suggestions, **local_suggestions},
            {**v034_qc, **v034_metadata.get("tail_geometry_qc", {})},
            fish_mask=fish_mask,
            fish_bbox=fish_bbox,
            mm_per_pixel=mm_per_pixel,
            v034_keypoints=v034_points,
        )
        hybrid_points = hybrid_result["hybrid_keypoints"]
        point_sources = hybrid_result["point_sources"]
        hybrid_qc = hybrid_result["hybrid_qc_results"]
        review_reason = hybrid_qc.get("hybrid_review_reason", "")

        image_errors: list[float] = []
        large_keys: list[str] = []
        errors_by_keypoint: dict[str, Any] = {}

        for key in KEYPOINT_KEYS:
            manual = xy(manual_points.get(key))
            hybrid = xy(hybrid_points.get(key))
            heat = xy(heatmap_points.get(key))
            v034 = xy(v034_points.get(key))
            mask_or_rule = xy(suggestions.get(key))
            err_px = distance_px(hybrid_points.get(key), manual_points.get(key))
            heat_err_px = distance_px(heatmap_points.get(key), manual_points.get(key))
            v034_err_px = distance_px(v034_points.get(key), manual_points.get(key))
            rule_err_px = distance_px(suggestions.get(key), manual_points.get(key))
            err_mm = err_px * mm_per_pixel if err_px is not None else np.nan
            is_large = bool(pd.notna(err_mm) and err_mm > 10.0)
            if pd.notna(err_mm):
                image_errors.append(float(err_mm))
            if is_large:
                large_keys.append(key)
            per_qc = hybrid_qc.get("per_point_qc", {}).get(key, {}) if isinstance(hybrid_qc.get("per_point_qc", {}), Mapping) else {}
            row = {
                "image_name": image_name,
                "specimen_id": record["specimen_id"],
                "split": record["split"],
                "keypoint_name": key,
                "manual_x": manual[0] if manual else np.nan,
                "manual_y": manual[1] if manual else np.nan,
                "heatmap_x": heat[0] if heat else np.nan,
                "heatmap_y": heat[1] if heat else np.nan,
                "heatmap_confidence": heatmap_conf.get(key, np.nan),
                "v034_x": v034[0] if v034 else np.nan,
                "v034_y": v034[1] if v034 else np.nan,
                "mask_suggestion_x": mask_or_rule[0] if mask_or_rule else np.nan,
                "mask_suggestion_y": mask_or_rule[1] if mask_or_rule else np.nan,
                "auto_x": hybrid[0] if hybrid else np.nan,
                "auto_y": hybrid[1] if hybrid else np.nan,
                "hybrid_x": hybrid[0] if hybrid else np.nan,
                "hybrid_y": hybrid[1] if hybrid else np.nan,
                "error_px": err_px if err_px is not None else np.nan,
                "error_mm": err_mm,
                "is_large_error": is_large,
                "auto_point_source": point_sources.get(key, ""),
                "point_source": point_sources.get(key, ""),
                "qc_flag": per_qc.get("qc_flag", ""),
                "review_reason": per_qc.get("review_reason", review_reason),
                "model_confidence": heatmap_conf.get(key, np.nan),
            }
            summary_rows.append(row)
            errors_by_keypoint[key] = row
            source_compare_rows.append(
                {
                    "image_name": image_name,
                    "specimen_id": record["specimen_id"],
                    "split": record["split"],
                    "keypoint_name": key,
                    "v034_error_mm": v034_err_px * mm_per_pixel if v034_err_px is not None else np.nan,
                    "heatmap_error_mm": heat_err_px * mm_per_pixel if heat_err_px is not None else np.nan,
                    "mask_or_rule_error_mm": rule_err_px * mm_per_pixel if rule_err_px is not None else np.nan,
                    "hybrid_error_mm": err_mm,
                    "hybrid_source": point_sources.get(key, ""),
                }
            )

        auto_measurements = safe_measurements(hybrid_points, mm_per_pixel)
        p7v_info = hybrid_qc.get("p7v", {}) if isinstance(hybrid_qc.get("p7v", {}), Mapping) else {}
        image_row = {
            "image_name": image_name,
            "specimen_id": record["specimen_id"],
            "split": record["split"],
            "mean_error_mm": float(np.mean(image_errors)) if image_errors else np.nan,
            "median_error_mm": float(np.median(image_errors)) if image_errors else np.nan,
            "max_error_mm": float(np.max(image_errors)) if image_errors else np.nan,
            "num_large_error_points": len(large_keys),
            "large_error_keypoints": ";".join(large_keys),
            "needs_review": bool(review_reason),
            "review_reason": review_reason,
            "p7v_valid": p7v_info.get("p7v_valid", ""),
            "hybrid_qc_pass": hybrid_qc.get("hybrid_qc_pass", ""),
            "hybrid_review_reason": review_reason,
            "auto_TL_final_mm": auto_measurements.get("TL_final_mm", np.nan),
            "auto_SL_final_mm": auto_measurements.get("SL_final_mm", np.nan),
        }
        image_rows.append(image_row)
        qc_rows.append(
            {
                "image_name": image_name,
                "specimen_id": record["specimen_id"],
                "split": record["split"],
                "segmentation_success": v034_metadata.get("segmentation_success", ""),
                "segmentation_quality": v034_metadata.get("segmentation_quality", ""),
                "tail_qc_pass": v034_qc.get("tail_qc_pass", ""),
                "P5_qc_pass": v034_qc.get("P5_qc_pass", ""),
                "P6_qc_pass": v034_qc.get("P6_qc_pass", ""),
                "p7v_valid": p7v_info.get("p7v_valid", ""),
                "body_depth_qc_pass": v034_qc.get("body_depth_qc_pass", ""),
                "peduncle_depth_qc_pass": v034_qc.get("peduncle_depth_qc_pass", ""),
                "axis_qc_pass": v034_qc.get("axis_qc_pass", ""),
                "hybrid_qc_pass": hybrid_qc.get("hybrid_qc_pass", ""),
                "hybrid_review_reason": review_reason,
            }
        )
        write_json(
            OUTPUT_DIR / "debug_json" / f"{Path(image_name).stem}_debug.json",
            {
                "image_name": image_name,
                "specimen_id": record["specimen_id"],
                "split": record["split"],
                "manual_corrected_keypoints": manual_points,
                "heatmap_keypoints": heatmap_points,
                "heatmap_confidences": heatmap_conf,
                "heatmap_debug_info": heatmap_result.get("heatmap_debug_info", {}),
                "v034_keypoints": v034_points,
                "v034_model_keypoints_raw": v034_model_raw,
                "mask_suggestions": mask_suggestions,
                "geometric_suggestions": geometric_suggestions,
                "local_structure_suggestions": local_suggestions,
                "hybrid_keypoints": hybrid_points,
                "point_sources": point_sources,
                "hybrid_qc_results": hybrid_qc,
                "errors_by_keypoint": errors_by_keypoint,
                "derived_points": {"hybrid": {"P7V_caudal_fin_posterior_endpoint": p7v_info.get("point")}},
                "preannotation_metadata": {"v034": v034_metadata, "v04_hybrid": hybrid_qc},
            },
        )
        save_hybrid_visual(
            image,
            OUTPUT_DIR / "visual_comparisons" / f"{Path(image_name).stem}_compare.png",
            manual_points=manual_points,
            heatmap_points=heatmap_points,
            v034_points=v034_points,
            suggestions=suggestions,
            hybrid_points=hybrid_points,
            p7v_info=p7v_info,
            image_summary=image_row,
            point_sources=point_sources,
        )

    summary_df = pd.DataFrame(summary_rows)
    by_point = summarize_by_point(summary_df)
    by_image = pd.DataFrame(image_rows).sort_values("median_error_mm", ascending=False)
    qc_df = pd.DataFrame(qc_rows)
    source_df = source_summary(summary_df)
    hybrid_compare = hybrid_source_comparison(source_compare_rows)

    summary_df.to_csv(OUTPUT_DIR / "preannotation_vs_manual_summary.csv", index=False, encoding="utf-8-sig")
    by_point.to_csv(OUTPUT_DIR / "keypoint_error_by_point.csv", index=False, encoding="utf-8-sig")
    by_image.to_csv(OUTPUT_DIR / "keypoint_error_by_image.csv", index=False, encoding="utf-8-sig")
    qc_df.to_csv(OUTPUT_DIR / "qc_summary.csv", index=False, encoding="utf-8-sig")
    source_df.to_csv(OUTPUT_DIR / "point_source_summary.csv", index=False, encoding="utf-8-sig")
    hybrid_compare.to_csv(OUTPUT_DIR / "hybrid_source_comparison.csv", index=False, encoding="utf-8-sig")

    v034_median, v034_mean, v034_by_key = _base_summary(V034_DIR / "preannotation_vs_manual_summary.csv")
    heat_median, heat_mean, heat_by_key = _heatmap_best_summary()
    hybrid_median = float(summary_df["error_mm"].median())
    hybrid_mean = float(summary_df["error_mm"].mean())
    p7v_valid_rate = f"{int(qc_df['p7v_valid'].astype(str).str.lower().eq('true').sum())}/{len(qc_df)}"
    large_error_count = int((summary_df["error_mm"] > 10).sum())
    hybrid_by_key = summary_df.groupby("keypoint_name")["error_mm"].median().to_dict()
    axis_median = float(summary_df[summary_df["keypoint_name"].str.startswith("C")]["error_mm"].median())

    def key_line(key: str) -> str:
        return (
            f"{key}: v0.3.4={v034_by_key.get(key, np.nan):.3f}, "
            f"heatmap={heat_by_key.get(key, np.nan):.3f}, "
            f"hybrid={hybrid_by_key.get(key, np.nan):.3f} mm"
        )

    recommend_lines = []
    for key in KEYPOINT_KEYS:
        row = hybrid_compare[hybrid_compare["keypoint_name"] == key]
        recommend = row.iloc[0]["recommend_default_source"] if not row.empty else ""
        recommend_lines.append(f"- {key}: {recommend}")

    readme = f"""# v0.4 Hybrid Heatmap-Geometry Compare 30

This benchmark does not train a model and does not write hybrid points back to corrected labels.

Sample size: {len(selected)}

YOLO/v0.3.4 model: `{v034_model}`

Heatmap model: `{heatmap_model}`

## Overall Comparison

| model | overall_median_error_mm | overall_mean_error_mm |
|---|---:|---:|
| v0.3.4 enhanced baseline | {v034_median:.3f} | {v034_mean:.3f} |
| heatmap-U-Net best | {heat_median:.3f} | {heat_mean:.3f} |
| v0.4 hybrid | {hybrid_median:.3f} | {hybrid_mean:.3f} |

P7V valid rate: {p7v_valid_rate}

Large error count (>10 mm): {large_error_count}

## Keypoint Comparison

{key_line('P2_eye_front')}

{key_line('P3_operculum_posterior')}

{key_line('P4_peduncle_start_midpoint')}

{key_line('P5_caudal_base_midpoint')}

{key_line('P6_caudal_fork_midpoint')}

{key_line('P7U_caudal_fin_upper_tip')}

{key_line('P7L_caudal_fin_lower_tip')}

{key_line('P8_body_depth_dorsal')}

{key_line('P9_body_depth_ventral')}

{key_line('P10_peduncle_depth_dorsal')}

{key_line('P11_peduncle_depth_ventral')}

C1-C4 hybrid median error: {axis_median:.3f} mm

## Recommended Default Source By Keypoint

{chr(10).join(recommend_lines)}

See `hybrid_source_comparison.csv` for source-by-source medians.
"""
    (OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    print("Finished v0.4 hybrid heatmap-geometry evaluation.")
    print("Overall median error:")
    print(f"v0.3.4 baseline: {v034_median:.3f} mm")
    print(f"heatmap-U-Net best: {heat_median:.3f} mm")
    print(f"v0.4 hybrid: {hybrid_median:.3f} mm")
    print("Keypoint comparison:")
    for key in (
        "P2_eye_front",
        "P3_operculum_posterior",
        "P4_peduncle_start_midpoint",
        "P5_caudal_base_midpoint",
        "P6_caudal_fork_midpoint",
        "P7U_caudal_fin_upper_tip",
        "P7L_caudal_fin_lower_tip",
        "P8_body_depth_dorsal",
        "P9_body_depth_ventral",
        "P10_peduncle_depth_dorsal",
        "P11_peduncle_depth_ventral",
    ):
        print(key_line(key))
    print(f"C1-C4: hybrid median {axis_median:.3f} mm")
    print("Recommended default source by keypoint:")
    for line in recommend_lines:
        print(line)
    print(f"Results saved to: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
