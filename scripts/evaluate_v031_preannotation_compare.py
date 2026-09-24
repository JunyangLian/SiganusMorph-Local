"""Compare v0.3.1 enhanced preannotation against corrected manual labels.

This script does not write back labels or train models. It samples existing
corrected real annotations, runs the current enhanced preannotation pipeline,
and writes error reports plus visual comparisons.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph.config import KEYPOINT_DEFS
from siganusmorph.image_utils import ensure_rgb, load_image_file
from siganusmorph.measurements import calculate_measurements
from siganusmorph.preannotation import default_preannotation_model, preannotate_warped_image


CORRECTED_DIR = PROJECT_ROOT / "results" / "real_annotation_5_15_v0.3_corrected" / "keypoints"
WARPED_DIR = PROJECT_ROOT / "data" / "real_images_warped"
OUTPUT_DIR = Path(os.environ.get("SIGANUS_COMPARE_OUTPUT_DIR", PROJECT_ROOT / "results" / "model_eval" / "v0.3.1_preannotation_compare_30"))
COMPARE_LABEL = os.environ.get("SIGANUS_COMPARE_LABEL", "v0.3.1")
REUSE_SAMPLE_MANIFEST = os.environ.get("SIGANUS_COMPARE_SAMPLE_MANIFEST", "")
SPLIT_MANIFEST = PROJECT_ROOT / "datasets" / "siganusmorph_real_5_15_yolopose_maskcrop_v0.3_corrected" / "split_manifest.csv"
RANDOM_SEED = 20260515
TARGET_SAMPLE_SIZE = 30
DESIRED_SPLITS = {"train": 15, "val": 7, "test": 8}
DEFAULT_MM_PER_PIXEL = 0.1

KEYPOINT_KEYS = tuple(f"{definition.code}_{definition.name}" for definition in KEYPOINT_DEFS)
KEY_TO_SHORT = {f"{definition.code}_{definition.name}": definition.name for definition in KEYPOINT_DEFS}
AXIS_KEYS = (
    "C1_head_axis_point",
    "C2_trunk_axis_point",
    "C3_posterior_trunk_axis_point",
    "C4_peduncle_axis_point",
)
LOCAL_SUGGESTION_KEYS = {
    "P2_eye_front": ("P2_eye_suggestion", "eye_detection", "P2_eye_quality"),
    "P3_operculum_posterior": ("P3_edge_suggestion", "operculum_edge_detection", "P3_edge_quality"),
    "P5_caudal_base_midpoint": ("P5_transition_suggestion", "caudal_base_transition", "P5_transition_quality"),
    "P6_caudal_fork_midpoint": ("P6_gap_suggestion", "tail_geometry_qc", "P6_gap_quality"),
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def xy(value: Any) -> tuple[float, float] | None:
    if isinstance(value, Mapping):
        if "x" in value and "y" in value:
            return float(value["x"]), float(value["y"])
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return None


def distance_px(a: Any, b: Any) -> float | None:
    pa = xy(a)
    pb = xy(b)
    if pa is None or pb is None:
        return None
    return math.hypot(pa[0] - pb[0], pa[1] - pb[1])


def full_to_short(points: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for full_key, short_name in KEY_TO_SHORT.items():
        point = xy(points.get(full_key))
        if point is not None:
            out[short_name] = {"x": point[0], "y": point[1]}
    return out


def base_from_image_name(image_name: str) -> str:
    return Path(image_name).stem.removesuffix("_warped")


def load_split_map() -> dict[str, dict[str, str]]:
    if not SPLIT_MANIFEST.exists():
        return {}
    df = pd.read_csv(SPLIT_MANIFEST)
    split_map: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        image_name = str(row.get("image_name", ""))
        base = base_from_image_name(image_name)
        split_map[base] = {
            "split": str(row.get("split", "")),
            "specimen_id": str(row.get("specimen_id", "")),
        }
    return split_map


def collect_available_records() -> list[dict[str, Any]]:
    split_map = load_split_map()
    records: list[dict[str, Any]] = []
    for json_path in sorted(CORRECTED_DIR.glob("*_keypoints.json")):
        payload = read_json(json_path)
        manual = payload.get("corrected_keypoints") or payload.get("keypoints") or {}
        if not isinstance(manual, Mapping):
            continue
        if any(key not in manual for key in KEYPOINT_KEYS):
            continue
        image_name = str(payload.get("image_name") or f"{json_path.stem.removesuffix('_keypoints')}_warped.png")
        base = base_from_image_name(image_name)
        warped_path = WARPED_DIR / f"{base}_warped.png"
        if not warped_path.exists():
            continue
        split_info = split_map.get(base, {})
        records.append(
            {
                "image_name": warped_path.name,
                "base": base,
                "specimen_id": str(payload.get("specimen_id") or split_info.get("specimen_id") or ""),
                "split": str(split_info.get("split") or ""),
                "warped_image_path": warped_path,
                "corrected_json_path": json_path,
                "payload": payload,
            }
        )
    return records


def records_from_sample_manifest(records: list[dict[str, Any]], manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"sample manifest not found: {manifest_path}")
    by_base = {str(record["base"]): record for record in records}
    df = pd.read_csv(manifest_path)
    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    for _, row in df.iterrows():
        image_name = str(row.get("image_name", ""))
        base = base_from_image_name(image_name)
        record = by_base.get(base)
        if record is None:
            missing.append(image_name)
            continue
        record = dict(record)
        if str(row.get("split", "")):
            record["split"] = str(row.get("split", ""))
        if str(row.get("specimen_id", "")):
            record["specimen_id"] = str(row.get("specimen_id", ""))
        selected.append(record)
    if missing:
        raise RuntimeError(f"{len(missing)} sample images were not available: {missing[:5]}")
    return selected


def sample_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = random.Random(RANDOM_SEED)
    if len(records) <= TARGET_SAMPLE_SIZE:
        return records

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_split[str(record.get("split", ""))].append(record)
    for pool in by_split.values():
        rng.shuffle(pool)

    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    specimen_counts: Counter[str] = Counter()

    def maybe_add(record: dict[str, Any]) -> bool:
        key = str(record["base"])
        specimen = str(record.get("specimen_id", ""))
        if key in selected_keys:
            return False
        if specimen and specimen_counts[specimen] >= 2:
            return False
        selected.append(record)
        selected_keys.add(key)
        if specimen:
            specimen_counts[specimen] += 1
        return True

    for split, desired in DESIRED_SPLITS.items():
        pool = by_split.get(split, [])
        # First pass: prioritize new specimens.
        for record in pool:
            if len([r for r in selected if r.get("split") == split]) >= desired:
                break
            specimen = str(record.get("specimen_id", ""))
            if specimen and specimen_counts[specimen] == 0:
                maybe_add(record)
        for record in pool:
            if len([r for r in selected if r.get("split") == split]) >= desired:
                break
            maybe_add(record)

    remaining = records[:]
    rng.shuffle(remaining)
    for record in remaining:
        if len(selected) >= TARGET_SAMPLE_SIZE:
            break
        maybe_add(record)
    # If max-2-per-specimen blocks reaching 30, relax it only for the final fill.
    for record in remaining:
        if len(selected) >= TARGET_SAMPLE_SIZE:
            break
        key = str(record["base"])
        if key in selected_keys:
            continue
        selected.append(record)
        selected_keys.add(key)
    return selected[:TARGET_SAMPLE_SIZE]


def point_source_for(key: str, metadata: Mapping[str, Any]) -> str:
    point_sources = metadata.get("point_sources", {})
    source = ""
    if isinstance(point_sources, Mapping):
        source = str(point_sources.get(key, "") or "")
    if key == "P1_snout_tip":
        source = str(metadata.get("p1_source") or source)
    if not source or source == "model_or_manual":
        return "model"
    if source == "mask_corrected":
        return "mask_left_boundary"
    return source


def safe_measurements(points: Mapping[str, Any], mm_per_pixel: float) -> dict[str, Any]:
    try:
        return calculate_measurements(full_to_short(points), mm_per_pixel, "auto")
    except Exception as exc:
        return {"measurement_error": str(exc)}


def font(size: int = 22) -> ImageFont.ImageFont:
    for candidate in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_label(draw: ImageDraw.ImageDraw, xy0: tuple[float, float], text: str, fill: tuple[int, int, int]) -> None:
    x0, y0 = xy0
    fnt = font(20)
    for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        draw.text((x0 + ox, y0 + oy), text, font=fnt, fill=(0, 0, 0))
    draw.text((x0, y0), text, font=fnt, fill=fill)


def draw_star(draw: ImageDraw.ImageDraw, xy0: tuple[float, float], radius: float, fill: tuple[int, int, int]) -> None:
    x0, y0 = xy0
    pts = []
    for i in range(10):
        angle = -math.pi / 2 + i * math.pi / 5
        r = radius if i % 2 == 0 else radius * 0.45
        pts.append((x0 + math.cos(angle) * r, y0 + math.sin(angle) * r))
    draw.polygon(pts, fill=fill, outline=(70, 20, 110))


def scale_point(point: Any, scale: float) -> tuple[float, float] | None:
    p = xy(point)
    if p is None:
        return None
    return p[0] * scale, p[1] * scale


def save_visual_comparison(
    image: np.ndarray,
    output_path: Path,
    manual_points: Mapping[str, Any],
    auto_points: Mapping[str, Any],
    suggestions: Mapping[str, Any],
    qc_results: Mapping[str, Any],
    image_summary: Mapping[str, Any],
) -> None:
    pil = Image.fromarray(ensure_rgb(image))
    max_width = 2200
    scale = min(1.0, max_width / max(1, pil.width))
    if scale < 1:
        pil = pil.resize((int(round(pil.width * scale)), int(round(pil.height * scale))), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(pil)
    radius = max(5, int(round(min(pil.size) / 260)))
    large_keys = set(str(image_summary.get("large_error_keypoints", "")).split(";"))

    for key, point in suggestions.items():
        p = scale_point(point, scale)
        if p is None:
            continue
        draw.ellipse((p[0] - radius, p[1] - radius, p[0] + radius, p[1] + radius), fill=(0, 220, 220), outline=(0, 70, 90), width=2)

    for key in KEYPOINT_KEYS:
        manual = scale_point(manual_points.get(key), scale)
        auto = scale_point(auto_points.get(key), scale)
        code = key.split("_", 1)[0]
        if manual is not None and auto is not None:
            draw.line((manual[0], manual[1], auto[0], auto[1]), fill=(255, 255, 255), width=1)
        if manual is not None:
            draw.ellipse((manual[0] - radius, manual[1] - radius, manual[0] + radius, manual[1] + radius), fill=(60, 220, 90), outline=(10, 60, 20), width=2)
        if auto is not None:
            draw.ellipse((auto[0] - radius, auto[1] - radius, auto[0] + radius, auto[1] + radius), outline=(255, 100, 20), width=4)
            draw_label(draw, (auto[0] + radius + 2, auto[1] - radius - 2), code, (255, 160, 40))
        if key in large_keys:
            anchor = auto or manual
            if anchor is not None:
                draw.ellipse(
                    (anchor[0] - radius - 10, anchor[1] - radius - 10, anchor[0] + radius + 10, anchor[1] + radius + 10),
                    outline=(255, 230, 0),
                    width=5,
                )

    p7v = qc_results.get("p7v_suggestion")
    if isinstance(p7v, (list, tuple)) and len(p7v) >= 2:
        p = scale_point(p7v, scale)
        if p is not None:
            if qc_results.get("p7v_valid") is False:
                size = radius + 12
                draw.line((p[0] - size, p[1] - size, p[0] + size, p[1] + size), fill=(255, 0, 0), width=5)
                draw.line((p[0] - size, p[1] + size, p[0] + size, p[1] - size), fill=(255, 0, 0), width=5)
                draw_label(draw, (p[0] + size + 3, p[1] - size), "P7V invalid", (255, 0, 0))
            else:
                draw_star(draw, p, radius + 9, (210, 90, 255))
                draw_label(draw, (p[0] + radius + 10, p[1] - radius), "P7V", (210, 90, 255))

    lines = [
        f"{image_summary.get('image_name', '')} | {image_summary.get('specimen_id', '')} | {image_summary.get('split', '')}",
        f"mean {float(image_summary.get('mean_error_mm', 0) or 0):.2f} mm | max {float(image_summary.get('max_error_mm', 0) or 0):.2f} mm",
        f"large: {image_summary.get('large_error_keypoints', '') or 'none'}",
        f"QC: {str(image_summary.get('review_reason', ''))[:130]}",
        "green=manual | orange=auto | cyan=suggestion | yellow=large error",
    ]
    fnt = font(22)
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
        sources = group["auto_point_source"].fillna("").astype(str)
        counts = Counter(sources)
        most_common = counts.most_common(1)[0][0] if counts else ""
        notes = ""
        if key in {"P5_caudal_base_midpoint", "P6_caudal_fork_midpoint", "P7U_caudal_fin_upper_tip", "P7L_caudal_fin_lower_tip", "P10_peduncle_depth_dorsal", "P11_peduncle_depth_ventral"}:
            notes = "priority_tail_or_depth_point"
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
                "most_common_point_source": most_common,
                "mean_model_raw_error_mm": group["model_raw_error_mm"].mean(),
                "median_model_raw_error_mm": group["model_raw_error_mm"].median(),
                "notes": notes,
            }
        )
    return pd.DataFrame(rows).sort_values("median_error_mm", ascending=False)


def summarize_point_sources(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_by_key = summary_df.groupby("keypoint_name").size().to_dict()
    for (key, source), group in summary_df.groupby(["keypoint_name", "auto_point_source"]):
        total = total_by_key.get(key, len(group))
        rows.append(
            {
                "keypoint_name": key,
                "source_type": source,
                "count": len(group),
                "percentage": round(100.0 * len(group) / max(1, total), 2),
                "mean_error_mm": group["error_mm"].mean(),
                "median_error_mm": group["error_mm"].median(),
            }
        )
    return pd.DataFrame(rows).sort_values(["keypoint_name", "count"], ascending=[True, False])


def summarize_axis_source_comparison(axis_df: pd.DataFrame) -> pd.DataFrame:
    if axis_df.empty:
        return pd.DataFrame(
            columns=[
                "keypoint_name",
                "num_images",
                "model_median_error_mm",
                "mask_suggestion_median_error_mm",
                "model_mean_error_mm",
                "mask_suggestion_mean_error_mm",
                "mask_better_count",
                "model_better_count",
                "recommend_use_mask_axis_points",
            ]
        )
    rows = []
    for key, group in axis_df.groupby("keypoint_name"):
        valid = group.dropna(subset=["model_error_mm", "mask_suggestion_error_mm"])
        mask_better = int((valid["mask_suggestion_error_mm"] < valid["model_error_mm"]).sum())
        model_better = int((valid["model_error_mm"] <= valid["mask_suggestion_error_mm"]).sum())
        mask_median = group["mask_suggestion_error_mm"].median()
        model_median = group["model_error_mm"].median()
        rows.append(
            {
                "keypoint_name": key,
                "num_images": int(len(group)),
                "model_median_error_mm": model_median,
                "mask_suggestion_median_error_mm": mask_median,
                "model_mean_error_mm": group["model_error_mm"].mean(),
                "mask_suggestion_mean_error_mm": group["mask_suggestion_error_mm"].mean(),
                "mask_better_count": mask_better,
                "model_better_count": model_better,
                "recommend_use_mask_axis_points": bool(
                    pd.notna(mask_median)
                    and pd.notna(model_median)
                    and mask_median < model_median
                    and mask_better > model_better
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("keypoint_name")


def summarize_local_structure_comparison(local_df: pd.DataFrame) -> pd.DataFrame:
    if local_df.empty:
        return pd.DataFrame(
            columns=[
                "keypoint_name",
                "num_images",
                "model_median_error_mm",
                "suggestion_median_error_mm",
                "model_mean_error_mm",
                "suggestion_mean_error_mm",
                "suggestion_better_count",
                "model_better_count",
                "recommend_use_suggestion_as_default",
                "suggestion_quality_good_count",
                "suggestion_quality_failed_count",
            ]
        )
    rows = []
    for key, group in local_df.groupby("keypoint_name"):
        valid = group.dropna(subset=["model_error_mm", "suggestion_error_mm"])
        suggestion_better = int((valid["suggestion_error_mm"] < valid["model_error_mm"]).sum())
        model_better = int((valid["model_error_mm"] <= valid["suggestion_error_mm"]).sum())
        model_median = group["model_error_mm"].median()
        suggestion_median = group["suggestion_error_mm"].median()
        quality = group["suggestion_quality"].fillna("").astype(str)
        good_count = int(quality.isin(["good", "ok"]).sum())
        failed_count = int(quality.str.startswith("failed").sum() + (quality == "").sum())
        recommend = bool(
            pd.notna(model_median)
            and pd.notna(suggestion_median)
            and suggestion_median + 0.5 < model_median
            and suggestion_better > model_better
            and good_count >= max(5, int(0.5 * len(group)))
        )
        rows.append(
            {
                "keypoint_name": key,
                "num_images": int(len(group)),
                "model_median_error_mm": model_median,
                "suggestion_median_error_mm": suggestion_median,
                "model_mean_error_mm": group["model_error_mm"].mean(),
                "suggestion_mean_error_mm": group["suggestion_error_mm"].mean(),
                "suggestion_better_count": suggestion_better,
                "model_better_count": model_better,
                "recommend_use_suggestion_as_default": recommend,
                "suggestion_quality_good_count": good_count,
                "suggestion_quality_failed_count": failed_count,
            }
        )
    return pd.DataFrame(rows).sort_values("keypoint_name")


def main() -> None:
    output = OUTPUT_DIR
    (output / "visual_comparisons").mkdir(parents=True, exist_ok=True)
    (output / "debug_json").mkdir(parents=True, exist_ok=True)

    records = collect_available_records()
    if not records:
        raise SystemExit("No corrected annotations with matching warped images were found.")
    if REUSE_SAMPLE_MANIFEST:
        selected = records_from_sample_manifest(records, Path(REUSE_SAMPLE_MANIFEST))
    else:
        selected = sample_records(records)
    model_path = default_preannotation_model(PROJECT_ROOT)

    sample_manifest = pd.DataFrame(
        [
            {
                "image_name": record["image_name"],
                "specimen_id": record["specimen_id"],
                "split": record["split"],
                "warped_image_path": str(record["warped_image_path"]),
                "corrected_json_path": str(record["corrected_json_path"]),
            }
            for record in selected
        ]
    )
    sample_manifest.to_csv(output / "sample_manifest.csv", index=False, encoding="utf-8-sig")

    all_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    axis_compare_rows: list[dict[str, Any]] = []
    local_compare_rows: list[dict[str, Any]] = []

    for idx, record in enumerate(selected, start=1):
        image_name = str(record["image_name"])
        print(f"[{idx}/{len(selected)}] evaluating {image_name}")
        payload = record["payload"]
        manual_points = payload.get("corrected_keypoints") or payload.get("keypoints") or {}
        mm_per_pixel = float(payload.get("mm_per_pixel") or DEFAULT_MM_PER_PIXEL)
        image = load_image_file(record["warped_image_path"])
        result = preannotate_warped_image(image, image_name, PROJECT_ROOT, model_path)
        metadata = result.get("metadata", {})
        auto_points = metadata.get("corrected_keypoints") or result.get("keypoints") or {}
        model_raw = metadata.get("model_keypoints_raw", {})
        mask_suggestions = metadata.get("mask_suggestions", {}) if isinstance(metadata.get("mask_suggestions", {}), Mapping) else {}
        geometric_suggestions = metadata.get("geometric_suggestions", {}) if isinstance(metadata.get("geometric_suggestions", {}), Mapping) else {}
        centerline_suggestions = metadata.get("centerline_suggestions", {}) if isinstance(metadata.get("centerline_suggestions", {}), Mapping) else {}
        local_structure_suggestions = metadata.get("local_structure_suggestions", {}) if isinstance(metadata.get("local_structure_suggestions", {}), Mapping) else {}
        suggestions = dict(mask_suggestions)
        suggestions.update(geometric_suggestions)
        suggestions.update(local_structure_suggestions)
        qc_results = metadata.get("qc_results", {}) if isinstance(metadata.get("qc_results", {}), Mapping) else {}
        point_sources = metadata.get("point_sources", {}) if isinstance(metadata.get("point_sources", {}), Mapping) else {}
        model_conf = result.get("confidences", {}) if isinstance(result.get("confidences", {}), Mapping) else {}
        flagged = qc_results.get("flagged_keypoints", {}) if isinstance(qc_results.get("flagged_keypoints", {}), Mapping) else {}
        review_reason = str(metadata.get("review_reason", "") or qc_results.get("review_reason", ""))

        image_errors: list[float] = []
        large_keys: list[str] = []
        errors_by_keypoint: dict[str, Any] = {}
        for key in KEYPOINT_KEYS:
            err_px = distance_px(auto_points.get(key), manual_points.get(key))
            raw_err_px = distance_px(model_raw.get(key), manual_points.get(key))
            err_mm = err_px * mm_per_pixel if err_px is not None else np.nan
            raw_err_mm = raw_err_px * mm_per_pixel if raw_err_px is not None else np.nan
            is_large = bool(pd.notna(err_mm) and err_mm > 10.0)
            if pd.notna(err_mm):
                image_errors.append(float(err_mm))
            if is_large:
                large_keys.append(key)
            manual = xy(manual_points.get(key))
            auto = xy(auto_points.get(key))
            raw = xy(model_raw.get(key))
            source = point_source_for(key, metadata)
            row = {
                "image_name": image_name,
                "specimen_id": record["specimen_id"],
                "split": record["split"],
                "keypoint_name": key,
                "manual_x": manual[0] if manual else np.nan,
                "manual_y": manual[1] if manual else np.nan,
                "auto_x": auto[0] if auto else np.nan,
                "auto_y": auto[1] if auto else np.nan,
                "model_raw_x": raw[0] if raw else np.nan,
                "model_raw_y": raw[1] if raw else np.nan,
                "error_px": err_px if err_px is not None else np.nan,
                "error_mm": err_mm,
                "model_raw_error_px": raw_err_px if raw_err_px is not None else np.nan,
                "model_raw_error_mm": raw_err_mm,
                "is_large_error": is_large,
                "auto_point_source": source,
                "model_confidence": model_conf.get(key, np.nan),
                "qc_flag": flagged.get(key, ""),
                "review_reason": review_reason,
            }
            all_rows.append(row)
            errors_by_keypoint[key] = row

            if key in AXIS_KEYS:
                mask_axis_point = centerline_suggestions.get(key) or mask_suggestions.get(key)
                mask_axis_err_px = distance_px(mask_axis_point, manual_points.get(key))
                axis_compare_rows.append(
                    {
                        "image_name": image_name,
                        "specimen_id": record["specimen_id"],
                        "split": record["split"],
                        "keypoint_name": key,
                        "model_error_px": raw_err_px if raw_err_px is not None else np.nan,
                        "model_error_mm": raw_err_mm,
                        "mask_suggestion_error_px": mask_axis_err_px if mask_axis_err_px is not None else np.nan,
                        "mask_suggestion_error_mm": mask_axis_err_px * mm_per_pixel if mask_axis_err_px is not None else np.nan,
                        "mask_suggestion_x": xy(mask_axis_point)[0] if xy(mask_axis_point) else np.nan,
                        "mask_suggestion_y": xy(mask_axis_point)[1] if xy(mask_axis_point) else np.nan,
                    }
                )

            if key in LOCAL_SUGGESTION_KEYS:
                suggestion_field, container_key, quality_key = LOCAL_SUGGESTION_KEYS[key]
                container = metadata.get(container_key, {}) if isinstance(metadata.get(container_key, {}), Mapping) else {}
                suggestion_point = local_structure_suggestions.get(key)
                if suggestion_point is None and isinstance(container, Mapping):
                    suggestion_point = container.get(suggestion_field)
                if suggestion_point is None and key == "P6_caudal_fork_midpoint":
                    suggestion_point = mask_suggestions.get(key) or auto_points.get(key)
                suggestion_err_px = distance_px(suggestion_point, manual_points.get(key))
                local_compare_rows.append(
                    {
                        "image_name": image_name,
                        "specimen_id": record["specimen_id"],
                        "split": record["split"],
                        "keypoint_name": key,
                        "model_error_px": raw_err_px if raw_err_px is not None else np.nan,
                        "model_error_mm": raw_err_mm,
                        "suggestion_error_px": suggestion_err_px if suggestion_err_px is not None else np.nan,
                        "suggestion_error_mm": suggestion_err_px * mm_per_pixel if suggestion_err_px is not None else np.nan,
                        "suggestion_x": xy(suggestion_point)[0] if xy(suggestion_point) else np.nan,
                        "suggestion_y": xy(suggestion_point)[1] if xy(suggestion_point) else np.nan,
                        "suggestion_quality": str(container.get(quality_key, "")) if isinstance(container, Mapping) else "",
                    }
                )

        auto_measurements = safe_measurements(auto_points, mm_per_pixel)
        manual_measurements = safe_measurements(manual_points, mm_per_pixel)
        manual_p7v = None
        if isinstance(payload.get("derived_points"), Mapping):
            manual_p7v = payload["derived_points"].get("P7V_caudal_fin_posterior_endpoint")
        auto_p7v = None
        if isinstance(auto_measurements.get("derived_points"), Mapping):
            auto_p7v = auto_measurements["derived_points"].get("P7V_caudal_fin_posterior_endpoint")
        p7v_error_px = distance_px(auto_p7v, manual_p7v) if manual_p7v is not None else np.nan

        image_row = {
            "image_name": image_name,
            "specimen_id": record["specimen_id"],
            "split": record["split"],
            "mean_error_mm": float(np.mean(image_errors)) if image_errors else np.nan,
            "median_error_mm": float(np.median(image_errors)) if image_errors else np.nan,
            "max_error_mm": float(np.max(image_errors)) if image_errors else np.nan,
            "num_large_error_points": len(large_keys),
            "large_error_keypoints": ";".join(large_keys),
            "needs_review": bool(metadata.get("needs_review", False)),
            "review_reason": review_reason,
            "p7v_valid": qc_results.get("p7v_valid", ""),
            "tail_qc_pass": qc_results.get("tail_qc_pass", ""),
            "body_depth_qc_pass": qc_results.get("body_depth_qc_pass", ""),
            "peduncle_depth_qc_pass": qc_results.get("peduncle_depth_qc_pass", ""),
            "axis_qc_pass": qc_results.get("axis_qc_pass", ""),
            "P7V_error_px": p7v_error_px,
            "P7V_error_mm": p7v_error_px * mm_per_pixel if pd.notna(p7v_error_px) else np.nan,
            "auto_TL_final_mm": auto_measurements.get("TL_final_mm", np.nan),
            "manual_TL_final_mm": manual_measurements.get("TL_final_mm", np.nan),
            "auto_SL_final_mm": auto_measurements.get("SL_final_mm", np.nan),
            "manual_SL_final_mm": manual_measurements.get("SL_final_mm", np.nan),
        }
        image_rows.append(image_row)

        qc_rows.append(
            {
                "image_name": image_name,
                "specimen_id": record["specimen_id"],
                "split": record["split"],
                "segmentation_success": metadata.get("segmentation_success", ""),
                "segmentation_quality": metadata.get("segmentation_quality", ""),
                "tail_qc_pass": qc_results.get("tail_qc_pass", ""),
                "P5_qc_pass": qc_results.get("P5_qc_pass", ""),
                "P6_qc_pass": qc_results.get("P6_qc_pass", ""),
                "p7v_valid": qc_results.get("p7v_valid", ""),
                "body_depth_qc_pass": qc_results.get("body_depth_qc_pass", ""),
                "peduncle_depth_qc_pass": qc_results.get("peduncle_depth_qc_pass", ""),
                "axis_qc_pass": qc_results.get("axis_qc_pass", ""),
                "needs_review": metadata.get("needs_review", ""),
                "review_reason": review_reason,
            }
        )

        debug_payload = {
            "image_name": image_name,
            "specimen_id": record["specimen_id"],
            "split": record["split"],
            "manual_corrected_keypoints": manual_points,
            "auto_preannotation_keypoints": auto_points,
            "model_keypoints_raw": model_raw,
            "mask_suggestions": mask_suggestions,
            "geometric_suggestions": geometric_suggestions,
            "point_sources": point_sources,
            "qc_results": qc_results,
            "errors_by_keypoint": errors_by_keypoint,
            "derived_points": {
                "auto": auto_measurements.get("derived_points", {}),
                "manual": payload.get("derived_points", {}),
            },
            "p7v_valid": qc_results.get("p7v_valid", ""),
            "review_reason": review_reason,
            "preannotation_metadata": metadata,
            "eye_detection": metadata.get("eye_detection", {}),
            "operculum_edge_detection": metadata.get("operculum_edge_detection", {}),
            "caudal_base_transition": metadata.get("caudal_base_transition", {}),
            "P6_gap_detection": metadata.get("tail_geometry_qc", {}),
            "local_structure_qc": metadata.get("local_structure_qc", {}),
        }
        write_json(output / "debug_json" / f"{Path(image_name).stem}_debug.json", debug_payload)
        save_visual_comparison(
            image,
            output / "visual_comparisons" / f"{Path(image_name).stem}_compare.png",
            manual_points,
            auto_points,
            suggestions,
            qc_results,
            image_row,
        )

    summary_df = pd.DataFrame(all_rows)
    by_point = summarize_by_point(summary_df)
    by_image = pd.DataFrame(image_rows).sort_values("median_error_mm", ascending=False)
    qc_df = pd.DataFrame(qc_rows)
    source_df = summarize_point_sources(summary_df)
    axis_compare_df = summarize_axis_source_comparison(pd.DataFrame(axis_compare_rows))
    local_compare_df = summarize_local_structure_comparison(pd.DataFrame(local_compare_rows))

    summary_df.to_csv(output / "preannotation_vs_manual_summary.csv", index=False, encoding="utf-8-sig")
    by_point.to_csv(output / "keypoint_error_by_point.csv", index=False, encoding="utf-8-sig")
    by_image.to_csv(output / "keypoint_error_by_image.csv", index=False, encoding="utf-8-sig")
    qc_df.to_csv(output / "qc_summary.csv", index=False, encoding="utf-8-sig")
    source_df.to_csv(output / "point_source_summary.csv", index=False, encoding="utf-8-sig")
    axis_compare_df.to_csv(output / "axis_point_source_comparison.csv", index=False, encoding="utf-8-sig")
    local_compare_df.to_csv(output / "local_structure_suggestion_comparison.csv", index=False, encoding="utf-8-sig")

    median_error = float(summary_df["error_mm"].median())
    mean_error = float(summary_df["error_mm"].mean())
    worst = by_point.head(6)["keypoint_name"].tolist()
    best = by_point.sort_values("median_error_mm", ascending=True).head(6)["keypoint_name"].tolist()

    p7_rows = summary_df[summary_df["keypoint_name"].isin([P for P in ("P7U_caudal_fin_upper_tip", "P7L_caudal_fin_lower_tip")])]
    p7_auto_median = float(p7_rows["error_mm"].median()) if not p7_rows.empty else np.nan
    p7_raw_median = float(p7_rows["model_raw_error_mm"].median()) if not p7_rows.empty else np.nan
    p7_improvement = p7_raw_median - p7_auto_median if pd.notna(p7_raw_median) and pd.notna(p7_auto_median) else np.nan
    mask_tail_counts = source_df[
        (source_df["keypoint_name"].isin(["P7U_caudal_fin_upper_tip", "P7L_caudal_fin_lower_tip"]))
        & (source_df["source_type"] == "mask_tail_rule")
    ]

    readme = f"""# {COMPARE_LABEL} Enhanced Preannotation Compare 30

Sample size: {len(selected)}

Model: `{model_path}`

Preannotation workflow: v0.3 YOLO-pose -> fish mask -> tail_geometry rules -> mask/geometric suggestions -> QC checks -> enhanced preannotation draft.

Overall mean keypoint error: {mean_error:.3f} mm

Overall median keypoint error: {median_error:.3f} mm

Best keypoints by median error: {", ".join(best)}

Worst keypoints by median error: {", ".join(worst)}

P7U/P7L raw model median error: {p7_raw_median:.3f} mm

P7U/P7L enhanced auto median error: {p7_auto_median:.3f} mm

P7U/P7L mask rule improvement: {p7_improvement:.3f} mm

Mask-tail-rule counts:

{mask_tail_counts.to_string(index=False) if not mask_tail_counts.empty else "No P7U/P7L mask_tail_rule rows found."}

C1-C4 model vs mask centerline suggestions:

{axis_compare_df.to_string(index=False) if not axis_compare_df.empty else "No axis centerline comparison rows found."}

Local-structure suggestion comparison:

{local_compare_df.to_string(index=False) if not local_compare_df.empty else "No local structure comparison rows found."}

Interpretation:

- Positive P7U/P7L mask rule improvement means the v0.3.1 tail rule is closer to manual labels than raw YOLO points.
- Large-error and QC columns should be used to decide which preannotations need manual confirmation first.
- This evaluation does not overwrite corrected labels and does not train a model.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    print(f"Evaluated {COMPARE_LABEL} enhanced preannotation on {len(selected)} images.")
    print(f"Median keypoint error: {median_error:.3f} mm.")
    print(f"Worst keypoints: {', '.join(worst)}")
    print(f"Best keypoints: {', '.join(best)}")
    print(f"P7U/P7L mask rule improvement: {p7_improvement:.3f} mm.")
    if not axis_compare_df.empty:
        print("C1-C4 model vs mask suggestion comparison saved to axis_point_source_comparison.csv")
    if not local_compare_df.empty:
        print("Local structure suggestion comparison saved to local_structure_suggestion_comparison.csv")
    print(f"Results saved to {output.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
