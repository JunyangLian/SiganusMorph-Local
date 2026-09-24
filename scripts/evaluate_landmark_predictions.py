"""Evaluate external landmark-model predictions against corrected keypoints."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
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


KEYPOINT_KEYS = tuple(f"{definition.code}_{definition.name}" for definition in KEYPOINT_DEFS)
DEFAULT_GT_DIR = PROJECT_ROOT / "results" / "real_annotation_5_15_v0.3_corrected" / "keypoints"
DEFAULT_SPLIT = PROJECT_ROOT / "datasets" / "siganusmorph_real_5_15_yolopose_maskcrop_v0.3_corrected" / "split_manifest.csv"
DEFAULT_CROP = PROJECT_ROOT / "datasets" / "siganusmorph_real_5_15_yolopose_maskcrop_v0.3_corrected" / "crop_metadata.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "model_eval" / "external_benchmark"
DEFAULT_MM_PER_PIXEL = 0.1


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def xy(value: Any) -> tuple[float, float] | None:
    if isinstance(value, Mapping) and "x" in value and "y" in value:
        return float(value["x"]), float(value["y"])
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return None


def base_from_image_name(image_name: str) -> str:
    stem = Path(image_name).stem
    changed = True
    while changed:
        changed = False
        for suffix in ("_512", "_warped", "_maskcrop"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                changed = True
    return stem


def load_ground_truth(gt_dir: Path) -> dict[str, dict[str, Any]]:
    gt: dict[str, dict[str, Any]] = {}
    for path in gt_dir.glob("*_keypoints.json"):
        payload = read_json(path)
        image_name = str(payload.get("image_name") or f"{path.stem.removesuffix('_keypoints')}.png")
        base = base_from_image_name(image_name)
        points = payload.get("corrected_keypoints") or payload.get("keypoints") or {}
        if not isinstance(points, Mapping):
            continue
        gt[base] = {
            "image_name": image_name,
            "specimen_id": str(payload.get("specimen_id", "")),
            "mm_per_pixel": float(payload.get("mm_per_pixel") or DEFAULT_MM_PER_PIXEL),
            "points": points,
        }
    return gt


def load_crop_metadata(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {base_from_image_name(str(row["image_name"])): row.to_dict() for _, row in df.iterrows()}


def load_split(path: Path) -> dict[str, dict[str, str]]:
    df = pd.read_csv(path)
    return {
        base_from_image_name(str(row["image_name"])): {
            "split": str(row.get("split", "")),
            "specimen_id": str(row.get("specimen_id", "")),
        }
        for _, row in df.iterrows()
    }


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = read_json(path)
        records = payload.get("predictions", payload if isinstance(payload, list) else [])
        out: dict[str, dict[str, Any]] = {}
        for record in records:
            image_name = str(record.get("image_name") or record.get("frame") or "")
            if not image_name:
                continue
            points = record.get("keypoints") or record.get("points") or {}
            confidences = record.get("confidences") or {}
            out[base_from_image_name(image_name)] = {
                "image_name": image_name,
                "points": points,
                "confidences": confidences,
            }
        return out
    return _load_predictions_csv(path)


def load_dlc_multilevel_csv(path: Path) -> dict[str, dict[str, Any]]:
    df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    out: dict[str, dict[str, Any]] = {}
    for index, row in df.iterrows():
        image_name = Path(str(index)).name
        points: dict[str, list[float]] = {}
        confidences: dict[str, float] = {}
        for key in KEYPOINT_KEYS:
            x_col = next((col for col in df.columns if len(col) >= 3 and col[1] == key and str(col[2]).lower() == "x"), None)
            y_col = next((col for col in df.columns if len(col) >= 3 and col[1] == key and str(col[2]).lower() == "y"), None)
            l_col = next((col for col in df.columns if len(col) >= 3 and col[1] == key and str(col[2]).lower() in {"likelihood", "confidence"}), None)
            if x_col is None or y_col is None or pd.isna(row[x_col]) or pd.isna(row[y_col]):
                continue
            points[key] = [float(row[x_col]), float(row[y_col])]
            if l_col is not None and pd.notna(row[l_col]):
                confidences[key] = float(row[l_col])
        out[base_from_image_name(image_name)] = {"image_name": image_name, "points": points, "confidences": confidences}
    return out


def _load_predictions_csv(path: Path) -> dict[str, dict[str, Any]]:
    df = pd.read_csv(path)
    if "image_name" not in df.columns:
        return load_dlc_multilevel_csv(path)
    out: dict[str, dict[str, Any]] = defaultdict(lambda: {"points": {}, "confidences": {}})
    if {"image_name", "keypoint_name", "x", "y"}.issubset(df.columns):
        for _, row in df.iterrows():
            base = base_from_image_name(str(row["image_name"]))
            out[base]["image_name"] = str(row["image_name"])
            out[base]["points"][str(row["keypoint_name"])] = [float(row["x"]), float(row["y"])]
            if "confidence" in row:
                out[base]["confidences"][str(row["keypoint_name"])] = row.get("confidence", np.nan)
    else:
        for _, row in df.iterrows():
            image_name = str(row.get("image_name", ""))
            base = base_from_image_name(image_name)
            out[base]["image_name"] = image_name
            for key in KEYPOINT_KEYS:
                x_col, y_col = f"{key}_x", f"{key}_y"
                if x_col in df.columns and y_col in df.columns and pd.notna(row.get(x_col)) and pd.notna(row.get(y_col)):
                    out[base]["points"][key] = [float(row[x_col]), float(row[y_col])]
    return dict(out)


def to_warped_points(points: Mapping[str, Any], base: str, coordinate_space: str, crop_meta: Mapping[str, Any]) -> dict[str, list[float]]:
    if coordinate_space == "warped":
        return {key: list(map(float, xy(value))) for key, value in points.items() if xy(value) is not None}
    if coordinate_space != "crop":
        raise ValueError("coordinate_space must be 'crop' or 'warped'")
    x0 = float(crop_meta.get("bbox_x1", 0.0))
    y0 = float(crop_meta.get("bbox_y1", 0.0))
    return {
        key: [float(pt[0]) + x0, float(pt[1]) + y0]
        for key, value in points.items()
        if (pt := xy(value)) is not None
    }


def distance(a: Any, b: Any) -> float | None:
    pa = xy(a)
    pb = xy(b)
    if pa is None or pb is None:
        return None
    return math.hypot(pa[0] - pb[0], pa[1] - pb[1])


def font(size: int = 20) -> ImageFont.ImageFont:
    for candidate in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def save_visual(image_path: Path, out_path: Path, gt_points: Mapping[str, Any], pred_points: Mapping[str, Any], title: str) -> None:
    if not image_path.exists():
        return
    image = Image.fromarray(ensure_rgb(load_image_file(image_path)))
    scale = min(1.0, 2200 / max(1, image.width))
    if scale < 1:
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    r = max(5, int(min(image.size) / 260))
    for key in KEYPOINT_KEYS:
        g = xy(gt_points.get(key))
        p = xy(pred_points.get(key))
        if g is not None:
            g2 = (g[0] * scale, g[1] * scale)
            draw.ellipse((g2[0] - r, g2[1] - r, g2[0] + r, g2[1] + r), fill=(60, 220, 90), outline=(0, 80, 0), width=2)
        if p is not None:
            p2 = (p[0] * scale, p[1] * scale)
            draw.ellipse((p2[0] - r, p2[1] - r, p2[0] + r, p2[1] + r), outline=(255, 120, 0), width=4)
            if g is not None:
                draw.line((g[0] * scale, g[1] * scale, p2[0], p2[1]), fill=(255, 255, 255), width=1)
    fnt = font(22)
    draw.rectangle((8, 8, 900, 48), fill=(0, 0, 0))
    draw.text((16, 14), title, fill=(255, 255, 255), font=fnt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Prediction JSON or CSV.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--input-type", default="external", help="Free text for model_comparison.csv.")
    parser.add_argument("--coordinate-space", choices=["crop", "warped"], default="crop")
    parser.add_argument("--ground-truth-dir", default=str(DEFAULT_GT_DIR))
    parser.add_argument("--split-manifest", default=str(DEFAULT_SPLIT))
    parser.add_argument("--crop-metadata", default=str(DEFAULT_CROP))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--eval-split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    output = Path(args.output_dir)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    gt = load_ground_truth(Path(args.ground_truth_dir))
    split = load_split(Path(args.split_manifest))
    crop_meta = load_crop_metadata(Path(args.crop_metadata))
    preds = load_predictions(Path(args.predictions))

    rows = []
    image_rows = []
    for base, pred in preds.items():
        if base not in gt:
            continue
        split_info = split.get(base, {})
        if split_info.get("split") != args.eval_split:
            continue
        crop = crop_meta.get(base, {})
        pred_warped = to_warped_points(pred["points"], base, args.coordinate_space, crop)
        gt_points = gt[base]["points"]
        mm_per_pixel = gt[base]["mm_per_pixel"]
        errors = []
        large = []
        for key in KEYPOINT_KEYS:
            err_px = distance(pred_warped.get(key), gt_points.get(key))
            err_mm = err_px * mm_per_pixel if err_px is not None else np.nan
            if pd.notna(err_mm):
                errors.append(float(err_mm))
            if pd.notna(err_mm) and err_mm > 10:
                large.append(key)
            rows.append(
                {
                    "model_name": args.model_name,
                    "image_name": gt[base]["image_name"],
                    "specimen_id": split_info.get("specimen_id") or gt[base]["specimen_id"],
                    "split": split_info.get("split", ""),
                    "keypoint_name": key,
                    "error_px": err_px if err_px is not None else np.nan,
                    "error_mm": err_mm,
                    "confidence": pred.get("confidences", {}).get(key, np.nan),
                }
            )
        image_rows.append(
            {
                "model_name": args.model_name,
                "image_name": gt[base]["image_name"],
                "specimen_id": split_info.get("specimen_id") or gt[base]["specimen_id"],
                "mean_error_mm": float(np.mean(errors)) if errors else np.nan,
                "median_error_mm": float(np.median(errors)) if errors else np.nan,
                "max_error_mm": float(np.max(errors)) if errors else np.nan,
                "num_large_error_points": len(large),
                "large_error_keypoints": ";".join(large),
            }
        )
        image_path = PROJECT_ROOT / str(crop.get("source_warped_image_path", ""))
        save_visual(
            image_path,
            output / "visual_comparisons" / f"{base}_{args.model_name}_compare.png",
            gt_points,
            pred_warped,
            f"{args.model_name} | {base} | median={image_rows[-1]['median_error_mm']:.2f} mm",
        )

    summary = pd.DataFrame(rows)
    by_image = pd.DataFrame(image_rows)
    by_point = (
        summary.groupby("keypoint_name")
        .agg(
            num_test_images=("image_name", "count"),
            mean_error_mm=("error_mm", "mean"),
            median_error_mm=("error_mm", "median"),
            max_error_mm=("error_mm", "max"),
            mean_confidence=("confidence", "mean"),
        )
        .reset_index()
        if not summary.empty
        else pd.DataFrame()
    )
    summary.to_csv(output / "keypoint_error_summary.csv", index=False, encoding="utf-8-sig")
    by_point.to_csv(output / "keypoint_error_by_point.csv", index=False, encoding="utf-8-sig")
    by_image.to_csv(output / "keypoint_error_by_image.csv", index=False, encoding="utf-8-sig")
    comparison_row = {
        "model_name": args.model_name,
        "input_type": args.input_type,
        "mean_error_mm": summary["error_mm"].mean() if not summary.empty else np.nan,
        "median_error_mm": summary["error_mm"].median() if not summary.empty else np.nan,
        "P2_median_error_mm": summary[summary["keypoint_name"] == "P2_eye_front"]["error_mm"].median() if not summary.empty else np.nan,
        "P3_median_error_mm": summary[summary["keypoint_name"] == "P3_operculum_posterior"]["error_mm"].median() if not summary.empty else np.nan,
        "P5_median_error_mm": summary[summary["keypoint_name"] == "P5_caudal_base_midpoint"]["error_mm"].median() if not summary.empty else np.nan,
        "P6_median_error_mm": summary[summary["keypoint_name"] == "P6_caudal_fork_midpoint"]["error_mm"].median() if not summary.empty else np.nan,
        "P7U_median_error_mm": summary[summary["keypoint_name"] == "P7U_caudal_fin_upper_tip"]["error_mm"].median() if not summary.empty else np.nan,
        "P7L_median_error_mm": summary[summary["keypoint_name"] == "P7L_caudal_fin_lower_tip"]["error_mm"].median() if not summary.empty else np.nan,
        "num_test_images": by_image["image_name"].nunique() if not by_image.empty else 0,
        "notes": args.notes,
    }
    comparison_path = output / "model_comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        comparison = comparison[comparison["model_name"] != args.model_name]
        comparison = pd.concat([comparison, pd.DataFrame([comparison_row])], ignore_index=True)
    else:
        comparison = pd.DataFrame([comparison_row])
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    print(f"Evaluated {args.model_name} on {comparison_row['num_test_images']} test images.")
    print(f"Median error: {comparison_row['median_error_mm']:.3f} mm")
    print(f"Results saved to {output.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
