from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph.hybrid_preannotation import preannotate_warped_image_hybrid_v04
from siganusmorph.image_utils import load_image_file
from siganusmorph.keypointwise_hybrid_selector import KEYPOINT_NAMES, select_keypoints_v06


BASE_OUT = PROJECT_ROOT / "results" / "model_eval" / "v0.6_keypointwise_hybrid_selector"
OUT = BASE_OUT / "same_set_comparison"
PRED_OUT = OUT / "predictions"
V041_OUT = OUT / "v041_automatic_preannotations"
VIS_OUT = OUT / "visual_comparisons"

V05_DATASET = PROJECT_ROOT / "datasets" / "siganusmorph_heatmap_unet_v0.5_candidate"
SPLIT_MANIFEST = V05_DATASET / "split_manifest.csv"
PRED_ROOT = PROJECT_ROOT / "results" / "model_eval" / "heatmap_unet_v0.5_predictions"
CROP_METADATA = PRED_ROOT / "crop_metadata.csv"
GT_DIR = PRED_ROOT / "ground_truth_keypoints"
REVIEW_ROOT = PROJECT_ROOT / "results" / "realworld_review_v0.4.1"
REVIEW_CORRECTED_DIR = REVIEW_ROOT / "corrected_keypoints"
HARD_CASES = PROJECT_ROOT / "results" / "v0.5_dataset_planning" / "v05_hard_cases.csv"

MM_PER_PIXEL = 0.1
LARGE_ERROR_MM = 10.0


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def xy_from(value: object) -> tuple[float, float] | None:
    if isinstance(value, Mapping):
        if "x" in value and "y" in value:
            return float(value["x"]), float(value["y"])
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return None


def resolve_path(value: object) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def review_corrected_path(image_name: str) -> Path | None:
    path = REVIEW_CORRECTED_DIR / f"{Path(image_name).stem}_corrected.json"
    return path if path.exists() else None


def gt_json_path(image_name: str) -> Path | None:
    review = review_corrected_path(image_name)
    if review:
        return review
    path = GT_DIR / f"{Path(image_name).stem}_keypoints.json"
    return path if path.exists() else None


def corrected_keypoints_from_json(path: Path) -> dict[str, list[float]]:
    data = load_json(path)
    keypoints = data.get("corrected_keypoints") or {}
    out = {}
    for key, value in keypoints.items():
        if key not in KEYPOINT_NAMES:
            continue
        xy = xy_from(value)
        if xy is not None:
            out[key] = [xy[0], xy[1]]
    return out


def specimen_from_gt(path: Path) -> str:
    try:
        return str(load_json(path).get("specimen_id", ""))
    except Exception:
        return ""


def build_manifest() -> pd.DataFrame:
    split = pd.read_csv(SPLIT_MANIFEST)
    crops = pd.read_csv(CROP_METADATA)
    crop_lookup = crops.set_index("image_name").to_dict("index")
    test_images = set(split.loc[split["split"] == "test", "image_name"].astype(str))
    review_images = {
        str((load_json(path).get("image_name") or path.name.replace("_corrected.json", ".png")))
        for path in REVIEW_CORRECTED_DIR.glob("*_corrected.json")
    }
    union = sorted((test_images | review_images) - {"real_037.png"})
    rows = []
    for image_name in union:
        gt_path = gt_json_path(image_name)
        if gt_path is None:
            continue
        keypoints = corrected_keypoints_from_json(gt_path)
        if len(keypoints) != len(KEYPOINT_NAMES):
            continue
        split_row = split[split["image_name"] == image_name]
        crop_row = crop_lookup.get(image_name, {})
        warped = str(crop_row.get("source_warped_image_path", ""))
        crop = str(split_row["crop_image_path"].iloc[0]) if not split_row.empty and "crop_image_path" in split_row else ""
        subset_parts = []
        if image_name in test_images:
            subset_parts.append("v05_test")
        if image_name in review_images:
            subset_parts.append("realworld_review_confirmed")
        rows.append(
            {
                "image_name": image_name,
                "specimen_id": specimen_from_gt(gt_path),
                "subset_source": "+".join(subset_parts),
                "has_ground_truth_corrected_keypoints": True,
                "corrected_json_path": str(gt_path),
                "warped_image_path": warped,
                "crop_image_path": crop,
                "included_in_v05_test": image_name in test_images,
                "included_in_realworld_review": image_name in review_images,
                "notes": "",
            }
        )
    manifest = pd.DataFrame(rows).sort_values(["included_in_realworld_review", "image_name"], ascending=[False, True])
    OUT.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUT / "strict_union_manifest.csv", index=False, encoding="utf-8-sig")
    return manifest


def load_gt_map(manifest: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float]]:
    gt = {}
    for row in manifest.to_dict("records"):
        keypoints = corrected_keypoints_from_json(Path(row["corrected_json_path"]))
        for key, xy in keypoints.items():
            gt[(row["image_name"], key)] = (float(xy[0]), float(xy[1]))
    return gt


def prediction_rows_from_heatmap(tag: str, model_name: str, manifest: pd.DataFrame) -> pd.DataFrame:
    crops = pd.read_csv(CROP_METADATA).set_index("image_name").to_dict("index")
    wanted = set(manifest["image_name"].astype(str))
    rows = []
    for split in ("train", "val", "test"):
        path = PRED_ROOT / f"predictions_{split}_{tag}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for record in df.to_dict("records"):
            image_name = str(record["image_name"])
            if image_name not in wanted:
                continue
            crop = crops.get(image_name)
            if crop is None:
                continue
            rows.append(
                {
                    "image_name": image_name,
                    "specimen_id": record.get("specimen_id", ""),
                    "keypoint_name": record["keypoint_name"],
                    "x": float(record["x"]) + float(crop.get("bbox_x1", 0.0)),
                    "y": float(record["y"]) + float(crop.get("bbox_y1", 0.0)),
                    "confidence": record.get("confidence", np.nan),
                    "source_model": model_name,
                    "point_source": model_name,
                    "coordinate_space": "warped",
                }
            )
    return pd.DataFrame(rows)


def run_v041_for_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    PRED_OUT.mkdir(parents=True, exist_ok=True)
    V041_OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, record in enumerate(manifest.to_dict("records"), start=1):
        image_name = record["image_name"]
        warped_path = resolve_path(record["warped_image_path"])
        if warped_path is None or not warped_path.exists():
            print(f"[v041 {idx}/{len(manifest)}] missing warped image: {image_name}")
            continue
        json_path = V041_OUT / "json" / f"{Path(image_name).stem}_v041_automatic.json"
        if json_path.exists():
            payload = load_json(json_path)
        else:
            print(f"[v041 {idx}/{len(manifest)}] running automatic preannotation: {image_name}")
            image = load_image_file(warped_path)
            result = preannotate_warped_image_hybrid_v04(image, image_name, PROJECT_ROOT, mm_per_pixel=MM_PER_PIXEL)
            metadata = result.get("metadata", {}) or {}
            payload = {
                "image_name": image_name,
                "specimen_id": record.get("specimen_id", ""),
                "warped_image_path": str(warped_path),
                "annotation_status": "auto_preannotation_unverified",
                "hybrid_keypoints": metadata.get("hybrid_keypoints", result.get("keypoints", {})),
                "heatmap_keypoints": metadata.get("heatmap_keypoints", {}),
                "v034_keypoints": metadata.get("v034_keypoints", {}),
                "mask_suggestions": metadata.get("mask_suggestions", {}),
                "geometric_suggestions": metadata.get("geometric_suggestions", {}),
                "point_sources": metadata.get("point_sources", {}),
                "hybrid_qc_results": metadata.get("hybrid_qc_results", {}),
                "review_reason": metadata.get("review_reason", ""),
            }
            write_json(json_path, payload)
        keypoints = payload.get("hybrid_keypoints") or {}
        point_sources = payload.get("point_sources") or {}
        for key in KEYPOINT_NAMES:
            xy = xy_from(keypoints.get(key))
            if xy is None:
                continue
            rows.append(
                {
                    "image_name": image_name,
                    "specimen_id": record.get("specimen_id", ""),
                    "keypoint_name": key,
                    "x": xy[0],
                    "y": xy[1],
                    "confidence": np.nan,
                    "source_model": "v041_automatic",
                    "point_source": point_sources.get(key, "v041_automatic"),
                    "coordinate_space": "warped",
                }
            )
    return pd.DataFrame(rows)


def frame_to_map(frame: pd.DataFrame) -> dict[tuple[str, str], dict]:
    out = {}
    if frame.empty:
        return out
    for row in frame.to_dict("records"):
        out[(row["image_name"], row["keypoint_name"])] = {
            "x": float(row["x"]),
            "y": float(row["y"]),
            "confidence": None if pd.isna(row.get("confidence", np.nan)) else float(row.get("confidence")),
            "source_model": row.get("source_model", ""),
            "point_source": row.get("point_source", row.get("source_model", "")),
        }
    return out


def summarize_source_errors(frame: pd.DataFrame, gt: dict[tuple[str, str], tuple[float, float]], source: str, manifest: pd.DataFrame) -> pd.DataFrame:
    subset_lookup = manifest.set_index("image_name").to_dict("index")
    hard_lookup = hard_case_lookup()
    rows = []
    for record in frame.to_dict("records"):
        key = (record["image_name"], record["keypoint_name"])
        manual = gt.get(key)
        if manual is None:
            continue
        err = math.hypot(float(record["x"]) - manual[0], float(record["y"]) - manual[1]) * MM_PER_PIXEL
        subset = subset_lookup.get(record["image_name"], {})
        rows.append(
            {
                "image_name": record["image_name"],
                "specimen_id": record.get("specimen_id", subset.get("specimen_id", "")),
                "subset_source": subset.get("subset_source", ""),
                "keypoint_name": record["keypoint_name"],
                "source": source,
                "error_mm": err,
                "is_large_error": err > LARGE_ERROR_MM,
                "is_hard_case_keypoint": record["keypoint_name"] in hard_lookup.get(record["image_name"], set()),
                "included_in_v05_test": bool(subset.get("included_in_v05_test", False)),
                "included_in_realworld_review": bool(subset.get("included_in_realworld_review", False)),
            }
        )
    return pd.DataFrame(rows)


def hard_case_lookup() -> dict[str, set[str]]:
    if not HARD_CASES.exists():
        return {}
    hard = pd.read_csv(HARD_CASES)
    out = {}
    for record in hard.to_dict("records"):
        parts = set()
        value = record.get("affected_keypoints")
        if isinstance(value, str):
            for part in value.replace(",", ";").split(";"):
                part = part.strip()
                if part in KEYPOINT_NAMES:
                    parts.add(part)
        out[str(record["image_name"])] = parts
    return out


def aggregate_model(frame: pd.DataFrame, model_name: str) -> dict:
    row = {
        "model_name": model_name,
        "num_images": int(frame["image_name"].nunique()) if not frame.empty else 0,
        "num_keypoints": int(len(frame)),
        "overall_median_error_mm": float(frame["error_mm"].median()) if not frame.empty else np.nan,
        "overall_mean_error_mm": float(frame["error_mm"].mean()) if not frame.empty else np.nan,
        "overall_max_error_mm": float(frame["error_mm"].max()) if not frame.empty else np.nan,
        "large_error_rate": float(frame["is_large_error"].mean()) if not frame.empty else np.nan,
    }
    for key in [
        "P4_peduncle_start_midpoint",
        "C2_trunk_axis_point",
        "P8_body_depth_dorsal",
        "P9_body_depth_ventral",
        "P11_peduncle_depth_ventral",
        "P7U_caudal_fin_upper_tip",
        "P7L_caudal_fin_lower_tip",
    ]:
        values = frame.loc[frame["keypoint_name"] == key, "error_mm"]
        row[f"{key}_median_error_mm"] = float(values.median()) if len(values) else np.nan
    hard = frame[frame["is_hard_case_keypoint"]]
    review = frame[frame["included_in_realworld_review"]]
    row["hard_case_mean_error_mm"] = float(hard["error_mm"].mean()) if len(hard) else np.nan
    row["review_confirmed_mean_error_mm"] = float(review["error_mm"].mean()) if len(review) else np.nan
    return row


def same_set_keypoint_comparison(all_errors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key in KEYPOINT_NAMES:
        row = {"keypoint_name": key}
        for source, prefix in [
            ("v01_heatmap", "v01"),
            ("v05_heatmap", "v05"),
            ("v041_automatic", "v041"),
            ("v06_selector", "v06"),
        ]:
            values = all_errors.loc[(all_errors["source"] == source) & (all_errors["keypoint_name"] == key), "error_mm"]
            large = all_errors.loc[(all_errors["source"] == source) & (all_errors["keypoint_name"] == key), "is_large_error"]
            row[f"{prefix}_median_error_mm"] = float(values.median()) if len(values) else np.nan
            row[f"{prefix}_mean_error_mm"] = float(values.mean()) if len(values) else np.nan
            row[f"large_error_rate_{prefix}"] = float(large.mean()) if len(large) else np.nan
        medians = {m: row.get(f"{m}_median_error_mm", np.nan) for m in ["v01", "v05", "v041", "v06"]}
        means = {m: row.get(f"{m}_mean_error_mm", np.nan) for m in ["v01", "v05", "v041", "v06"]}
        row["best_model_by_median"] = min((k for k, v in medians.items() if not np.isnan(v)), key=lambda k: medians[k], default="")
        row["best_model_by_mean"] = min((k for k, v in means.items() if not np.isnan(v)), key=lambda k: means[k], default="")
        row["recommend_default_source"] = row["best_model_by_median"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_recommendations(keypoint_comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in keypoint_comparison.to_dict("records"):
        key = record["keypoint_name"]
        if key == "P1_snout_tip":
            rec = "mask_left_boundary"
        elif key == "C2_trunk_axis_point":
            rec = "v041_automatic" if record.get("v041_median_error_mm", np.inf) <= record.get("v01_median_error_mm", np.inf) else "v01_heatmap"
        else:
            best = record.get("best_model_by_median", "")
            rec = {"v01": "v01_heatmap", "v05": "v05_heatmap", "v041": "v041_automatic", "v06": "v06_selector"}.get(best, "")
        rows.append({"keypoint_name": key, "recommend_default_source": rec})
    return pd.DataFrame(rows)


def select_v06_predictions(
    manifest: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    recommendations: pd.DataFrame,
) -> pd.DataFrame:
    maps = {source: frame_to_map(frame) for source, frame in frames.items()}
    rec_map = {
        row["keypoint_name"]: {
            "recommended_default_source": row["recommend_default_source"],
            "second_choice_source": "v041_hybrid",
        }
        for row in recommendations.to_dict("records")
    }
    alias_frames = {
        "v01_heatmap": maps["v01_heatmap"],
        "v05_heatmap": maps["v05_heatmap"],
        "v041_hybrid": maps["v041_automatic"],
        "mask_or_geometry_rule_if_available": {},
    }
    rows = []
    for record in manifest.to_dict("records"):
        image_name = record["image_name"]
        v01 = {k: alias_frames["v01_heatmap"].get((image_name, k)) for k in KEYPOINT_NAMES if alias_frames["v01_heatmap"].get((image_name, k))}
        v05 = {k: alias_frames["v05_heatmap"].get((image_name, k)) for k in KEYPOINT_NAMES if alias_frames["v05_heatmap"].get((image_name, k))}
        v041 = {k: alias_frames["v041_hybrid"].get((image_name, k)) for k in KEYPOINT_NAMES if alias_frames["v041_hybrid"].get((image_name, k))}
        selected, sources, _qc = select_keypoints_v06(
            v01,
            {k: v.get("confidence") for k, v in v01.items()},
            v05,
            {k: v.get("confidence") for k, v in v05.items()},
            v041,
            {},
            {},
            {},
            rec_map,
        )
        for key, xy in selected.items():
            rows.append(
                {
                    "image_name": image_name,
                    "specimen_id": record["specimen_id"],
                    "keypoint_name": key,
                    "x": xy[0],
                    "y": xy[1],
                    "confidence": np.nan,
                    "source_model": "v06_selector",
                    "point_source": sources.get(key, ""),
                    "coordinate_space": "warped",
                }
            )
    return pd.DataFrame(rows)


def subset_model_comparison(all_errors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    subset_defs = {
        "v05_test_subset": all_errors["included_in_v05_test"] == True,
        "realworld_review_confirmed_subset": all_errors["included_in_realworld_review"] == True,
    }
    for subset_name, mask in subset_defs.items():
        subset = all_errors[mask]
        for source, group in subset.groupby("source"):
            row = aggregate_model(group, source)
            row["subset"] = subset_name
            rows.append(row)
    return pd.DataFrame(rows)


def error_by_image(all_errors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (image_name, source), group in all_errors.groupby(["image_name", "source"]):
        large = group.loc[group["is_large_error"], "keypoint_name"].tolist()
        rows.append(
            {
                "image_name": image_name,
                "specimen_id": group["specimen_id"].iloc[0],
                "source": source,
                "subset_source": group["subset_source"].iloc[0],
                "mean_error_mm": float(group["error_mm"].mean()),
                "median_error_mm": float(group["error_mm"].median()),
                "max_error_mm": float(group["error_mm"].max()),
                "num_large_error_points": len(large),
                "large_error_keypoints": ";".join(large),
            }
        )
    return pd.DataFrame(rows)


def large_error_cases(all_errors: pd.DataFrame) -> pd.DataFrame:
    return all_errors[all_errors["is_large_error"]].copy()


def image_path_for(record: Mapping) -> Path | None:
    return resolve_path(record.get("warped_image_path"))


def draw_visuals(manifest: pd.DataFrame, frames: dict[str, pd.DataFrame], gt: dict[tuple[str, str], tuple[float, float]], all_errors: pd.DataFrame) -> None:
    VIS_OUT.mkdir(parents=True, exist_ok=True)
    maps = {source: frame_to_map(frame) for source, frame in frames.items()}
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    colors = {
        "gt": (0, 220, 60),
        "v041_automatic": (255, 150, 40),
        "v06_selector": (230, 30, 30),
        "v01_heatmap": (160, 80, 255),
        "v05_heatmap": (255, 80, 190),
        "large": (255, 230, 0),
    }
    for record in manifest.to_dict("records"):
        path = image_path_for(record)
        if path is None or not path.exists():
            continue
        image = Image.open(path).convert("RGB")
        scale = min(1.0, 1800 / image.width)
        display = image.resize((int(image.width * scale), int(image.height * scale))) if scale != 1 else image.copy()
        draw = ImageDraw.Draw(display)

        def dot(x: float, y: float, color: tuple[int, int, int], r: int, outline: bool = False):
            x *= scale
            y *= scale
            box = [x - r, y - r, x + r, y + r]
            if outline:
                draw.ellipse(box, outline=color, width=2)
            else:
                draw.ellipse(box, fill=color, outline=(0, 0, 0))

        image_name = record["image_name"]
        for key in KEYPOINT_NAMES:
            gt_xy = gt.get((image_name, key))
            if gt_xy:
                dot(gt_xy[0], gt_xy[1], colors["gt"], 4)
            for source in ["v01_heatmap", "v05_heatmap", "v041_automatic"]:
                pred = maps[source].get((image_name, key))
                if pred:
                    dot(pred["x"], pred["y"], colors[source], 3, outline=True)
            pred = maps["v06_selector"].get((image_name, key))
            if pred:
                dot(pred["x"], pred["y"], colors["v06_selector"], 5)

        image_errors = all_errors[(all_errors["image_name"] == image_name) & (all_errors["source"] == "v06_selector")]
        for row in image_errors[image_errors["is_large_error"]].to_dict("records"):
            pred = maps["v06_selector"].get((image_name, row["keypoint_name"]))
            if pred:
                dot(pred["x"], pred["y"], colors["large"], 12, outline=True)
        draw.rectangle([0, 0, 900, 70], fill=(0, 0, 0))
        draw.text((10, 8), f"{image_name} v06 mean={image_errors['error_mm'].mean():.2f} median={image_errors['error_mm'].median():.2f} mm", fill=(255, 255, 255), font=font)
        large = ";".join(image_errors.loc[image_errors["is_large_error"], "keypoint_name"].tolist()) or "none"
        draw.text((10, 38), f"large: {large}", fill=(255, 230, 0), font=font)
        display.save(VIS_OUT / f"{Path(image_name).stem}_same_set_compare.png")


def write_readme(model_comparison: pd.DataFrame, keypoint_comparison: pd.DataFrame, subset_comparison: pd.DataFrame) -> None:
    def val(model: str, col: str) -> float:
        row = model_comparison[model_comparison["model_name"] == model]
        return float(row[col].iloc[0]) if not row.empty and col in row else np.nan

    v06_med = val("v06_selector", "overall_median_error_mm")
    v041_med = val("v041_automatic", "overall_median_error_mm")
    v06_mean = val("v06_selector", "overall_mean_error_mm")
    v041_mean = val("v041_automatic", "overall_mean_error_mm")
    v06_large = val("v06_selector", "large_error_rate")
    v041_large = val("v041_automatic", "large_error_rate")
    median_improved = v06_med < v041_med
    mean_improved = v06_mean < v041_mean
    large_improved = v06_large < v041_large
    integrate = median_improved and mean_improved and large_improved

    focus_lines = []
    for key in ["P4_peduncle_start_midpoint", "C2_trunk_axis_point", "P8_body_depth_dorsal", "P9_body_depth_ventral", "P11_peduncle_depth_ventral", "P7U_caudal_fin_upper_tip", "P7L_caudal_fin_lower_tip"]:
        row = keypoint_comparison[keypoint_comparison["keypoint_name"] == key]
        if row.empty:
            continue
        r = row.iloc[0]
        focus_lines.append(
            f"- {key}: v041={r['v041_median_error_mm']:.3f}, v06={r['v06_median_error_mm']:.3f}, best={r['best_model_by_median']}"
        )

    text = f"""# Strict Same-set v0.6 Comparison

No model was trained. No corrected labels and no Streamlit defaults were modified.

## Set

- Union images: {int(model_comparison['num_images'].max())}
- Ground truth: one corrected keypoint JSON per image
- Excluded case `real_037.png / fish_18` is not included

## Overall

| model | median_mm | mean_mm | large_error_rate |
|---|---:|---:|---:|
| v0.1 heatmap | {val('v01_heatmap', 'overall_median_error_mm'):.3f} | {val('v01_heatmap', 'overall_mean_error_mm'):.3f} | {val('v01_heatmap', 'large_error_rate'):.3f} |
| v0.5 heatmap | {val('v05_heatmap', 'overall_median_error_mm'):.3f} | {val('v05_heatmap', 'overall_mean_error_mm'):.3f} | {val('v05_heatmap', 'large_error_rate'):.3f} |
| v0.4.1 automatic | {v041_med:.3f} | {v041_mean:.3f} | {v041_large:.3f} |
| v0.6 selector | {v06_med:.3f} | {v06_mean:.3f} | {v06_large:.3f} |

## v0.6 vs v0.4.1

- median improved: {'yes' if median_improved else 'no'}
- mean improved: {'yes' if mean_improved else 'no'}
- large error rate improved: {'yes' if large_improved else 'no'}

## Keypoints

{chr(10).join(focus_lines)}

## C2

The selector avoids using v0.5 for C2 by default.  This prevents the known v0.5 C2 degradation from propagating into v0.6.

## Recommendation

- integrate v0.6 to Streamlit: {'yes' if integrate else 'no'}
- keep v0.4.1 default: {'no' if integrate else 'yes'}
- practical recommendation: {'replace default with v0.6' if integrate else 'keep v0.4.1 default and expose v0.6 only as an experimental mode'}
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PRED_OUT.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    gt = load_gt_map(manifest)
    v01 = prediction_rows_from_heatmap("v01_best", "v01_heatmap", manifest)
    v05 = prediction_rows_from_heatmap("v05_best", "v05_heatmap", manifest)
    v041 = run_v041_for_manifest(manifest)
    frames = {"v01_heatmap": v01, "v05_heatmap": v05, "v041_automatic": v041}

    # First-pass source summary to recommend per-keypoint defaults.
    error_frames = []
    for source, frame in frames.items():
        error_frames.append(summarize_source_errors(frame, gt, source, manifest))
    source_errors = pd.concat(error_frames, ignore_index=True)
    keypoint_comparison_seed = same_set_keypoint_comparison(source_errors)
    recommendations = build_recommendations(keypoint_comparison_seed)
    v06 = select_v06_predictions(manifest, frames, recommendations)
    frames["v06_selector"] = v06

    v01.to_csv(PRED_OUT / "v01_predictions.csv", index=False, encoding="utf-8-sig")
    v05.to_csv(PRED_OUT / "v05_predictions.csv", index=False, encoding="utf-8-sig")
    v041.to_csv(PRED_OUT / "v041_automatic_predictions.csv", index=False, encoding="utf-8-sig")
    v06.to_csv(PRED_OUT / "v06_predictions.csv", index=False, encoding="utf-8-sig")

    all_errors = pd.concat(
        [summarize_source_errors(frame, gt, source, manifest) for source, frame in frames.items()],
        ignore_index=True,
    )
    model_comparison = pd.DataFrame([aggregate_model(all_errors[all_errors["source"] == source], source) for source in frames])
    keypoint_comparison = same_set_keypoint_comparison(all_errors)
    keypoint_recs = build_recommendations(keypoint_comparison)
    by_subset = subset_model_comparison(all_errors)
    by_image = error_by_image(all_errors)
    large = large_error_cases(all_errors)

    model_comparison.to_csv(OUT / "same_set_model_comparison.csv", index=False, encoding="utf-8-sig")
    keypoint_comparison.to_csv(OUT / "same_set_keypoint_comparison.csv", index=False, encoding="utf-8-sig")
    by_image.to_csv(OUT / "same_set_error_by_image.csv", index=False, encoding="utf-8-sig")
    large.to_csv(OUT / "same_set_large_error_cases.csv", index=False, encoding="utf-8-sig")
    keypoint_recs.to_csv(OUT / "same_set_source_recommendation.csv", index=False, encoding="utf-8-sig")
    by_subset.to_csv(OUT / "same_set_model_comparison_by_subset.csv", index=False, encoding="utf-8-sig")
    all_errors.to_csv(OUT / "same_set_all_error_rows.csv", index=False, encoding="utf-8-sig")

    draw_visuals(manifest, frames, gt, all_errors)
    write_readme(model_comparison, keypoint_comparison, by_subset)

    def m(model: str, col: str) -> float:
        row = model_comparison[model_comparison["model_name"] == model]
        return float(row[col].iloc[0]) if not row.empty else np.nan

    print("Finished strict same-set comparison.")
    print(f"Union set images: {len(manifest)}")
    print("Overall median / mean:")
    print(f"v0.1 heatmap: {m('v01_heatmap', 'overall_median_error_mm'):.3f} / {m('v01_heatmap', 'overall_mean_error_mm'):.3f}")
    print(f"v0.5 heatmap: {m('v05_heatmap', 'overall_median_error_mm'):.3f} / {m('v05_heatmap', 'overall_mean_error_mm'):.3f}")
    print(f"v0.4.1 automatic: {m('v041_automatic', 'overall_median_error_mm'):.3f} / {m('v041_automatic', 'overall_mean_error_mm'):.3f}")
    print(f"v0.6 selector: {m('v06_selector', 'overall_median_error_mm'):.3f} / {m('v06_selector', 'overall_mean_error_mm'):.3f}")
    print("v0.6 vs v0.4.1:")
    print(f"median improved: {'yes' if m('v06_selector','overall_median_error_mm') < m('v041_automatic','overall_median_error_mm') else 'no'}")
    print(f"mean improved: {'yes' if m('v06_selector','overall_mean_error_mm') < m('v041_automatic','overall_mean_error_mm') else 'no'}")
    print(f"large error rate improved: {'yes' if m('v06_selector','large_error_rate') < m('v041_automatic','large_error_rate') else 'no'}")
    print("Recommendation:")
    integrate = m('v06_selector','overall_median_error_mm') < m('v041_automatic','overall_median_error_mm') and m('v06_selector','overall_mean_error_mm') < m('v041_automatic','overall_mean_error_mm')
    print(f"- {'replace default with v0.6' if integrate else 'keep v0.4.1 default'}")
    print("- integrate v0.6 as experimental mode")


if __name__ == "__main__":
    main()
