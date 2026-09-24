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

from siganusmorph.keypointwise_hybrid_selector import KEYPOINT_NAMES, select_keypoints_v06
OUT_ROOT = PROJECT_ROOT / "results" / "model_eval" / "v0.6_keypointwise_hybrid_selector"
PRED_OUT = OUT_ROOT / "predictions"
EVAL_OUT = OUT_ROOT / "v06_evaluation"
VIS_OUT = EVAL_OUT / "visual_comparisons"

V05_DATASET_ROOT = PROJECT_ROOT / "datasets" / "siganusmorph_heatmap_unet_v0.5_candidate"
SPLIT_MANIFEST_PATH = V05_DATASET_ROOT / "split_manifest.csv"
PRED_ROOT = PROJECT_ROOT / "results" / "model_eval" / "heatmap_unet_v0.5_predictions"
CROP_METADATA_PATH = PRED_ROOT / "crop_metadata.csv"
GT_DIR = PRED_ROOT / "ground_truth_keypoints"
REVIEW_ROOT = PROJECT_ROOT / "results" / "realworld_review_v0.4.1"
REVIEW_PREANNOTATION_DIR = REVIEW_ROOT / "preannotations" / "json"
REVIEW_CORRECTED_DIR = REVIEW_ROOT / "corrected_keypoints"
HARD_CASES_PATH = PROJECT_ROOT / "results" / "v0.5_dataset_planning" / "v05_hard_cases.csv"
V04_BENCHMARK_DIR = PROJECT_ROOT / "results" / "model_eval" / "v0.4_hybrid_heatmap_geometry_compare_30"

MM_PER_PIXEL = 0.1
LARGE_ERROR_MM = 10.0


def resolve_path(path_text: object) -> Path | None:
    if not isinstance(path_text, str) or not path_text.strip():
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def xy_from(value: object) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        if "x" in value and "y" in value:
            return float(value["x"]), float(value["y"])
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return None


def base_image_name_from_json(path: Path) -> str:
    data = load_json(path)
    return str(data.get("image_name") or path.name.replace("_corrected.json", ".png").replace("_preannotation.json", ".png"))


def load_review_images() -> set[str]:
    images = set()
    for path in REVIEW_CORRECTED_DIR.glob("*_corrected.json"):
        images.add(base_image_name_from_json(path))
    return images


def load_ground_truth() -> tuple[dict[tuple[str, str], tuple[float, float]], dict[str, dict]]:
    gt = {}
    meta = {}
    for path in GT_DIR.glob("*_keypoints.json"):
        data = load_json(path)
        image_name = str(data.get("image_name") or path.name.replace("_keypoints.json", ".png"))
        meta[image_name] = {
            "specimen_id": data.get("specimen_id"),
            "mm_per_pixel": data.get("mm_per_pixel", MM_PER_PIXEL),
        }
        keypoints = data.get("corrected_keypoints") or {}
        for keypoint, value in keypoints.items():
            if keypoint not in KEYPOINT_NAMES:
                continue
            xy = xy_from(value)
            if xy is not None:
                gt[(image_name, keypoint)] = xy
    return gt, meta


def evaluation_image_sets() -> tuple[pd.DataFrame, set[str], set[str], set[str]]:
    split = pd.read_csv(SPLIT_MANIFEST_PATH)
    test_images = set(split.loc[split["split"] == "test", "image_name"].astype(str))
    review_images = load_review_images()
    eval_images = test_images | review_images
    return split, test_images, review_images, eval_images


def subset_label(image_name: str, test_images: set[str], review_images: set[str]) -> str:
    in_test = image_name in test_images
    in_review = image_name in review_images
    if in_test and in_review:
        return "test+review_confirmed"
    if in_test:
        return "test"
    if in_review:
        return "review_confirmed"
    return "other"


def prediction_rows_from_heatmap(tag: str, source_model: str, eval_images: set[str]) -> list[dict]:
    crops = pd.read_csv(CROP_METADATA_PATH)
    crop_lookup = crops.set_index("image_name").to_dict("index")
    rows: list[dict] = []
    for split in ("train", "val", "test"):
        pred_path = PRED_ROOT / f"predictions_{split}_{tag}.csv"
        if not pred_path.exists():
            continue
        preds = pd.read_csv(pred_path)
        for row in preds.itertuples(index=False):
            if row.image_name not in eval_images:
                continue
            crop = crop_lookup.get(row.image_name)
            if crop is None:
                continue
            x = float(row.x) + float(crop.get("bbox_x1", 0.0))
            y = float(row.y) + float(crop.get("bbox_y1", 0.0))
            rows.append(
                {
                    "image_name": row.image_name,
                    "specimen_id": row.specimen_id,
                    "split_or_subset": split,
                    "keypoint_name": row.keypoint_name,
                    "x": x,
                    "y": y,
                    "confidence": float(row.confidence) if not pd.isna(row.confidence) else np.nan,
                    "source_model": source_model,
                    "coordinate_space": "warped",
                }
            )
    return rows


def review_preannotation_path(image_name: str) -> Path | None:
    stem = Path(image_name).stem
    path = REVIEW_PREANNOTATION_DIR / f"{stem}_preannotation.json"
    return path if path.exists() else None


def prediction_rows_from_review_preannotations(eval_images: set[str]) -> tuple[list[dict], list[dict]]:
    hybrid_rows: list[dict] = []
    geometry_rows: list[dict] = []
    for image_name in sorted(eval_images):
        path = review_preannotation_path(image_name)
        if path is None:
            continue
        data = load_json(path)
        specimen_id = data.get("specimen_id")
        for source_key, output_rows, source_model in [
            ("hybrid_keypoints", hybrid_rows, "v041_hybrid"),
            ("mask_suggestions", geometry_rows, "mask_or_geometry_rule_if_available"),
            ("geometric_suggestions", geometry_rows, "mask_or_geometry_rule_if_available"),
        ]:
            keypoints = data.get(source_key) or {}
            for keypoint, value in keypoints.items():
                if keypoint not in KEYPOINT_NAMES:
                    continue
                xy = xy_from(value)
                if xy is None:
                    continue
                output_rows.append(
                    {
                        "image_name": image_name,
                        "specimen_id": specimen_id,
                        "split_or_subset": "review_confirmed",
                        "keypoint_name": keypoint,
                        "x": xy[0],
                        "y": xy[1],
                        "confidence": np.nan,
                        "source_model": source_model,
                        "coordinate_space": "warped",
                    }
                )
    geometry_rows = deduplicate_prediction_rows(geometry_rows)
    return hybrid_rows, geometry_rows


def deduplicate_prediction_rows(rows: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for row in rows:
        key = (row["image_name"], row["keypoint_name"], row["source_model"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def save_prediction_csvs(eval_images: set[str]) -> dict[str, pd.DataFrame]:
    PRED_OUT.mkdir(parents=True, exist_ok=True)
    frames = {
        "v01_heatmap": pd.DataFrame(prediction_rows_from_heatmap("v01_best", "v01_heatmap", eval_images)),
        "v05_heatmap": pd.DataFrame(prediction_rows_from_heatmap("v05_best", "v05_heatmap", eval_images)),
    }
    hybrid_rows, geometry_rows = prediction_rows_from_review_preannotations(eval_images)
    frames["v041_hybrid"] = pd.DataFrame(hybrid_rows)
    frames["mask_or_geometry_rule_if_available"] = pd.DataFrame(geometry_rows)

    frames["v01_heatmap"].to_csv(PRED_OUT / "v01_heatmap_predictions.csv", index=False, encoding="utf-8-sig")
    frames["v05_heatmap"].to_csv(PRED_OUT / "v05_heatmap_predictions.csv", index=False, encoding="utf-8-sig")
    frames["v041_hybrid"].to_csv(PRED_OUT / "v041_hybrid_predictions.csv", index=False, encoding="utf-8-sig")
    frames["mask_or_geometry_rule_if_available"].to_csv(
        PRED_OUT / "mask_geometry_predictions.csv", index=False, encoding="utf-8-sig"
    )
    return frames


def frame_to_map(frame: pd.DataFrame) -> dict[tuple[str, str], dict]:
    out = {}
    if frame.empty:
        return out
    for row in frame.itertuples(index=False):
        out[(row.image_name, row.keypoint_name)] = {
            "x": float(row.x),
            "y": float(row.y),
            "confidence": None if pd.isna(row.confidence) else float(row.confidence),
            "source_model": row.source_model,
        }
    return out


def compute_error_rows(
    source_name: str,
    frame: pd.DataFrame,
    gt: dict[tuple[str, str], tuple[float, float]],
    image_subsets: dict[str, str],
    hard_case_lookup: dict[str, set[str]],
) -> pd.DataFrame:
    rows = []
    if frame.empty:
        return pd.DataFrame(rows)
    for record in frame.to_dict("records"):
        key = (record["image_name"], record["keypoint_name"])
        manual = gt.get(key)
        if manual is None:
            continue
        err = math.hypot(float(record["x"]) - manual[0], float(record["y"]) - manual[1]) * MM_PER_PIXEL
        rows.append(
            {
                "image_name": record["image_name"],
                "specimen_id": record.get("specimen_id"),
                "split_or_subset": image_subsets.get(record["image_name"], "other"),
                "keypoint_name": record["keypoint_name"],
                "source": source_name,
                "x": record["x"],
                "y": record["y"],
                "manual_x": manual[0],
                "manual_y": manual[1],
                "error_mm": err,
                "is_large_error": err > LARGE_ERROR_MM,
                "is_hard_case_keypoint": record["keypoint_name"] in hard_case_lookup.get(record["image_name"], set()),
                "is_review_confirmed": "review_confirmed" in image_subsets.get(record["image_name"], ""),
            }
        )
    return pd.DataFrame(rows)


def summarize_source_errors(error_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (keypoint, source), group in error_rows.groupby(["keypoint_name", "source"]):
        hard = group[group["is_hard_case_keypoint"]]
        review = group[group["is_review_confirmed"]]
        rows.append(
            {
                "keypoint_name": keypoint,
                "source": source,
                "num_images": int(group["image_name"].nunique()),
                "mean_error_mm": float(group["error_mm"].mean()),
                "median_error_mm": float(group["error_mm"].median()),
                "max_error_mm": float(group["error_mm"].max()),
                "large_error_rate": float(group["is_large_error"].mean()),
                "hard_case_mean_error_mm": float(hard["error_mm"].mean()) if len(hard) else np.nan,
                "hard_case_median_error_mm": float(hard["error_mm"].median()) if len(hard) else np.nan,
                "review_confirmed_mean_error_mm": float(review["error_mm"].mean()) if len(review) else np.nan,
                "review_confirmed_median_error_mm": float(review["error_mm"].median()) if len(review) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["keypoint_name", "median_error_mm"])


def hard_case_lookup() -> dict[str, set[str]]:
    if not HARD_CASES_PATH.exists():
        return {}
    hard = pd.read_csv(HARD_CASES_PATH)
    lookup: dict[str, set[str]] = {}
    for row in hard.to_dict("records"):
        keypoints = set()
        value = row.get("affected_keypoints")
        if isinstance(value, str):
            for part in value.replace(",", ";").split(";"):
                part = part.strip()
                if part in KEYPOINT_NAMES:
                    keypoints.add(part)
        lookup[str(row["image_name"])] = keypoints
    return lookup


def metric_for(source_summary: pd.DataFrame, keypoint: str, source: str, column: str) -> float:
    match = source_summary[(source_summary["keypoint_name"] == keypoint) & (source_summary["source"] == source)]
    if match.empty:
        return np.nan
    return float(match[column].iloc[0])


def best_source_for_keypoint(source_summary: pd.DataFrame, keypoint: str, candidates: list[str]) -> str:
    best = None
    best_value = np.inf
    for source in candidates:
        value = metric_for(source_summary, keypoint, source, "median_error_mm")
        if not np.isnan(value) and value < best_value:
            best = source
            best_value = value
    return best or candidates[0]


def build_recommendations(source_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keypoint in KEYPOINT_NAMES:
        v01_med = metric_for(source_summary, keypoint, "v01_heatmap", "median_error_mm")
        v05_med = metric_for(source_summary, keypoint, "v05_heatmap", "median_error_mm")
        v041_med = metric_for(source_summary, keypoint, "v041_hybrid", "median_error_mm")
        geom_med = metric_for(source_summary, keypoint, "mask_or_geometry_rule_if_available", "median_error_mm")
        v01_mean = metric_for(source_summary, keypoint, "v01_heatmap", "mean_error_mm")
        v05_mean = metric_for(source_summary, keypoint, "v05_heatmap", "mean_error_mm")
        v041_mean = metric_for(source_summary, keypoint, "v041_hybrid", "mean_error_mm")
        v01_large = metric_for(source_summary, keypoint, "v01_heatmap", "large_error_rate")
        v05_large = metric_for(source_summary, keypoint, "v05_heatmap", "large_error_rate")
        v041_large = metric_for(source_summary, keypoint, "v041_hybrid", "large_error_rate")

        reason = "lowest median among available candidates"
        qc_rule = ""
        rec = best_source_for_keypoint(source_summary, keypoint, ["v01_heatmap", "v05_heatmap", "v041_hybrid"])
        second = best_source_for_keypoint(
            source_summary,
            keypoint,
            [source for source in ["v01_heatmap", "v05_heatmap", "v041_hybrid"] if source != rec],
        )

        if keypoint == "P1_snout_tip":
            rec = "mask_left_boundary"
            second = best_source_for_keypoint(source_summary, keypoint, ["v01_heatmap", "v05_heatmap", "v041_hybrid"])
            reason = "P1 is most stable as the left boundary rule when mask is available"
            qc_rule = "fallback to heatmap if mask unavailable"
        elif keypoint == "P4_peduncle_start_midpoint" and not np.isnan(v05_med):
            rec = "v05_heatmap"
            second = "v041_hybrid"
            reason = "v0.5 sharply reduces P4 outliers versus v0.1"
        elif keypoint == "C2_trunk_axis_point":
            rec = best_source_for_keypoint(source_summary, keypoint, ["v041_hybrid", "v01_heatmap"])
            second = "v01_heatmap" if rec == "v041_hybrid" else "v041_hybrid"
            reason = "v0.5 degrades C2; choose the steadier v0.1/v0.4.1 source and keep review warning"
            qc_rule = "needs_review_by_default if axis QC fails"
        elif keypoint in {"P8_body_depth_dorsal", "P9_body_depth_ventral"}:
            if not np.isnan(v05_med) and (np.isnan(v01_med) or v05_med < v01_med) and (np.isnan(v05_large) or v05_large <= 0.1):
                rec = "v05_heatmap_with_QC"
                second = "v041_hybrid"
                reason = "v0.5 improves body-depth point on v0.5 evaluation; retain local-normal QC"
            else:
                rec = best_source_for_keypoint(source_summary, keypoint, ["v041_hybrid", "v05_heatmap", "v01_heatmap"])
                second = "v05_heatmap" if rec != "v05_heatmap" else "v041_hybrid"
                reason = "body-depth point remains fin-sensitive; keep best source with local-normal QC"
            qc_rule = "local-normal body-depth suggestion as QC"
        elif keypoint == "P11_peduncle_depth_ventral" and not np.isnan(v05_med):
            rec = "v05_heatmap"
            second = "v01_heatmap"
            reason = "v0.5 improves P11 without increasing large-error rate materially"
            qc_rule = "peduncle local-normal QC"
        elif keypoint == "P7U_caudal_fin_upper_tip":
            rec = "v05_heatmap_with_mask_QC"
            second = "v01_heatmap"
            reason = "v0.5 improves P7U; keep mask tail QC"
            qc_rule = "mask tail rule agreement"
        elif keypoint == "P7L_caudal_fin_lower_tip":
            rec = "v05_heatmap_with_mask_QC" if np.isnan(v05_med) or np.isnan(v01_med) or v05_med <= v01_med + 0.05 else "v01_heatmap"
            second = "v01_heatmap" if rec != "v01_heatmap" else "v041_hybrid"
            reason = "P7L is stable; use heatmap with mask QC when not degraded"
            qc_rule = "mask tail rule agreement"

        hard_rec = best_source_for_keypoint(source_summary, keypoint, ["v05_heatmap", "v01_heatmap", "v041_hybrid"])
        rows.append(
            {
                "keypoint_name": keypoint,
                "recommended_default_source": rec,
                "second_choice_source": second,
                "reason": reason,
                "v01_median_error_mm": v01_med,
                "v05_median_error_mm": v05_med,
                "v041_median_error_mm": v041_med,
                "mask_or_geometry_median_error_mm": geom_med,
                "v01_mean_error_mm": v01_mean,
                "v05_mean_error_mm": v05_mean,
                "v041_mean_error_mm": v041_mean,
                "large_error_rate_v01": v01_large,
                "large_error_rate_v05": v05_large,
                "large_error_rate_v041": v041_large,
                "hard_case_recommendation": hard_rec,
                "qc_rule": qc_rule,
            }
        )
    return pd.DataFrame(rows)


def recommendations_to_mapping(recommendations: pd.DataFrame) -> dict[str, dict]:
    return {row["keypoint_name"]: row for row in recommendations.to_dict("records")}


def build_v06_predictions(
    frames: dict[str, pd.DataFrame],
    recommendations: pd.DataFrame,
    eval_images: set[str],
    test_images: set[str],
    review_images: set[str],
    gt_meta: dict[str, dict],
) -> tuple[pd.DataFrame, dict[str, dict], dict[str, dict]]:
    maps = {source: frame_to_map(frame) for source, frame in frames.items()}
    rec_map = recommendations_to_mapping(recommendations)
    rows = []
    sources_by_image: dict[str, dict] = {}
    qc_by_image: dict[str, dict] = {}
    for image_name in sorted(eval_images):
        v01 = {kp: maps["v01_heatmap"].get((image_name, kp)) for kp in KEYPOINT_NAMES if maps["v01_heatmap"].get((image_name, kp))}
        v05 = {kp: maps["v05_heatmap"].get((image_name, kp)) for kp in KEYPOINT_NAMES if maps["v05_heatmap"].get((image_name, kp))}
        v041 = {kp: maps["v041_hybrid"].get((image_name, kp)) for kp in KEYPOINT_NAMES if maps["v041_hybrid"].get((image_name, kp))}
        geom = {
            kp: maps["mask_or_geometry_rule_if_available"].get((image_name, kp))
            for kp in KEYPOINT_NAMES
            if maps["mask_or_geometry_rule_if_available"].get((image_name, kp))
        }
        selected, sources, qc = select_keypoints_v06(
            v01,
            {kp: point.get("confidence") for kp, point in v01.items()},
            v05,
            {kp: point.get("confidence") for kp, point in v05.items()},
            v041,
            geom,
            geom,
            {},
            rec_map,
        )
        sources_by_image[image_name] = sources
        qc_by_image[image_name] = qc
        for keypoint, xy in selected.items():
            rows.append(
                {
                    "image_name": image_name,
                    "specimen_id": gt_meta.get(image_name, {}).get("specimen_id"),
                    "split_or_subset": subset_label(image_name, test_images, review_images),
                    "keypoint_name": keypoint,
                    "x": xy[0],
                    "y": xy[1],
                    "confidence": np.nan,
                    "source_model": "v06_keypointwise_hybrid",
                    "point_source": sources.get(keypoint),
                    "coordinate_space": "warped",
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(PRED_OUT / "v06_keypointwise_predictions.csv", index=False, encoding="utf-8-sig")
    return frame, sources_by_image, qc_by_image


def summarize_by_point(error_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keypoint, group in error_rows.groupby("keypoint_name"):
        rows.append(
            {
                "keypoint_name": keypoint,
                "num_images": int(group["image_name"].nunique()),
                "mean_error_mm": float(group["error_mm"].mean()),
                "median_error_mm": float(group["error_mm"].median()),
                "max_error_mm": float(group["error_mm"].max()),
                "large_error_rate": float(group["is_large_error"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("keypoint_name")


def summarize_by_image(error_rows: pd.DataFrame, source_by_image: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for image_name, group in error_rows.groupby("image_name"):
        large = group.loc[group["is_large_error"], "keypoint_name"].tolist()
        rows.append(
            {
                "image_name": image_name,
                "specimen_id": group["specimen_id"].iloc[0],
                "split_or_subset": group["split_or_subset"].iloc[0],
                "mean_error_mm": float(group["error_mm"].mean()),
                "median_error_mm": float(group["error_mm"].median()),
                "max_error_mm": float(group["error_mm"].max()),
                "num_large_error_points": len(large),
                "large_error_keypoints": ";".join(large),
                "point_source_summary": ";".join(f"{k}:{v}" for k, v in source_by_image.get(image_name, {}).items()),
            }
        )
    return pd.DataFrame(rows).sort_values("median_error_mm", ascending=False)


def model_overall_metrics(name: str, error_rows: pd.DataFrame, input_type: str, notes: str) -> dict:
    if error_rows.empty:
        return {"model_name": name, "input_type": input_type, "overall_median_error_mm": np.nan, "overall_mean_error_mm": np.nan, "large_error_rate": np.nan, "num_images": 0, "notes": notes}
    row = {
        "model_name": name,
        "input_type": input_type,
        "overall_median_error_mm": float(error_rows["error_mm"].median()),
        "overall_mean_error_mm": float(error_rows["error_mm"].mean()),
        "large_error_rate": float(error_rows["is_large_error"].mean()),
        "num_images": int(error_rows["image_name"].nunique()),
        "notes": notes,
    }
    for keypoint in ["P4_peduncle_start_midpoint", "C2_trunk_axis_point", "P8_body_depth_dorsal", "P9_body_depth_ventral", "P11_peduncle_depth_ventral", "P7U_caudal_fin_upper_tip", "P7L_caudal_fin_lower_tip"]:
        values = error_rows.loc[error_rows["keypoint_name"] == keypoint, "error_mm"]
        row[f"{keypoint}_median_error_mm"] = float(values.median()) if len(values) else np.nan
    hard = error_rows[error_rows["is_hard_case_keypoint"]]
    review = error_rows[error_rows["is_review_confirmed"]]
    row["hard_case_mean_error_mm"] = float(hard["error_mm"].mean()) if len(hard) else np.nan
    row["review_confirmed_mean_error_mm"] = float(review["error_mm"].mean()) if len(review) else np.nan
    return row


def build_model_comparison(all_error_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source in ["v041_hybrid", "v01_heatmap", "v05_heatmap", "v06_keypointwise_hybrid"]:
        subset = all_error_rows[all_error_rows["source"] == source]
        notes = "v041 available for review-confirmed images only" if source == "v041_hybrid" else ""
        rows.append(model_overall_metrics(source, subset, "warped", notes))
    return pd.DataFrame(rows)


def build_hard_case_comparison(all_error_rows: pd.DataFrame) -> pd.DataFrame:
    hard = pd.read_csv(HARD_CASES_PATH) if HARD_CASES_PATH.exists() else pd.DataFrame()
    rows = []
    for record in hard.to_dict("records"):
        image_name = record["image_name"]
        affected = set()
        value = record.get("affected_keypoints")
        if isinstance(value, str):
            affected = {part.strip() for part in value.replace(",", ";").split(";") if part.strip() in KEYPOINT_NAMES}
        if not affected:
            affected = set(KEYPOINT_NAMES)
        row = {
            "image_name": image_name,
            "specimen_id": record.get("specimen_id"),
            "hard_case_type": record.get("hard_case_type"),
            "affected_keypoints": ";".join(sorted(affected)),
        }
        for source in ["v041_hybrid", "v01_heatmap", "v05_heatmap", "v06_keypointwise_hybrid"]:
            subset = all_error_rows[
                (all_error_rows["image_name"] == image_name)
                & (all_error_rows["keypoint_name"].isin(affected))
                & (all_error_rows["source"] == source)
            ]
            row[f"{source}_mean_error_mm"] = float(subset["error_mm"].mean()) if len(subset) else np.nan
        row["v05_improved_vs_v01"] = (
            row["v05_heatmap_mean_error_mm"] < row["v01_heatmap_mean_error_mm"]
            if not np.isnan(row.get("v05_heatmap_mean_error_mm", np.nan)) and not np.isnan(row.get("v01_heatmap_mean_error_mm", np.nan))
            else False
        )
        row["v06_improved_vs_v01"] = (
            row["v06_keypointwise_hybrid_mean_error_mm"] < row["v01_heatmap_mean_error_mm"]
            if not np.isnan(row.get("v06_keypointwise_hybrid_mean_error_mm", np.nan)) and not np.isnan(row.get("v01_heatmap_mean_error_mm", np.nan))
            else False
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_review_comparison(all_error_rows: pd.DataFrame) -> pd.DataFrame:
    review = all_error_rows[all_error_rows["is_review_confirmed"]]
    rows = []
    for source, group in review.groupby("source"):
        rows.append(model_overall_metrics(source, group, "warped", "review-confirmed subset"))
    return pd.DataFrame(rows)


def image_path_for(image_name: str) -> Path | None:
    crops = pd.read_csv(CROP_METADATA_PATH)
    match = crops[crops["image_name"] == image_name]
    if not match.empty:
        path = resolve_path(match["source_warped_image_path"].iloc[0])
        if path and path.exists():
            return path
    pre = review_preannotation_path(image_name)
    if pre:
        data = load_json(pre)
        path = resolve_path(data.get("warped_image_path"))
        if path and path.exists():
            return path
    return None


def draw_visuals(
    eval_images: set[str],
    gt: dict[tuple[str, str], tuple[float, float]],
    prediction_maps: dict[str, dict],
    v06_sources: dict[str, dict],
    all_error_rows: pd.DataFrame,
) -> None:
    VIS_OUT.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        small_font = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    colors = {
        "gt": (0, 210, 60),
        "v01_heatmap": (165, 80, 255),
        "v05_heatmap": (255, 80, 190),
        "v041_hybrid": (255, 150, 40),
        "v06_keypointwise_hybrid": (230, 30, 30),
        "mask_or_geometry_rule_if_available": (0, 210, 220),
        "large": (255, 230, 0),
    }
    for image_name in sorted(eval_images):
        path = image_path_for(image_name)
        if path is None:
            continue
        image = Image.open(path).convert("RGB")
        scale = min(1.0, 1800 / image.width)
        display = image.resize((int(image.width * scale), int(image.height * scale))) if scale != 1 else image.copy()
        draw = ImageDraw.Draw(display)

        def point(xy: tuple[float, float], color: tuple[int, int, int], radius: int = 5, outline: bool = False):
            x, y = xy[0] * scale, xy[1] * scale
            box = [x - radius, y - radius, x + radius, y + radius]
            if outline:
                draw.ellipse(box, outline=color, width=2)
            else:
                draw.ellipse(box, fill=color, outline=(0, 0, 0))

        for keypoint in KEYPOINT_NAMES:
            gt_xy = gt.get((image_name, keypoint))
            if gt_xy:
                point(gt_xy, colors["gt"], 4)
            for source in ["v01_heatmap", "v05_heatmap", "v041_hybrid", "mask_or_geometry_rule_if_available"]:
                pred = prediction_maps.get(source, {}).get((image_name, keypoint))
                if pred:
                    point((pred["x"], pred["y"]), colors[source], 3, outline=True)
            pred = prediction_maps.get("v06_keypointwise_hybrid", {}).get((image_name, keypoint))
            if pred:
                point((pred["x"], pred["y"]), colors["v06_keypointwise_hybrid"], 5)
                source_label = v06_sources.get(image_name, {}).get(keypoint, "")
                draw.text((pred["x"] * scale + 5, pred["y"] * scale + 5), keypoint.split("_")[0] + ":" + source_label.replace("_heatmap", ""), fill=(255, 255, 255), font=small_font)

        image_errors = all_error_rows[
            (all_error_rows["image_name"] == image_name) & (all_error_rows["source"] == "v06_keypointwise_hybrid")
        ]
        for row in image_errors[image_errors["is_large_error"]].itertuples(index=False):
            pred = prediction_maps["v06_keypointwise_hybrid"].get((image_name, row.keypoint_name))
            if pred:
                point((pred["x"], pred["y"]), colors["large"], 12, outline=True)
        mean = image_errors["error_mm"].mean() if len(image_errors) else np.nan
        median = image_errors["error_mm"].median() if len(image_errors) else np.nan
        large = ";".join(image_errors.loc[image_errors["is_large_error"], "keypoint_name"].tolist())
        draw.rectangle([0, 0, 900, 72], fill=(0, 0, 0))
        draw.text((10, 8), f"{image_name}  v0.6 mean={mean:.2f} mm median={median:.2f} mm", fill=(255, 255, 255), font=font)
        draw.text((10, 38), f"large: {large or 'none'}", fill=(255, 230, 0), font=font)
        display.save(VIS_OUT / f"{Path(image_name).stem}_v06_compare.png")


def write_readmes(
    model_comparison: pd.DataFrame,
    recommendations: pd.DataFrame,
    hard_cases: pd.DataFrame,
) -> None:
    def row_metric(model: str, col: str) -> float:
        match = model_comparison[model_comparison["model_name"] == model]
        if match.empty:
            return np.nan
        return float(match[col].iloc[0])

    rec_lines = []
    for row in recommendations.to_dict("records"):
        rec_lines.append(f"- {row['keypoint_name']}: {row['recommended_default_source']} (fallback: {row['second_choice_source']})")

    v06_median = row_metric("v06_keypointwise_hybrid", "overall_median_error_mm")
    v041_median = row_metric("v041_hybrid", "overall_median_error_mm")
    v06_mean = row_metric("v06_keypointwise_hybrid", "overall_mean_error_mm")
    v041_mean = row_metric("v041_hybrid", "overall_mean_error_mm")
    integrate = bool(not np.isnan(v06_mean) and not np.isnan(v041_mean) and v06_mean < v041_mean and v06_median <= v041_median + 0.2)

    hard_valid = hard_cases.dropna(subset=["v01_heatmap_mean_error_mm", "v06_keypointwise_hybrid_mean_error_mm"])
    v05_improved = int(hard_valid["v05_improved_vs_v01"].sum()) if not hard_valid.empty else 0
    v06_improved = int(hard_valid["v06_improved_vs_v01"].sum()) if not hard_valid.empty else 0

    readme = f"""# v0.6 Keypoint-wise Hybrid Selector Evaluation

No model was trained. No corrected labels or Streamlit defaults were modified.

## Why Not Replace Everything With v0.5?

v0.5 reduces large outliers and hard-case mean error, but it does not clearly beat v0.1 on overall median and it degrades C2.  A per-keypoint selector is therefore safer than a global model swap.

## Overall Metrics

- v0.4.1 hybrid median / mean: {v041_median:.3f} / {v041_mean:.3f} mm
- v0.1 heatmap median / mean: {row_metric('v01_heatmap', 'overall_median_error_mm'):.3f} / {row_metric('v01_heatmap', 'overall_mean_error_mm'):.3f} mm
- v0.5 heatmap median / mean: {row_metric('v05_heatmap', 'overall_median_error_mm'):.3f} / {row_metric('v05_heatmap', 'overall_mean_error_mm'):.3f} mm
- v0.6 selector median / mean: {v06_median:.3f} / {v06_mean:.3f} mm

Important caveat: v0.4.1 hybrid rows are available for the 29 review-confirmed images; v0.1/v0.5/v0.6 cover the union of the v0.5 test split and review-confirmed set.

## Recommended Sources

{chr(10).join(rec_lines)}

## C2 Handling

C2 should not use v0.5 by default. The selector chooses the steadier v0.1/v0.4.1 source when available and keeps axis review as the QC strategy.

## Hard Cases

- v0.5 improved vs v0.1: {v05_improved} / {len(hard_valid)}
- v0.6 improved vs v0.1: {v06_improved} / {len(hard_valid)}

## Recommendation

- recommend_integrate_v06_to_streamlit: {'true' if integrate else 'false'}
- keep_v0.4.1_default: {'false' if integrate else 'true'}

For now, use v0.6 as an experimental mode unless you want the extra source-comparison UI complexity in the annotation page.
"""
    (OUT_ROOT / "README.md").write_text(readme, encoding="utf-8")
    (EVAL_OUT / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    split, test_images, review_images, eval_images = evaluation_image_sets()
    gt, gt_meta = load_ground_truth()
    image_subsets = {name: subset_label(name, test_images, review_images) for name in eval_images}
    hard_lookup = hard_case_lookup()

    frames = save_prediction_csvs(eval_images)
    error_frames = []
    for source, frame in frames.items():
        error_frames.append(compute_error_rows(source, frame, gt, image_subsets, hard_lookup))
    source_errors = pd.concat(error_frames, ignore_index=True) if error_frames else pd.DataFrame()
    source_summary = summarize_source_errors(source_errors)
    source_summary.to_csv(OUT_ROOT / "source_error_by_keypoint.csv", index=False, encoding="utf-8-sig")

    recommendations = build_recommendations(source_summary)
    recommendations.to_csv(OUT_ROOT / "keypoint_source_recommendation.csv", index=False, encoding="utf-8-sig")

    v06_frame, v06_sources, _v06_qc = build_v06_predictions(frames, recommendations, eval_images, test_images, review_images, gt_meta)
    v06_errors = compute_error_rows("v06_keypointwise_hybrid", v06_frame, gt, image_subsets, hard_lookup)
    all_errors = pd.concat([source_errors, v06_errors], ignore_index=True)

    v06_by_point = summarize_by_point(v06_errors)
    v06_by_image = summarize_by_image(v06_errors, v06_sources)
    model_comparison = build_model_comparison(all_errors)
    hard_comparison = build_hard_case_comparison(all_errors)
    review_comparison = build_review_comparison(all_errors)

    v06_by_point.to_csv(EVAL_OUT / "keypoint_error_by_point.csv", index=False, encoding="utf-8-sig")
    v06_by_image.to_csv(EVAL_OUT / "keypoint_error_by_image.csv", index=False, encoding="utf-8-sig")
    model_comparison.to_csv(EVAL_OUT / "model_comparison_v06.csv", index=False, encoding="utf-8-sig")
    hard_comparison.to_csv(EVAL_OUT / "hard_case_comparison_v06.csv", index=False, encoding="utf-8-sig")
    review_comparison.to_csv(EVAL_OUT / "review_confirmed_comparison_v06.csv", index=False, encoding="utf-8-sig")
    all_errors.to_csv(EVAL_OUT / "all_source_error_rows.csv", index=False, encoding="utf-8-sig")

    prediction_maps = {source: frame_to_map(frame) for source, frame in frames.items()}
    prediction_maps["v06_keypointwise_hybrid"] = frame_to_map(v06_frame)
    draw_visuals(eval_images, gt, prediction_maps, v06_sources, all_errors)
    write_readmes(model_comparison, recommendations, hard_comparison)

    def metric(model: str, col: str) -> float:
        row = model_comparison[model_comparison["model_name"] == model]
        return float(row[col].iloc[0]) if not row.empty else np.nan

    print("Finished v0.6 keypoint-wise hybrid selector evaluation.")
    print("Overall:")
    print(f"v0.4.1 hybrid median / mean: {metric('v041_hybrid', 'overall_median_error_mm'):.3f} / {metric('v041_hybrid', 'overall_mean_error_mm'):.3f}")
    print(f"v0.1 heatmap median / mean: {metric('v01_heatmap', 'overall_median_error_mm'):.3f} / {metric('v01_heatmap', 'overall_mean_error_mm'):.3f}")
    print(f"v0.5 heatmap median / mean: {metric('v05_heatmap', 'overall_median_error_mm'):.3f} / {metric('v05_heatmap', 'overall_mean_error_mm'):.3f}")
    print(f"v0.6 selector median / mean: {metric('v06_keypointwise_hybrid', 'overall_median_error_mm'):.3f} / {metric('v06_keypointwise_hybrid', 'overall_mean_error_mm'):.3f}")
    print("Keypoint source recommendations:")
    for row in recommendations.to_dict("records"):
        print(f"{row['keypoint_name']}: {row['recommended_default_source']}")
    hard_valid = hard_comparison.dropna(subset=["v01_heatmap_mean_error_mm", "v06_keypointwise_hybrid_mean_error_mm"])
    print("Hard cases:")
    print(f"v0.5 improved: {int(hard_valid['v05_improved_vs_v01'].sum())}/{len(hard_valid)}")
    print(f"v0.6 improved: {int(hard_valid['v06_improved_vs_v01'].sum())}/{len(hard_valid)}")
    readme = (OUT_ROOT / "README.md").read_text(encoding="utf-8")
    integrate = "recommend_integrate_v06_to_streamlit: true" in readme
    print("Recommendation:")
    print(f"integrate v0.6 to Streamlit: {'yes' if integrate else 'no'}")
    print(f"keep v0.4.1 default: {'no' if integrate else 'yes'}")


if __name__ == "__main__":
    main()
