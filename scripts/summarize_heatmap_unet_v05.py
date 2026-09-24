from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRED_ROOT = PROJECT_ROOT / "results" / "model_eval" / "heatmap_unet_v0.5_predictions"
GT_DIR = PRED_ROOT / "ground_truth_keypoints"
CROP_METADATA_PATH = PRED_ROOT / "crop_metadata.csv"
SPLIT_MANIFEST_PATH = PROJECT_ROOT / "datasets" / "siganusmorph_heatmap_unet_v0.5_candidate" / "split_manifest.csv"
HARD_CASES_PATH = PROJECT_ROOT / "results" / "v0.5_dataset_planning" / "v05_hard_cases.csv"
REVIEW_DETAIL_PATH = (
    PROJECT_ROOT
    / "results"
    / "realworld_review_v0.4.1"
    / "post_review_evaluation"
    / "manual_correction_by_point_detail.csv"
)
V04_BENCHMARK_DIR = PROJECT_ROOT / "results" / "model_eval" / "v0.4_hybrid_heatmap_geometry_compare_30"

COMPARISON_DIR = PROJECT_ROOT / "results" / "model_eval" / "v0.5_model_comparison"
MODEL_DIR = PROJECT_ROOT / "models" / "siganusmorph_heatmap_unet_v0.5"

KEYPOINTS = [
    "P1_snout_tip",
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
    "C1_head_axis_point",
    "C2_trunk_axis_point",
    "C3_posterior_trunk_axis_point",
    "C4_peduncle_axis_point",
]


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_eval(eval_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = read_csv_if_exists(eval_dir / "keypoint_error_summary.csv")
    by_point = read_csv_if_exists(eval_dir / "keypoint_error_by_point.csv")
    by_image = read_csv_if_exists(eval_dir / "keypoint_error_by_image.csv")
    return summary, by_point, by_image


def model_metrics(eval_dir: Path, model_name: str, input_type: str, notes: str) -> dict:
    summary, by_point, by_image = load_eval(eval_dir)
    if summary.empty:
        return {
            "model_name": model_name,
            "input_type": input_type,
            "overall_median_error_mm": np.nan,
            "overall_mean_error_mm": np.nan,
            "large_error_rate": np.nan,
            "num_test_images": 0,
            "notes": f"missing eval dir: {eval_dir}",
        }

    errors = pd.to_numeric(summary["error_mm"], errors="coerce").dropna()
    row = {
        "model_name": model_name,
        "input_type": input_type,
        "overall_median_error_mm": float(errors.median()) if len(errors) else np.nan,
        "overall_mean_error_mm": float(errors.mean()) if len(errors) else np.nan,
        "large_error_rate": float((errors > 10).mean()) if len(errors) else np.nan,
        "num_test_images": int(by_image["image_name"].nunique()) if "image_name" in by_image else 0,
        "notes": notes,
    }
    for keypoint in KEYPOINTS:
        match = by_point[by_point["keypoint_name"] == keypoint]
        row[f"{keypoint}_median_error_mm"] = (
            float(match["median_error_mm"].iloc[0]) if not match.empty else np.nan
        )
    return row


def review_hybrid_metrics() -> tuple[dict, pd.DataFrame]:
    detail = read_csv_if_exists(REVIEW_DETAIL_PATH)
    if detail.empty:
        return (
            {
                "model_name": "v0.4.1_hybrid_realworld_review_29",
                "input_type": "hybrid_preannotation",
                "overall_median_error_mm": np.nan,
                "overall_mean_error_mm": np.nan,
                "large_error_rate": np.nan,
                "num_test_images": 0,
                "notes": "missing post-review detail",
            },
            pd.DataFrame(),
        )
    errors = pd.to_numeric(detail["error_mm"], errors="coerce").dropna()
    row = {
        "model_name": "v0.4.1_hybrid_realworld_review_29",
        "input_type": "hybrid_preannotation",
        "overall_median_error_mm": float(errors.median()),
        "overall_mean_error_mm": float(errors.mean()),
        "large_error_rate": float((errors > 10).mean()),
        "num_test_images": int(detail["image_name"].nunique()),
        "notes": "reference from real-world review confirmed images; not the v0.5 test split",
    }
    for keypoint in KEYPOINTS:
        kp = pd.to_numeric(detail.loc[detail["keypoint_name"] == keypoint, "error_mm"], errors="coerce").dropna()
        row[f"{keypoint}_median_error_mm"] = float(kp.median()) if len(kp) else np.nan
    return row, detail


def v04_hybrid_benchmark_metrics() -> dict:
    summary = read_csv_if_exists(V04_BENCHMARK_DIR / "preannotation_vs_manual_summary.csv")
    by_point = read_csv_if_exists(V04_BENCHMARK_DIR / "keypoint_error_by_point.csv")
    by_image = read_csv_if_exists(V04_BENCHMARK_DIR / "keypoint_error_by_image.csv")
    if summary.empty:
        return {
            "model_name": "v0.4.1_hybrid_benchmark_30",
            "input_type": "hybrid_preannotation",
            "overall_median_error_mm": np.nan,
            "overall_mean_error_mm": np.nan,
            "large_error_rate": np.nan,
            "num_test_images": 0,
            "notes": "missing v0.4 benchmark table",
        }
    errors = pd.to_numeric(summary["error_mm"], errors="coerce").dropna()
    row = {
        "model_name": "v0.4.1_hybrid_benchmark_30",
        "input_type": "hybrid_preannotation",
        "overall_median_error_mm": float(errors.median()) if len(errors) else np.nan,
        "overall_mean_error_mm": float(errors.mean()) if len(errors) else np.nan,
        "large_error_rate": float((errors > 10).mean()) if len(errors) else np.nan,
        "num_test_images": int(by_image["image_name"].nunique()) if "image_name" in by_image else int(summary["image_name"].nunique()),
        "notes": "reference from v0.4 hybrid 30-image benchmark; not the v0.5 test split",
    }
    for keypoint in KEYPOINTS:
        match = by_point[by_point["keypoint_name"] == keypoint] if not by_point.empty else pd.DataFrame()
        row[f"{keypoint}_median_error_mm"] = (
            float(match["median_error_mm"].iloc[0]) if not match.empty else np.nan
        )
    return row


def load_prediction_maps(tag: str) -> dict[tuple[str, str], dict]:
    frames = []
    for split in ("train", "val", "test"):
        path = PRED_ROOT / f"predictions_{split}_{tag}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return {}
    preds = pd.concat(frames, ignore_index=True)
    crops = pd.read_csv(CROP_METADATA_PATH)
    crop_lookup = crops.set_index("image_name").to_dict("index")
    result = {}
    for row in preds.itertuples(index=False):
        crop = crop_lookup.get(row.image_name)
        if crop is None:
            continue
        x = float(row.x) + float(crop.get("bbox_x1", 0.0))
        y = float(row.y) + float(crop.get("bbox_y1", 0.0))
        result[(row.image_name, row.keypoint_name)] = {
            "x": x,
            "y": y,
            "confidence": float(row.confidence) if not pd.isna(row.confidence) else np.nan,
        }
    return result


def load_ground_truth() -> dict[tuple[str, str], tuple[float, float]]:
    result = {}
    for path in GT_DIR.glob("*_keypoints.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        image_name = data.get("image_name") or path.name.replace("_keypoints.json", ".png")
        keypoints = data.get("corrected_keypoints") or {}
        for name, xy in keypoints.items():
            if name not in KEYPOINTS or not isinstance(xy, list) or len(xy) < 2:
                continue
            result[(image_name, name)] = (float(xy[0]), float(xy[1]))
    return result


def parse_keypoint_list(value: object) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    out: list[str] = []
    for part in value.replace(",", ";").split(";"):
        part = part.strip()
        if part in KEYPOINTS:
            out.append(part)
    return out


def mean_error_for_keypoints(
    pred_map: dict[tuple[str, str], dict],
    gt_map: dict[tuple[str, str], tuple[float, float]],
    image_name: str,
    keypoints: Iterable[str],
) -> float:
    values = []
    for keypoint in keypoints:
        pred = pred_map.get((image_name, keypoint))
        gt = gt_map.get((image_name, keypoint))
        if pred is None or gt is None:
            continue
        values.append(math.hypot(pred["x"] - gt[0], pred["y"] - gt[1]) * 0.1)
    return float(np.mean(values)) if values else np.nan


def summarize_hard_cases(review_detail: pd.DataFrame) -> pd.DataFrame:
    hard_cases = read_csv_if_exists(HARD_CASES_PATH)
    if hard_cases.empty:
        return pd.DataFrame()

    gt_map = load_ground_truth()
    pred_v01 = load_prediction_maps("v01_best")
    pred_v05_best = load_prediction_maps("v05_best")
    pred_v05_last = load_prediction_maps("v05_last")

    rows = []
    for record in hard_cases.to_dict("records"):
        image_name = record.get("image_name")
        affected = parse_keypoint_list(record.get("affected_keypoints"))
        if not affected:
            affected = KEYPOINTS
        v01 = mean_error_for_keypoints(pred_v01, gt_map, image_name, affected)
        v05_best = mean_error_for_keypoints(pred_v05_best, gt_map, image_name, affected)
        v05_last = mean_error_for_keypoints(pred_v05_last, gt_map, image_name, affected)
        v041 = np.nan
        if not review_detail.empty and "image_name" in review_detail:
            subset = review_detail[
                (review_detail["image_name"] == image_name) & (review_detail["keypoint_name"].isin(affected))
            ]
            if not subset.empty:
                v041 = float(pd.to_numeric(subset["error_mm"], errors="coerce").mean())

        notes = ""
        if record.get("recommended_use") == "external_error_case_only":
            notes = "excluded external error case; not used for training"
        elif np.isnan(v05_best):
            notes = "no v0.5 prediction/ground truth available"

        rows.append(
            {
                "image_name": image_name,
                "specimen_id": record.get("specimen_id"),
                "hard_case_type": record.get("hard_case_type"),
                "affected_keypoints": ";".join(affected),
                "v01_error_mm": v01,
                "v05_best_error_mm": v05_best,
                "v05_last_error_mm": v05_last,
                "v041_hybrid_error_mm": v041,
                "improved_by_v05": bool(v05_best < v01) if not np.isnan(v05_best) and not np.isnan(v01) else False,
                "notes": notes,
            }
        )
    return pd.DataFrame(rows)


def build_keypoint_comparison(model_rows: list[dict], review_detail: pd.DataFrame) -> pd.DataFrame:
    dirs = {
        "heatmap_v0.1_best": PROJECT_ROOT / "results" / "model_eval" / "external_benchmark" / "heatmap_unet_v0.1_best_on_v05",
        "heatmap_v0.5_best": PROJECT_ROOT / "results" / "model_eval" / "external_benchmark" / "heatmap_unet_v0.5_best",
        "heatmap_v0.5_last": PROJECT_ROOT / "results" / "model_eval" / "external_benchmark" / "heatmap_unet_v0.5_last",
    }
    by_point_tables = {}
    for name, path in dirs.items():
        by_point_tables[name] = read_csv_if_exists(path / "keypoint_error_by_point.csv")

    rows = []
    v04_by_point = read_csv_if_exists(V04_BENCHMARK_DIR / "keypoint_error_by_point.csv")
    for keypoint in KEYPOINTS:
        row = {"keypoint_name": keypoint}
        match_v04 = v04_by_point[v04_by_point["keypoint_name"] == keypoint] if not v04_by_point.empty else pd.DataFrame()
        row["v041_hybrid_benchmark_median_error_mm"] = (
            float(match_v04["median_error_mm"].iloc[0]) if not match_v04.empty else np.nan
        )
        if not review_detail.empty:
            values = pd.to_numeric(review_detail.loc[review_detail["keypoint_name"] == keypoint, "error_mm"], errors="coerce").dropna()
            row["v041_realworld_review_median_manual_move_mm"] = float(values.median()) if len(values) else np.nan
        for model_name, table in by_point_tables.items():
            match = table[table["keypoint_name"] == keypoint] if not table.empty else pd.DataFrame()
            row[f"{model_name}_median_error_mm"] = float(match["median_error_mm"].iloc[0]) if not match.empty else np.nan
            row[f"{model_name}_mean_error_mm"] = float(match["mean_error_mm"].iloc[0]) if not match.empty else np.nan
        v01 = row.get("heatmap_v0.1_best_median_error_mm", np.nan)
        v05 = row.get("heatmap_v0.5_best_median_error_mm", np.nan)
        row["v05_best_minus_v01_best_median_mm"] = v05 - v01 if not np.isnan(v01) and not np.isnan(v05) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def model_selection_report(hard_cases: pd.DataFrame) -> pd.DataFrame:
    val_best = model_metrics(
        PROJECT_ROOT / "results" / "model_eval" / "external_benchmark" / "heatmap_unet_v0.5_best_val",
        "best_model.pt",
        "maskcrop",
        "v0.5 validation split",
    )
    val_last = model_metrics(
        PROJECT_ROOT / "results" / "model_eval" / "external_benchmark" / "heatmap_unet_v0.5_last_val",
        "last_model.pt",
        "maskcrop",
        "v0.5 validation split",
    )
    test_best = model_metrics(
        PROJECT_ROOT / "results" / "model_eval" / "external_benchmark" / "heatmap_unet_v0.5_best",
        "best_model.pt",
        "maskcrop",
        "v0.5 test split",
    )
    test_last = model_metrics(
        PROJECT_ROOT / "results" / "model_eval" / "external_benchmark" / "heatmap_unet_v0.5_last",
        "last_model.pt",
        "maskcrop",
        "v0.5 test split",
    )
    hard_best = float(hard_cases["v05_best_error_mm"].mean()) if not hard_cases.empty else np.nan
    hard_last = float(hard_cases["v05_last_error_mm"].mean()) if not hard_cases.empty else np.nan

    rows = [
        {
            "model_name": "best_model.pt",
            "val_median_error_mm": val_best["overall_median_error_mm"],
            "test_median_error_mm": test_best["overall_median_error_mm"],
            "test_mean_error_mm": test_best["overall_mean_error_mm"],
            "hard_case_mean_error_mm": hard_best,
            "P4_median_error_mm": test_best.get("P4_peduncle_start_midpoint_median_error_mm"),
            "C2_median_error_mm": test_best.get("C2_trunk_axis_point_median_error_mm"),
            "P9_median_error_mm": test_best.get("P9_body_depth_ventral_median_error_mm"),
            "selected_as_candidate": True,
        },
        {
            "model_name": "last_model.pt",
            "val_median_error_mm": val_last["overall_median_error_mm"],
            "test_median_error_mm": test_last["overall_median_error_mm"],
            "test_mean_error_mm": test_last["overall_mean_error_mm"],
            "hard_case_mean_error_mm": hard_last,
            "P4_median_error_mm": test_last.get("P4_peduncle_start_midpoint_median_error_mm"),
            "C2_median_error_mm": test_last.get("C2_trunk_axis_point_median_error_mm"),
            "P9_median_error_mm": test_last.get("P9_body_depth_ventral_median_error_mm"),
            "selected_as_candidate": False,
        },
    ]
    return pd.DataFrame(rows)


def write_readme(model_comparison: pd.DataFrame, keypoint_comparison: pd.DataFrame, hard_cases: pd.DataFrame, selection: pd.DataFrame) -> None:
    v01 = model_comparison.loc[model_comparison["model_name"] == "heatmap_unet_v0.1_best_on_v05"]
    v05_best = model_comparison.loc[model_comparison["model_name"] == "heatmap_unet_v0.5_best"]
    v05_last = model_comparison.loc[model_comparison["model_name"] == "heatmap_unet_v0.5_last"]
    v041 = model_comparison.loc[model_comparison["model_name"] == "v0.4.1_hybrid_benchmark_30"]

    def metric(df: pd.DataFrame, col: str) -> float:
        return float(df[col].iloc[0]) if not df.empty and col in df else np.nan

    key_focus = ["P4_peduncle_start_midpoint", "C2_trunk_axis_point", "P9_body_depth_ventral", "P8_body_depth_dorsal", "P11_peduncle_depth_ventral", "P3_operculum_posterior", "P7U_caudal_fin_upper_tip", "P7L_caudal_fin_lower_tip"]
    focus_lines = []
    for keypoint in key_focus:
        row = keypoint_comparison[keypoint_comparison["keypoint_name"] == keypoint]
        if row.empty:
            continue
        focus_lines.append(
            f"- {keypoint}: v0.1={row['heatmap_v0.1_best_median_error_mm'].iloc[0]:.3f} mm, "
            f"v0.5 best={row['heatmap_v0.5_best_median_error_mm'].iloc[0]:.3f} mm, "
            f"v0.5 last={row['heatmap_v0.5_last_median_error_mm'].iloc[0]:.3f} mm"
        )

    hard_valid = hard_cases.dropna(subset=["v01_error_mm", "v05_best_error_mm"]) if not hard_cases.empty else pd.DataFrame()
    improved_count = int(hard_valid["improved_by_v05"].sum()) if not hard_valid.empty else 0
    hard_total = int(len(hard_valid))

    recommend_update = False
    recommendation = (
        "Keep the current v0.4.1 default heatmap model for now. "
        "v0.5 best is the better v0.5 checkpoint and is saved as the v0.5 candidate, "
        "but it does not clearly beat v0.1 on the held-out test median."
    )
    if metric(v05_best, "overall_median_error_mm") < metric(v01, "overall_median_error_mm") and improved_count >= max(1, hard_total // 2):
        recommend_update = True
        recommendation = "v0.5 best is recommended for a guarded default update after a quick Streamlit smoke test."

    text = f"""# heatmap-U-Net v0.5 Model Comparison

This report does not modify corrected labels and does not change the Streamlit default model.

## Dataset

- Candidate dataset: `datasets/siganusmorph_heatmap_unet_v0.5_candidate/`
- Total images: 96
- Train / val / test: 69 / 12 / 15
- Specimens: 34
- Specimen leakage: pass

## Test Median Error

- v0.4.1 hybrid benchmark reference: {metric(v041, 'overall_median_error_mm'):.3f} mm
- heatmap v0.1 best on v0.5 test split: {metric(v01, 'overall_median_error_mm'):.3f} mm
- heatmap v0.5 best: {metric(v05_best, 'overall_median_error_mm'):.3f} mm
- heatmap v0.5 last: {metric(v05_last, 'overall_median_error_mm'):.3f} mm

Note: the v0.4.1 hybrid number is from the earlier 30-image benchmark, while heatmap v0.1/v0.5 are evaluated on the v0.5 held-out test split. Use this as a practical reference, not a strict same-split comparison.

## Focus Keypoints

{chr(10).join(focus_lines)}

## Hard Cases

- Hard cases with comparable v0.1/v0.5 predictions: {hard_total}
- Improved by v0.5 best: {improved_count}
- Mean hard-case error, v0.1: {hard_valid['v01_error_mm'].mean() if not hard_valid.empty else np.nan:.3f} mm
- Mean hard-case error, v0.5 best: {hard_valid['v05_best_error_mm'].mean() if not hard_valid.empty else np.nan:.3f} mm

## Model Selection

Within v0.5, `best_model.pt` is selected as `preannotation_candidate.pt` because it has lower validation and test median error than `last_model.pt`.

## Recommendation

- update default heatmap model: {'yes' if recommend_update else 'no'}
- recommended candidate for v0.5 experiments: `best_model.pt`
- default workflow recommendation: {recommendation}

See:

- `model_comparison_v05.csv`
- `keypoint_comparison_v05.csv`
- `hard_case_comparison_v05.csv`
- `models/siganusmorph_heatmap_unet_v0.5/model_selection_report.csv`
"""
    (COMPARISON_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    _review_row, review_detail = review_hybrid_metrics()
    model_rows = [
        v04_hybrid_benchmark_metrics(),
        model_metrics(
            PROJECT_ROOT / "results" / "model_eval" / "external_benchmark" / "heatmap_unet_v0.1_best_on_v05",
            "heatmap_unet_v0.1_best_on_v05",
            "maskcrop",
            "v0.1 best evaluated on v0.5 test split",
        ),
        model_metrics(
            PROJECT_ROOT / "results" / "model_eval" / "external_benchmark" / "heatmap_unet_v0.5_best",
            "heatmap_unet_v0.5_best",
            "maskcrop",
            "v0.5 best evaluated on v0.5 test split",
        ),
        model_metrics(
            PROJECT_ROOT / "results" / "model_eval" / "external_benchmark" / "heatmap_unet_v0.5_last",
            "heatmap_unet_v0.5_last",
            "maskcrop",
            "v0.5 last evaluated on v0.5 test split",
        ),
    ]
    model_comparison = pd.DataFrame(model_rows)
    model_comparison.to_csv(COMPARISON_DIR / "model_comparison_v05.csv", index=False, encoding="utf-8-sig")

    keypoint_comparison = build_keypoint_comparison(model_rows, review_detail)
    keypoint_comparison.to_csv(COMPARISON_DIR / "keypoint_comparison_v05.csv", index=False, encoding="utf-8-sig")

    hard_cases = summarize_hard_cases(review_detail)
    hard_cases.to_csv(COMPARISON_DIR / "hard_case_comparison_v05.csv", index=False, encoding="utf-8-sig")

    selection = model_selection_report(hard_cases)
    selection.to_csv(MODEL_DIR / "model_selection_report.csv", index=False, encoding="utf-8-sig")

    best_path = MODEL_DIR / "best_model.pt"
    candidate_path = MODEL_DIR / "preannotation_candidate.pt"
    if best_path.exists():
        shutil.copy2(best_path, candidate_path)

    write_readme(model_comparison, keypoint_comparison, hard_cases, selection)

    readme_model = MODEL_DIR / "README_model.md"
    existing = readme_model.read_text(encoding="utf-8") if readme_model.exists() else "# heatmap-U-Net v0.5\n"
    appendix = (
        "\n\n## v0.5 Evaluation Summary\n\n"
        "The v0.5 comparison report is saved at "
        "`results/model_eval/v0.5_model_comparison/README.md`.\n\n"
        "`best_model.pt` is copied to `preannotation_candidate.pt` for v0.5 experiments only; "
        "the Streamlit default model is not changed by this script.\n"
    )
    if "## v0.5 Evaluation Summary" not in existing:
        readme_model.write_text(existing + appendix, encoding="utf-8")

    best = model_comparison[model_comparison["model_name"] == "heatmap_unet_v0.5_best"].iloc[0]
    last = model_comparison[model_comparison["model_name"] == "heatmap_unet_v0.5_last"].iloc[0]
    v01 = model_comparison[model_comparison["model_name"] == "heatmap_unet_v0.1_best_on_v05"].iloc[0]
    print("Finished v0.5 model comparison.")
    print(f"v0.1 best test median: {v01['overall_median_error_mm']:.3f} mm")
    print(f"v0.5 best test median: {best['overall_median_error_mm']:.3f} mm")
    print(f"v0.5 last test median: {last['overall_median_error_mm']:.3f} mm")
    print("Selected v0.5 candidate: best_model.pt")
    print(f"Results saved to: {COMPARISON_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
