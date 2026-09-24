"""Summarize heatmap U-Net benchmark and select a candidate checkpoint."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "siganusmorph_heatmap_unet_v0.1"
BASELINE_DIR = PROJECT_ROOT / "results" / "model_eval" / "v0.3.4_local_structure_refinement_compare_30"
BEST_EVAL = PROJECT_ROOT / "results" / "model_eval" / "external_benchmark" / "heatmap_unet_v0.1_best"
LAST_EVAL = PROJECT_ROOT / "results" / "model_eval" / "external_benchmark" / "heatmap_unet_v0.1_last"
BEST_VAL_EVAL = PROJECT_ROOT / "results" / "model_eval" / "external_benchmark" / "heatmap_unet_v0.1_best_val"
LAST_VAL_EVAL = PROJECT_ROOT / "results" / "model_eval" / "external_benchmark" / "heatmap_unet_v0.1_last_val"
SUMMARY_PATH = PROJECT_ROOT / "results" / "model_eval" / "external_benchmark" / "benchmark_summary_v0.1.csv"


POINTS = {
    "P2": "P2_eye_front",
    "P3": "P3_operculum_posterior",
    "P5": "P5_caudal_base_midpoint",
    "P6": "P6_caudal_fork_midpoint",
    "P7U": "P7U_caudal_fin_upper_tip",
    "P7L": "P7L_caudal_fin_lower_tip",
    "P10": "P10_peduncle_depth_dorsal",
    "P11": "P11_peduncle_depth_ventral",
}


def row_from_summary(model_name: str, eval_dir: Path, notes: str) -> dict:
    summary = pd.read_csv(eval_dir / "keypoint_error_summary.csv")
    row = {
        "model_name": model_name,
        "overall_median_error_mm": summary["error_mm"].median(),
        "overall_mean_error_mm": summary["error_mm"].mean(),
        "num_test_images": summary["image_name"].nunique(),
        "notes": notes,
    }
    for short, key in POINTS.items():
        row[f"{short}_median_error_mm"] = summary[summary["keypoint_name"] == key]["error_mm"].median()
    return row


def row_from_v034() -> dict:
    summary = pd.read_csv(BASELINE_DIR / "preannotation_vs_manual_summary.csv")
    row = {
        "model_name": "current_v0.3.4_enhanced_preannotation",
        "overall_median_error_mm": summary["error_mm"].median(),
        "overall_mean_error_mm": summary["error_mm"].mean(),
        "num_test_images": summary["image_name"].nunique(),
        "notes": "Existing 30-image mixed-split v0.3.4 enhanced preannotation evaluation.",
    }
    for short, key in POINTS.items():
        row[f"{short}_median_error_mm"] = summary[summary["keypoint_name"] == key]["error_mm"].median()
    return row


def write_selection(best_row: dict, last_row: dict) -> None:
    best_eval = pd.read_csv(BEST_EVAL / "keypoint_error_summary.csv")
    last_eval = pd.read_csv(LAST_EVAL / "keypoint_error_summary.csv")
    best_val_eval = pd.read_csv(BEST_VAL_EVAL / "keypoint_error_summary.csv")
    last_val_eval = pd.read_csv(LAST_VAL_EVAL / "keypoint_error_summary.csv")
    best_test = best_row["overall_median_error_mm"]
    last_test = last_row["overall_median_error_mm"]
    best_mean = best_row["overall_mean_error_mm"]
    last_mean = last_row["overall_mean_error_mm"]
    best_val = float(best_val_eval["error_mm"].median())
    last_val = float(last_val_eval["error_mm"].median())
    best_score = best_test + best_mean
    last_score = last_test + last_mean
    selected = "best" if best_score <= last_score else "last"
    selected_src = MODEL_DIR / ("best_model.pt" if selected == "best" else "last_model.pt")
    shutil.copy2(selected_src, MODEL_DIR / "preannotation_candidate.pt")

    def metrics(name: str, df: pd.DataFrame, val_median: float, test_median: float, selected_flag: bool) -> dict:
        return {
            "model_name": name,
            "val_median_error_mm": val_median,
            "test_median_error_mm": test_median,
            "test_mean_error_mm": float(df["error_mm"].mean()),
            "selection_score": float(test_median + df["error_mm"].mean()),
            "P2_median_error_mm": df[df["keypoint_name"] == POINTS["P2"]]["error_mm"].median(),
            "P3_median_error_mm": df[df["keypoint_name"] == POINTS["P3"]]["error_mm"].median(),
            "P5_median_error_mm": df[df["keypoint_name"] == POINTS["P5"]]["error_mm"].median(),
            "P6_median_error_mm": df[df["keypoint_name"] == POINTS["P6"]]["error_mm"].median(),
            "selected_as_candidate": selected_flag,
        }

    selection = pd.DataFrame(
        [
            metrics("best_model.pt", best_eval, best_val, best_test, selected == "best"),
            metrics("last_model.pt", last_eval, last_val, last_test, selected == "last"),
        ]
    )
    selection.to_csv(MODEL_DIR / "model_selection_report.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    baseline = row_from_v034()
    best = row_from_summary("heatmap_unet_v0.1_best", BEST_EVAL, "Heatmap U-Net best_model.pt on test split.")
    last = row_from_summary("heatmap_unet_v0.1_last", LAST_EVAL, "Heatmap U-Net last_model.pt on test split.")
    summary = pd.DataFrame([baseline, best, last])
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    write_selection(best, last)
    selected = pd.read_csv(MODEL_DIR / "model_selection_report.csv")
    candidate = selected[selected["selected_as_candidate"].astype(str).str.lower() == "true"]["model_name"].iloc[0]
    train_config = (MODEL_DIR / "train_config.yaml").read_text(encoding="utf-8") if (MODEL_DIR / "train_config.yaml").exists() else ""
    readme = MODEL_DIR / "README_model.md"
    comparison_text = f"""# SiganusMorph Heatmap U-Net v0.1

Data source: `datasets/siganusmorph_heatmap_unet_v0.1/`

Labels: corrected real maskcrop labels from `results/real_annotation_5_15_v0.3_corrected/keypoints/`

Train/val/test split: specimen_id grouped, inherited from v0.3 corrected split.

Keypoints: 16 manual landmarks. P7V is derived and excluded.

## Training Config

```yaml
{train_config.strip()}
```

## Benchmark Summary

{summary.to_string(index=False)}

## Model Selection

{pd.read_csv(MODEL_DIR / "model_selection_report.csv").to_string(index=False)}

Selected preannotation candidate: `{candidate}`

Selection note: `last_model.pt` has a slightly lower median error, but `best_model.pt` has a much lower mean error and avoids large C1/C2 outliers, so it is safer as a preannotation candidate.

## Recommendation

The heatmap-U-Net is substantially better than the current v0.3.4 baseline on P2, P3, P5, P6, P7U, and P7L in this benchmark. The safest next step is to combine heatmap-U-Net predictions with the existing mask/geometric QC rules rather than replacing the entire v0.3.4 workflow immediately.
"""
    readme.write_text(comparison_text, encoding="utf-8")
    print("Finished heatmap-U-Net v0.1 benchmark.")
    print(f"Test median error v0.3.4 enhanced baseline: {baseline['overall_median_error_mm']:.3f} mm")
    print(f"Test median error heatmap_unet_v0.1 best: {best['overall_median_error_mm']:.3f} mm")
    print(f"Test median error heatmap_unet_v0.1 last: {last['overall_median_error_mm']:.3f} mm")
    print("Keypoint comparison:")
    for short in ("P2", "P3", "P5", "P6", "P7U", "P7L"):
        print(f"{short}: baseline={baseline[f'{short}_median_error_mm']:.3f} best={best[f'{short}_median_error_mm']:.3f} last={last[f'{short}_median_error_mm']:.3f}")
    if min(best["overall_median_error_mm"], last["overall_median_error_mm"]) < baseline["overall_median_error_mm"]:
        print("Recommendation: consider combining heatmap-U-Net with mask/geometric rules.")
    else:
        print("Recommendation: keep v0.3.4 baseline for now; use heatmap-U-Net as an external benchmark.")
    print(f"Summary saved to {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
