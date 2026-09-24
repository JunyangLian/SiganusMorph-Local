"""Run v0.3.4 local-structure comparison on the same 30-image sample."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OLD_DIR = PROJECT_ROOT / "results" / "model_eval" / "v0.3.3_geometry_refinement_compare_30"
OUTPUT_DIR = PROJECT_ROOT / "results" / "model_eval" / "v0.3.4_local_structure_refinement_compare_30"

os.environ["SIGANUS_COMPARE_LABEL"] = "v0.3.4_local_structure_refinement"
os.environ["SIGANUS_COMPARE_OUTPUT_DIR"] = str(OUTPUT_DIR)
os.environ["SIGANUS_COMPARE_SAMPLE_MANIFEST"] = str(OLD_DIR / "sample_manifest.csv")

from evaluate_v031_preannotation_compare import main as compare_main


def _load_debug(image_name: str) -> dict:
    path = OLD_DIR / "debug_json" / f"{Path(image_name).stem}_debug.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _p6_failure_type(row: pd.Series, debug: dict) -> tuple[str, str, bool]:
    metadata = debug.get("preannotation_metadata", {}) if isinstance(debug.get("preannotation_metadata", {}), dict) else {}
    tail_qc = metadata.get("tail_geometry_qc", {}) if isinstance(metadata.get("tail_geometry_qc", {}), dict) else {}
    auto = debug.get("auto_preannotation_keypoints", {}) if isinstance(debug.get("auto_preannotation_keypoints", {}), dict) else {}
    manual = debug.get("manual_corrected_keypoints", {}) if isinstance(debug.get("manual_corrected_keypoints", {}), dict) else {}
    p6_auto = auto.get("P6_caudal_fork_midpoint") or [None, None]
    p6_manual = manual.get("P6_caudal_fork_midpoint") or [None, None]
    if str(tail_qc.get("tail_mask_gap_visible", "")).lower() == "false":
        return "fork_gap_not_visible", "Keep P6 as manual-review priority when the tail gap is not visible.", False
    if str(tail_qc.get("tail_mask_gap_filled", "")).lower() == "true":
        return "tail_gap_still_filled", "Use the tail_detail_mask preview to tune blue-gap extraction before changing P6.", True
    try:
        dx = float(p6_auto[0]) - float(p6_manual[0])
        dy = float(p6_auto[1]) - float(p6_manual[1])
    except Exception:
        return "biologically_ambiguous_fork", "Inspect manual label and tail-detail preview.", False
    if abs(dy) > abs(dx) * 1.35:
        return "wrong_gap_center_selected", "Use the gap axial depth but recenter P6 between P7U/P7L along the normal axis.", True
    if dx < -35:
        return "gap_candidate_on_edge", "Raise the minimum projection threshold for P6 gap candidates.", True
    return "biologically_ambiguous_fork", "Keep P6 highlighted for user confirmation.", False


def write_p6_remaining_review() -> None:
    old_summary = pd.read_csv(OLD_DIR / "preannotation_vs_manual_summary.csv")
    p6_rows = old_summary[
        (old_summary["keypoint_name"] == "P6_caudal_fork_midpoint")
        & ((old_summary["is_large_error"].astype(str).str.lower() == "true") | (old_summary["error_mm"] > 10))
    ].copy()
    if p6_rows.empty:
        pd.DataFrame().to_csv(OUTPUT_DIR / "P6_remaining_error_case_review.csv", index=False, encoding="utf-8-sig")
        return
    p7u = old_summary[old_summary["keypoint_name"] == "P7U_caudal_fin_upper_tip"].set_index("image_name")
    p7l = old_summary[old_summary["keypoint_name"] == "P7L_caudal_fin_lower_tip"].set_index("image_name")
    review_rows = []
    case_dir = OUTPUT_DIR / "P6_remaining_error_cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    for _, row in p6_rows.iterrows():
        image_name = str(row["image_name"])
        debug = _load_debug(image_name)
        metadata = debug.get("preannotation_metadata", {}) if isinstance(debug.get("preannotation_metadata", {}), dict) else {}
        tail_qc = metadata.get("tail_geometry_qc", {}) if isinstance(metadata.get("tail_geometry_qc", {}), dict) else {}
        failure_type, suggested_fix, worth_fix = _p6_failure_type(row, debug)
        visual = OLD_DIR / "visual_comparisons" / f"{Path(image_name).stem}_compare.png"
        if visual.exists():
            shutil.copy2(visual, case_dir / visual.name)
        tail_preview = PROJECT_ROOT / "results" / "tail_detail_previews" / f"{Path(image_name).stem}_tail_detail_preview.png"
        if tail_preview.exists():
            shutil.copy2(tail_preview, case_dir / tail_preview.name)
        review_rows.append(
            {
                "image_name": image_name,
                "specimen_id": row.get("specimen_id", ""),
                "split": row.get("split", ""),
                "P6_error_mm": row.get("error_mm", ""),
                "P6_auto_x": row.get("auto_x", ""),
                "P6_auto_y": row.get("auto_y", ""),
                "P6_manual_x": row.get("manual_x", ""),
                "P6_manual_y": row.get("manual_y", ""),
                "P6_source": row.get("auto_point_source", metadata.get("P6_source", "")),
                "P6_gap_quality": metadata.get("P6_gap_quality", ""),
                "P6_concavity_quality": metadata.get("P6_concavity_quality", ""),
                "tail_detail_mask_quality": metadata.get("tail_detail_mask_quality", ""),
                "tail_mask_gap_visible": metadata.get("tail_mask_gap_visible", tail_qc.get("tail_mask_gap_visible", "")),
                "tail_mask_gap_filled": metadata.get("tail_mask_gap_filled", tail_qc.get("tail_mask_gap_filled", "")),
                "P5_qc_pass": tail_qc.get("P5_qc_pass", ""),
                "P7U_error_mm": p7u.loc[image_name, "error_mm"] if image_name in p7u.index else "",
                "P7L_error_mm": p7l.loc[image_name, "error_mm"] if image_name in p7l.index else "",
                "tail_axis_source": metadata.get("tail_axis_source", ""),
                "p7v_valid": metadata.get("p7v_valid", ""),
                "possible_failure_type": failure_type,
                "suggested_fix": suggested_fix,
                "worth_algorithm_fix": worth_fix,
                "notes": "Generated from v0.3.3 remaining P6 large-error cases.",
            }
        )
    review_df = pd.DataFrame(review_rows)
    review_df.to_csv(OUTPUT_DIR / "P6_remaining_error_case_review.csv", index=False, encoding="utf-8-sig")
    type_rows = []
    for failure_type, group in review_df.groupby("possible_failure_type"):
        worth = bool(group["worth_algorithm_fix"].astype(bool).any())
        examples = ";".join(group["image_name"].astype(str).head(5))
        suggested = "; ".join(dict.fromkeys(group["suggested_fix"].astype(str)))
        type_rows.append(
            {
                "failure_type": failure_type,
                "count": len(group),
                "example_images": examples,
                "suggested_rule_change": suggested,
                "worth_algorithm_fix": worth,
            }
        )
    pd.DataFrame(type_rows).to_csv(OUTPUT_DIR / "P6_remaining_error_type_summary.csv", index=False, encoding="utf-8-sig")


def _metric(path: Path, key: str) -> tuple[float, int]:
    summary = pd.read_csv(path / "preannotation_vs_manual_summary.csv")
    point = pd.read_csv(path / "keypoint_error_by_point.csv")
    rows = summary[summary["keypoint_name"] == key]
    median = float(point[point["keypoint_name"] == key].iloc[0]["median_error_mm"])
    large = int((rows["error_mm"] > 10).sum())
    return median, large


def append_v034_readme_and_print() -> None:
    old_summary = pd.read_csv(OLD_DIR / "preannotation_vs_manual_summary.csv")
    new_summary = pd.read_csv(OUTPUT_DIR / "preannotation_vs_manual_summary.csv")
    local = pd.read_csv(OUTPUT_DIR / "local_structure_suggestion_comparison.csv")
    old_overall = float(old_summary["error_mm"].median())
    new_overall = float(new_summary["error_mm"].median())
    p6_old, p6_large_old = _metric(OLD_DIR, "P6_caudal_fork_midpoint")
    p6_new, p6_large_new = _metric(OUTPUT_DIR, "P6_caudal_fork_midpoint")

    def local_line(key: str) -> str:
        row = local[local["keypoint_name"] == key]
        if row.empty:
            return f"{key}: no suggestion rows"
        r = row.iloc[0]
        return f"{key}: model median {r['model_median_error_mm']:.3f} mm, suggestion median {r['suggestion_median_error_mm']:.3f} mm"

    recommendations = []
    label_map = {
        "P2_eye_front": "P2",
        "P3_operculum_posterior": "P3",
        "P5_caudal_base_midpoint": "P5",
        "P6_caudal_fork_midpoint": "P6",
    }
    for _, row in local.iterrows():
        key = str(row["keypoint_name"])
        short = label_map.get(key, key)
        if key == "P6_caudal_fork_midpoint":
            recommendations.append(f"- {short}: use gap rule, keep large-error cases for manual review")
        elif bool(row.get("recommend_use_suggestion_as_default", False)):
            recommendations.append(f"- {short}: use suggestion as default")
        else:
            recommendations.append(f"- {short}: QC only")

    extra = f"""

## v0.3.4 Local Structure Summary

Overall median error: {old_overall:.3f} -> {new_overall:.3f} mm

P6 median error: {p6_old:.3f} -> {p6_new:.3f} mm

P6 large error count: {p6_large_old} -> {p6_large_new}

{local_line('P2_eye_front')}

{local_line('P3_operculum_posterior')}

{local_line('P5_caudal_base_midpoint')}

Recommended default changes:

{chr(10).join(recommendations)}

P6 remaining error review:

- `P6_remaining_error_case_review.csv`
- `P6_remaining_error_type_summary.csv`
- `P6_remaining_error_cases/`
"""
    readme = OUTPUT_DIR / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + extra, encoding="utf-8")
    print("Finished v0.3.4 local structure refinement evaluation.")
    print(f"Overall median error: {old_overall:.3f} -> {new_overall:.3f}")
    print(f"P6 median error: {p6_old:.3f} -> {p6_new:.3f}")
    print(f"P6 large error count: {p6_large_old} -> {p6_large_new}")
    for key in ("P2_eye_front", "P3_operculum_posterior", "P5_caudal_base_midpoint"):
        print(local_line(key))
    print("Recommended default changes:")
    for line in recommendations:
        print(line)
    print(f"Results saved to: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    compare_main()
    write_p6_remaining_review()
    append_v034_readme_and_print()
