"""Prepare v0.7 dataset planning files after 5.24 review confirmation.

This script does not train a model or export training tensors. It combines
existing stable labels with confirmed 5.24 labels, checks specimen-level split
rules, and writes planning manifests under results/v0.7_dataset_planning/.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph.config import KEYPOINT_DEFS  # noqa: E402


OUT_DIR = PROJECT_ROOT / "results" / "v0.7_dataset_planning"
OLD_FINAL = PROJECT_ROOT / "results" / "batch_measurement_v0.6.4_stable" / "final_analysis_dataset.csv"
OLD_CORRECTED_DIRS = [
    PROJECT_ROOT / "results" / "real_annotation_5_15_v0.3_corrected" / "keypoints",
    PROJECT_ROOT / "results" / "realworld_review_v0.4.1" / "corrected_keypoints",
]
NEW_REVIEW_ROOT = PROJECT_ROOT / "results" / "realworld_review_5_24_v0.6.5"
NEW_CORRECTED_DIR = NEW_REVIEW_ROOT / "corrected_keypoints"
OLD_SPLIT_PATHS = [
    PROJECT_ROOT / "datasets" / "siganusmorph_heatmap_unet_v0.5_candidate" / "split_manifest.csv",
    PROJECT_ROOT / "datasets" / "siganusmorph_real_5_15_yolopose_maskcrop_v0.3_corrected" / "split_manifest.csv",
]
KEYPOINT_KEYS = [f"{definition.code}_{definition.name}" for definition in KEYPOINT_DEFS]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _has_16(payload: dict[str, Any]) -> bool:
    points = payload.get("corrected_keypoints", {}) if isinstance(payload.get("corrected_keypoints", {}), dict) else {}
    return all(key in points for key in KEYPOINT_KEYS)


def _old_split_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in OLD_SPLIT_PATHS:
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        for _, row in df.iterrows():
            specimen = str(row.get("specimen_id", "") or "")
            split = str(row.get("split", "") or row.get("new_split", "") or "")
            if specimen and split and specimen not in mapping:
                mapping[specimen] = split
    return mapping


def collect_labels() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    old_final = pd.read_csv(OLD_FINAL, dtype=str, keep_default_na=False) if OLD_FINAL.exists() else pd.DataFrame()
    old_images = set(old_final.get("image_name", pd.Series(dtype=str)).astype(str)) if not old_final.empty else set()
    for folder in OLD_CORRECTED_DIRS:
        for path in sorted(folder.glob("*.json")):
            payload = _read_json(path)
            image_name = str(payload.get("image_name", path.stem.replace("_corrected", ".png")))
            if old_images and image_name not in old_images:
                continue
            rows.append(
                {
                    "image_name": image_name,
                    "specimen_id": str(payload.get("specimen_id", "")),
                    "source_batch": "5_15_stable",
                    "source_set": "v0.6.4_final_analysis",
                    "source_json_path": str(path),
                    "annotation_status": payload.get("annotation_status", "corrected_and_confirmed"),
                    "has_16_keypoints": _has_16(payload),
                    "include_in_v07_candidate": _has_16(payload),
                    "exclude_reason": "" if _has_16(payload) else "missing_16_keypoints",
                    "notes": "",
                }
            )
    sample = pd.read_csv(NEW_REVIEW_ROOT / "review_sample_manifest.csv", dtype=str, keep_default_na=False) if (NEW_REVIEW_ROOT / "review_sample_manifest.csv").exists() else pd.DataFrame()
    sample_by_image = {str(row["image_name"]): row for _, row in sample.iterrows()} if not sample.empty else {}
    for path in sorted(NEW_CORRECTED_DIR.glob("*_corrected.json")):
        payload = _read_json(path)
        image_name = str(payload.get("image_name", path.stem.replace("_corrected", ".png")))
        sample_row = sample_by_image.get(image_name, {})
        specimen = str(payload.get("specimen_id", sample_row.get("specimen_id", "")) or "")
        status = str(payload.get("annotation_status", sample_row.get("review_status", "")) or "")
        has_16 = _has_16(payload)
        include = status == "corrected_and_confirmed" and has_16 and specimen not in {"", "unknown"}
        reasons = []
        if status != "corrected_and_confirmed":
            reasons.append("not_corrected_and_confirmed")
        if not has_16:
            reasons.append("missing_16_keypoints")
        if specimen in {"", "unknown"}:
            reasons.append("specimen_id_unconfirmed")
        rows.append(
            {
                "image_name": image_name,
                "specimen_id": specimen,
                "source_batch": "5_24_new_fish",
                "source_set": "realworld_review_5_24_confirmed",
                "source_json_path": str(path),
                "annotation_status": status,
                "has_16_keypoints": has_16,
                "include_in_v07_candidate": include,
                "exclude_reason": ";".join(reasons),
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def assign_splits(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_map = _old_split_map()
    included = manifest[manifest["include_in_v07_candidate"].astype(str).str.lower().isin(["true", "1"])].copy()
    specimen_to_rows = included.groupby("specimen_id").size().to_dict() if not included.empty else {}
    counts_by_split: dict[str, int] = defaultdict(int)
    for specimen, split in split_map.items():
        if specimen in specimen_to_rows:
            counts_by_split[split] += int(specimen_to_rows[specimen])
    target_order = ["train", "val", "test"]
    specimen_assignment: dict[str, str] = {}
    for specimen in sorted(specimen_to_rows):
        if specimen in split_map:
            specimen_assignment[specimen] = split_map[specimen]
        else:
            split = min(target_order, key=lambda item: counts_by_split[item] / {"train": 0.72, "val": 0.14, "test": 0.14}[item])
            specimen_assignment[specimen] = split
            counts_by_split[split] += int(specimen_to_rows[specimen])
    rows: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        specimen = str(row.get("specimen_id", ""))
        include = str(row.get("include_in_v07_candidate", "")).lower() in {"true", "1"}
        old_split = split_map.get(specimen, "")
        new_split = specimen_assignment.get(specimen, "") if include else ""
        rows.append(
            {
                "image_name": row.get("image_name", ""),
                "specimen_id": specimen,
                "source_batch": row.get("source_batch", ""),
                "source_set": row.get("source_set", ""),
                "old_split": old_split,
                "new_split": new_split,
                "split_assignment_reason": "inherit_existing_specimen_split" if old_split else ("new_specimen_balance_split" if include else "excluded_from_candidate"),
                "is_new_specimen": specimen not in split_map and include,
                "leakage_check_pass": True,
            }
        )
    split_plan = pd.DataFrame(rows)
    summary_rows = []
    for split, group in split_plan[split_plan["new_split"].ne("")].groupby("new_split"):
        summary_rows.append(
            {
                "split": split,
                "num_images": len(group),
                "num_specimens": group["specimen_id"].nunique(),
                "num_original_images": int(group["source_batch"].eq("5_15_stable").sum()),
                "num_5_24_images": int(group["source_batch"].eq("5_24_new_fish").sum()),
                "specimen_ids": ";".join(sorted(set(group["specimen_id"].astype(str)))),
            }
        )
    return split_plan, pd.DataFrame(summary_rows)


def integrity_report(manifest: pd.DataFrame, split_plan: pd.DataFrame) -> pd.DataFrame:
    rows = []
    duplicate_images = manifest[manifest.duplicated("image_name", keep=False)]["image_name"].unique().tolist() if not manifest.empty else []
    leakage = []
    if not split_plan.empty:
        for specimen, group in split_plan[split_plan["new_split"].ne("")].groupby("specimen_id"):
            if group["new_split"].nunique() > 1:
                leakage.append(specimen)
    unknown_included = manifest[
        manifest["include_in_v07_candidate"].astype(str).str.lower().isin(["true", "1"])
        & manifest["specimen_id"].astype(str).isin(["", "unknown"])
    ]["image_name"].tolist() if not manifest.empty else []
    checks = [
        ("duplicate_image_name", not duplicate_images, duplicate_images, "Resolve duplicate source labels before training."),
        ("specimen_split_leakage", not leakage, leakage, "Keep each specimen_id in only one split."),
        ("unknown_specimen_included", not unknown_included, unknown_included, "Confirm specimen_id before including."),
        ("keypoint_schema_16_points", bool(manifest["has_16_keypoints"].astype(str).str.lower().isin(["true", "1"]).all()) if not manifest.empty else True, [], "Exclude incomplete labels."),
    ]
    for name, ok, examples, action in checks:
        rows.append({"check_name": name, "status": "pass" if ok else "fail", "num_issues": 0 if ok else len(examples), "issue_examples": ";".join(map(str, examples[:10])), "suggested_action": action})
    return pd.DataFrame(rows)


def hard_cases(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pre_qc_path = NEW_REVIEW_ROOT / "preannotation_qc.csv"
    pre_qc = pd.read_csv(pre_qc_path, dtype=str, keep_default_na=False) if pre_qc_path.exists() else pd.DataFrame()
    for _, row in pre_qc.iterrows():
        reasons = []
        if str(row.get("P6_needs_review", "")).lower() == "true":
            reasons.append("P6_warning")
        if str(row.get("measurements_needs_review", "")).lower() == "true":
            reasons.append("measurement_axis_warning")
        if reasons:
            rows.append(
                {
                    "image_name": row.get("image_name", ""),
                    "specimen_id": row.get("specimen_id", ""),
                    "source_batch": "5_24_new_fish",
                    "hard_case_type": ";".join(reasons),
                    "affected_keypoints": "P6" if "P6_warning" in reasons else "",
                    "review_reason": row.get("notes", ""),
                    "recommended_use": "train_hard_case",
                }
            )
    return pd.DataFrame(rows)


def write_training_plan(manifest: pd.DataFrame, split_summary: pd.DataFrame, integrity: pd.DataFrame) -> None:
    included = manifest[manifest["include_in_v07_candidate"].astype(str).str.lower().isin(["true", "1"])].copy()
    new_confirmed = included[included["source_batch"].eq("5_24_new_fish")]
    leakage_pass = bool(integrity.loc[integrity["check_name"].eq("specimen_split_leakage"), "status"].eq("pass").all()) if not integrity.empty else True
    recommend = len(new_confirmed) >= 40 and not new_confirmed["specimen_id"].astype(str).isin(["", "unknown"]).any()
    text = f"""# v0.7 Training Plan

This is a planning document only. No model training has been run.

## Candidate Data

- Total candidate images: {len(included)}
- Total specimens: {included['specimen_id'].nunique() if not included.empty else 0}
- Existing stable images: {int(included['source_batch'].eq('5_15_stable').sum()) if not included.empty else 0}
- 5.24 confirmed images: {len(new_confirmed)}
- Specimen leakage check: {'pass' if leakage_pass else 'fail'}

## Split Summary

{split_summary.to_markdown(index=False) if not split_summary.empty else 'No split summary available yet.'}

## Training Goals

- Improve robustness on new 5.24 fish while preserving stable v0.6.4 performance.
- Monitor P6 warnings, complex tails, P8/P9 corrections, high curvature, and warp/segmentation QC warnings.
- Keep specimen-level splits. Do not include pending, excluded, needs-retake, or unknown-specimen images.

## Recommendation

`recommend_train_v07 = {str(bool(recommend)).lower()}`

If this is `false`, finish manual confirmation and specimen ID cleanup before training.
"""
    (OUT_DIR / "v07_training_plan.md").write_text(text, encoding="utf-8")
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = collect_labels()
    split_plan, split_summary = assign_splits(manifest)
    integrity = integrity_report(manifest, split_plan)
    hard = hard_cases(manifest)
    manifest.to_csv(OUT_DIR / "v07_candidate_label_manifest.csv", index=False, encoding="utf-8-sig")
    split_plan.to_csv(OUT_DIR / "v07_split_plan.csv", index=False, encoding="utf-8-sig")
    split_summary.to_csv(OUT_DIR / "v07_split_summary.csv", index=False, encoding="utf-8-sig")
    integrity.to_csv(OUT_DIR / "v07_data_integrity_report.csv", index=False, encoding="utf-8-sig")
    hard.to_csv(OUT_DIR / "v07_hard_cases.csv", index=False, encoding="utf-8-sig")
    write_training_plan(manifest, split_summary, integrity)
    included = manifest[manifest["include_in_v07_candidate"].astype(str).str.lower().isin(["true", "1"])]
    print("Finished v0.7 dataset planning.")
    print(f"Candidate images: {len(included)}")
    print(f"Specimens: {included['specimen_id'].nunique() if not included.empty else 0}")
    print(f"Outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
