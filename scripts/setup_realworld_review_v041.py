"""Set up v0.4.1 real-world review workflow.

The script creates a review manifest, selects unseen real images, runs the
current v0.4.1 hybrid preannotation, and saves unverified preannotation bundles.
It does not train models and does not modify existing corrected labels.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph.hybrid_preannotation import preannotate_warped_image_hybrid_v04  # noqa: E402
from siganusmorph.image_utils import load_image_file  # noqa: E402
from siganusmorph.realworld_review import (  # noqa: E402
    REVIEW_VERSION,
    manifest_path,
    path_from_project,
    preannotation_json_path,
    review_root,
    sample_manifest_path,
    save_preannotation_artifacts,
    warped_name_from_image_name,
)
from siganusmorph.segmentation import segment_fish_from_blue_board  # noqa: E402


SPLIT_MANIFEST = PROJECT_ROOT / "datasets" / "siganusmorph_heatmap_unet_v0.1" / "split_manifest.csv"
REAL_MANIFEST = PROJECT_ROOT / "data" / "real_images_manifest.csv"
MAX_REVIEW_IMAGES = 30
MIN_TARGET_REVIEW_IMAGES = 20


def _bool_text(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _source_status(image_name: str, used_set: set[str]) -> str:
    return "used_in_training_dataset" if image_name in used_set else "unseen_real_image"


def _quality_flag(row: pd.Series, warped_path: Path, segmentation_success: bool, segmentation_quality: str) -> tuple[str, bool, str]:
    warp_success = _bool_text(row.get("warp_success", ""))
    if not warp_success:
        return "unusable_for_review", False, "warp_failed"
    if not warped_path.exists():
        return "unusable_for_review", False, "missing_warped_image"
    exclude_reason = str(row.get("exclude_reason", "") or "")
    if "bad_warp" in exclude_reason or "reflection" in exclude_reason:
        return "unusable_for_review", False, exclude_reason
    if not segmentation_success:
        return "segmentation_warning", False, "segmentation_failed"
    if str(segmentation_quality).startswith("warning"):
        return "segmentation_warning", True, str(segmentation_quality)
    return "ok", True, ""


def build_review_manifest() -> pd.DataFrame:
    if not REAL_MANIFEST.exists():
        raise FileNotFoundError(f"Missing manifest: {REAL_MANIFEST}")
    real_df = pd.read_csv(REAL_MANIFEST, dtype=str, keep_default_na=False)
    split_df = pd.read_csv(SPLIT_MANIFEST, dtype=str, keep_default_na=False) if SPLIT_MANIFEST.exists() else pd.DataFrame()
    used_set = set(split_df.get("image_name", pd.Series(dtype=str)).astype(str)) if not split_df.empty else set()
    rows: list[dict[str, Any]] = []
    for _, row in real_df.iterrows():
        image_name = str(row.get("image_name", ""))
        warped_rel = str(row.get("warped_image_path", "")) or str(Path("data") / "real_images_warped" / warped_name_from_image_name(image_name))
        warped_path = path_from_project(PROJECT_ROOT, warped_rel)
        seg_success = False
        seg_quality = ""
        if warped_path.exists() and _bool_text(row.get("warp_success", "")):
            try:
                image = load_image_file(warped_path)
                _mask, _bbox, quality = segment_fish_from_blue_board(image)
                seg_success = bool(quality.get("segmentation_success", False))
                seg_quality = str(quality.get("segmentation_quality", ""))
            except Exception as exc:
                seg_success = False
                seg_quality = f"segmentation_exception:{exc}"
        quality_flag, include_default, exclude_reason = _quality_flag(row, warped_path, seg_success, seg_quality)
        used = image_name in used_set
        source_status = _source_status(image_name, used_set)
        include_for_review = include_default and not used
        rows.append(
            {
                "image_name": image_name,
                "original_file_name": row.get("original_file_name", ""),
                "original_path": row.get("original_path", ""),
                "warped_image_path": warped_rel,
                "specimen_id": row.get("specimen_id", ""),
                "view_id": row.get("view_id", ""),
                "source_status": source_status,
                "used_in_training_dataset": used,
                "warp_success": _bool_text(row.get("warp_success", "")),
                "segmentation_success": seg_success,
                "quality_flag": quality_flag,
                "include_for_review": include_for_review,
                "exclude_reason": exclude_reason,
                "notes": row.get("curation_notes", "") or row.get("notes", ""),
            }
        )
    out = pd.DataFrame(rows)
    path = manifest_path(PROJECT_ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out


def choose_review_sample(review_df: pd.DataFrame) -> pd.DataFrame:
    candidates = review_df[
        (review_df["source_status"] == "unseen_real_image")
        & (review_df["include_for_review"].astype(str).str.lower() == "true")
    ].copy()
    if candidates.empty:
        candidates = review_df[
            (review_df["source_status"] == "unseen_real_image")
            & (review_df["warp_success"].astype(str).str.lower() == "true")
        ].copy()
    candidates["_quality_rank"] = candidates["quality_flag"].map({"ok": 0, "segmentation_warning": 1}).fillna(9)
    candidates["_name_rank"] = candidates["image_name"].astype(str)
    candidates = candidates.sort_values(["_quality_rank", "specimen_id", "_name_rank"])
    selected_rows = []
    counts: dict[str, int] = defaultdict(int)
    for _, row in candidates.iterrows():
        specimen = str(row.get("specimen_id", ""))
        if counts[specimen] >= 2:
            continue
        selected_rows.append(row)
        counts[specimen] += 1
        if len(selected_rows) >= MAX_REVIEW_IMAGES:
            break
    if len(selected_rows) < min(MIN_TARGET_REVIEW_IMAGES, len(candidates)):
        selected_names = {str(row["image_name"]) for row in selected_rows}
        for _, row in candidates.iterrows():
            if str(row["image_name"]) in selected_names:
                continue
            selected_rows.append(row)
            if len(selected_rows) >= min(MAX_REVIEW_IMAGES, len(candidates)):
                break
    sample = pd.DataFrame(selected_rows).drop(columns=[col for col in ("_quality_rank", "_name_rank") if col in candidates.columns], errors="ignore")
    if sample.empty:
        sample = pd.DataFrame(columns=list(review_df.columns))
    sample["review_status"] = "pending_review"
    sample["preannotation_json_path"] = sample["image_name"].astype(str).map(lambda name: str(preannotation_json_path(PROJECT_ROOT, name)))
    path = sample_manifest_path(PROJECT_ROOT)
    sample.to_csv(path, index=False, encoding="utf-8-sig")
    return sample


def run_preannotations(sample: pd.DataFrame) -> pd.DataFrame:
    qc_rows = []
    for idx, row in sample.iterrows():
        image_name = str(row["image_name"])
        warped_path = path_from_project(PROJECT_ROOT, str(row["warped_image_path"]))
        print(f"[{idx + 1}/{len(sample)}] preannotating {image_name}")
        if not warped_path.exists():
            qc_rows.append({"image_name": image_name, "preannotation_success": False, "review_reason": "missing_warped_image"})
            continue
        image = load_image_file(warped_path)
        try:
            result = preannotate_warped_image_hybrid_v04(image, warped_path.name, PROJECT_ROOT, mm_per_pixel=0.1)
            save_preannotation_artifacts(PROJECT_ROOT, row.to_dict(), image, result, mm_per_pixel=0.1)
            metadata = result.get("metadata", {})
            qc_rows.append(
                {
                    "image_name": image_name,
                    "specimen_id": row.get("specimen_id", ""),
                    "preannotation_success": True,
                    "segmentation_success": metadata.get("segmentation_success", ""),
                    "segmentation_quality": metadata.get("segmentation_quality", ""),
                    "p7v_valid": metadata.get("p7v_valid", ""),
                    "curvature_qc_level": metadata.get("curvature_qc_level", ""),
                    "review_reason": metadata.get("review_reason", ""),
                }
            )
        except Exception as exc:
            qc_rows.append({"image_name": image_name, "specimen_id": row.get("specimen_id", ""), "preannotation_success": False, "review_reason": str(exc)})
    qc_df = pd.DataFrame(qc_rows)
    qc_path = review_root(PROJECT_ROOT) / "preannotations" / "qc_summary.csv"
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    qc_df.to_csv(qc_path, index=False, encoding="utf-8-sig")
    return qc_df


def write_readme(review_df: pd.DataFrame, sample: pd.DataFrame, qc_df: pd.DataFrame) -> None:
    unseen_count = int((review_df["source_status"] == "unseen_real_image").sum()) if not review_df.empty else 0
    excluded = review_df[review_df["include_for_review"].astype(str).str.lower() != "true"] if not review_df.empty else pd.DataFrame()
    exclude_counts = excluded["exclude_reason"].replace("", "not_selected_or_used").value_counts().to_dict() if not excluded.empty else {}
    pre_success = int(qc_df["preannotation_success"].astype(str).str.lower().eq("true").sum()) if not qc_df.empty else 0
    text = f"""# v0.4.1 Real-World Review Workflow

Workflow version: `{REVIEW_VERSION}`

This setup uses real images only. It does not train a model and does not write unverified points as corrected labels.

## Sample

- Review sample images: {len(sample)}
- Unseen real images in manifest: {unseen_count}
- Successful preannotations: {pre_success}
- Excluded / not selected counts: {exclude_counts}

If fewer than 20 unseen usable images are selected, it means the imported non-training pool was smaller or failed warp/segmentation checks.

## How To Review

Open Streamlit and go to the `realworld_review` page. Each image loads the saved v0.4.1 preannotation as an editable draft. Save only after manual confirmation.

Corrected labels are saved to:

- `results/realworld_review_v0.4.1/corrected_keypoints/`
- `results/realworld_review_v0.4.1/corrected_previews/`

After confirming a batch, run:

```bash
python scripts/evaluate_realworld_review_v041.py
```

to summarize correction effort and QC usefulness.
"""
    (review_root(PROJECT_ROOT) / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    root = review_root(PROJECT_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    review_df = build_review_manifest()
    sample = choose_review_sample(review_df)
    qc_df = run_preannotations(sample)
    write_readme(review_df, sample, qc_df)
    unseen_count = int((review_df["source_status"] == "unseen_real_image").sum()) if not review_df.empty else 0
    excluded_count = int((review_df["include_for_review"].astype(str).str.lower() != "true").sum()) if not review_df.empty else 0
    print("Finished v0.4.1 real-world review workflow setup.")
    print(f"Review sample images: {len(sample)}")
    print(f"Unseen real images: {unseen_count}")
    print(f"Excluded images: {excluded_count}")
    print("Preannotations saved to:")
    print("results/realworld_review_v0.4.1/preannotations/")
    print("Review queue ready in Streamlit.")
    print("After manual confirmation, run post-review evaluation to summarize correction effort.")


if __name__ == "__main__":
    main()
