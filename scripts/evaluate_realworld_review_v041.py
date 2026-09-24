"""Post-review evaluation for v0.4.1 real-world review workflow."""

from __future__ import annotations

import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph.image_utils import load_image_file  # noqa: E402
from siganusmorph.realworld_review import (  # noqa: E402
    KEYPOINT_KEYS,
    base_stem,
    corrected_dir,
    corrected_json_path,
    draw_review_comparison,
    path_from_project,
    preannotation_dir,
    preannotation_json_path,
    read_json,
    review_root,
    sample_manifest_path,
    time_log_path,
    xy_from_payload,
)


OUTPUT_DIR = review_root(PROJECT_ROOT) / "post_review_evaluation"
MM_PER_PIXEL = 0.1


def _distance_mm(a: Any, b: Any) -> float:
    pa = xy_from_payload(a)
    pb = xy_from_payload(b)
    if pa is None or pb is None:
        return float("nan")
    return math.hypot(pa[0] - pb[0], pa[1] - pb[1]) * MM_PER_PIXEL


def _corrected_payloads() -> list[dict[str, Any]]:
    out = []
    for path in sorted(corrected_dir(PROJECT_ROOT).glob("*_corrected.json")):
        try:
            payload = read_json(path)
        except Exception:
            continue
        payload["_path"] = str(path)
        out.append(payload)
    return out


def _confirmed_payloads_from_manifest(sample: pd.DataFrame) -> list[dict[str, Any]]:
    """Load only confirmed review samples that have a corrected JSON on disk."""
    if sample.empty or "image_name" not in sample.columns:
        return _corrected_payloads()
    if "review_status" not in sample.columns:
        sample["review_status"] = ""
    out = []
    confirmed = sample.loc[sample["review_status"].astype(str) == "corrected_and_confirmed"].copy()
    for _, row in confirmed.iterrows():
        image_name = str(row.get("image_name", ""))
        path = corrected_json_path(PROJECT_ROOT, image_name)
        if not path.exists():
            continue
        try:
            payload = read_json(path)
        except Exception:
            continue
        payload["_path"] = str(path)
        out.append(payload)
    return out


def _warning_tokens(review_reason: str) -> list[str]:
    return [token.strip() for token in str(review_reason or "").replace(",", ";").split(";") if token.strip()]


def _preannotation_payload(image_name: str) -> dict[str, Any]:
    path = preannotation_json_path(PROJECT_ROOT, image_name)
    if not path.exists():
        return {}
    try:
        return read_json(path)
    except Exception:
        return {}


def _preannotation_mode(image_name: str) -> str:
    payload = _preannotation_payload(image_name)
    metadata = payload.get("preannotation_metadata", {}) if isinstance(payload, Mapping) else {}
    if isinstance(metadata, Mapping):
        return str(metadata.get("preannotation_mode", ""))
    return ""


def _suggested_next_action(exclude_reason: str) -> str:
    if exclude_reason == "failed_preannotation":
        return "Inspect error_cases preview/debug; keep excluded from metrics and improve preannotation before adding to training."
    if exclude_reason:
        return "Review manually before using as training or evaluation data."
    return ""


def _copy_error_case_files(image_name: str, output_dir: Path) -> None:
    stem = base_stem(image_name)
    error_dir = review_root(PROJECT_ROOT) / "error_cases"
    error_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        preannotation_dir(PROJECT_ROOT) / "previews" / f"{stem}_preannotation.png",
        preannotation_dir(PROJECT_ROOT) / "debug_json" / f"{stem}_debug.json",
        preannotation_dir(PROJECT_ROOT) / "json" / f"{stem}_preannotation.json",
    ]
    for src in sources:
        if src.exists():
            shutil.copy2(src, error_dir / src.name)
            shutil.copy2(src, output_dir / "error_cases" / src.name)


def _excluded_cases(sample: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    columns = [
        "image_name",
        "specimen_id",
        "exclude_reason",
        "preannotation_mode",
        "reviewer_notes",
        "suggested_next_action",
    ]
    (output_dir / "error_cases").mkdir(parents=True, exist_ok=True)
    if sample.empty or "review_status" not in sample.columns:
        return pd.DataFrame(columns=columns)
    excluded = sample.loc[sample["review_status"].astype(str) == "excluded"].copy()
    rows = []
    for _, row in excluded.iterrows():
        image_name = str(row.get("image_name", ""))
        reason = str(row.get("exclude_reason", ""))
        if reason == "failed_preannotation":
            _copy_error_case_files(image_name, output_dir)
        rows.append(
            {
                "image_name": image_name,
                "specimen_id": str(row.get("specimen_id", "")),
                "exclude_reason": reason,
                "preannotation_mode": _preannotation_mode(image_name),
                "reviewer_notes": str(row.get("notes", "")),
                "suggested_next_action": _suggested_next_action(reason),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "visual_comparisons").mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(sample_manifest_path(PROJECT_ROOT), dtype=str, keep_default_na=False) if sample_manifest_path(PROJECT_ROOT).exists() else pd.DataFrame()
    sample_by_image = {str(row["image_name"]): row for _, row in sample.iterrows()} if not sample.empty else {}
    corrected_payloads = _confirmed_payloads_from_manifest(sample)
    confirmed_images = {str(payload.get("image_name", "")) for payload in corrected_payloads}
    excluded_df = _excluded_cases(sample, OUTPUT_DIR)

    rows = []
    image_rows = []
    qc_rows = []
    for payload in corrected_payloads:
        image_name = str(payload.get("image_name", ""))
        corrected = payload.get("corrected_keypoints", {})
        pre = payload.get("preannotation", {})
        pre_points = pre.get("hybrid_keypoints", {}) if isinstance(pre, Mapping) else {}
        if not isinstance(pre_points, Mapping):
            pre_points = {}
        pre_reason = str(pre.get("review_reason", "")) if isinstance(pre, Mapping) else ""
        specimen_id = str(payload.get("specimen_id", ""))
        per_image_errors = []
        moved_keys = []
        for key in KEYPOINT_KEYS:
            err = _distance_mm(pre_points.get(key), corrected.get(key))
            was_moved = bool(np.isfinite(err) and err > 0.5)
            if np.isfinite(err):
                per_image_errors.append(err)
            if was_moved:
                moved_keys.append(key)
            rows.append(
                {
                    "image_name": image_name,
                    "specimen_id": specimen_id,
                    "keypoint_name": key,
                    "error_mm": err,
                    "displacement_mm": err,
                    "was_moved": was_moved,
                    "point_source_before": (pre.get("point_sources", {}) or {}).get(key, "") if isinstance(pre, Mapping) else "",
                    "review_reason": pre_reason,
                }
            )
        warnings = _warning_tokens(pre_reason)
        large_correction = [key for key in moved_keys if next((r["displacement_mm"] for r in rows if r["image_name"] == image_name and r["keypoint_name"] == key), 0) > 5]
        image_rows.append(
            {
                "image_name": image_name,
                "specimen_id": specimen_id,
                "mean_displacement_mm": float(np.mean(per_image_errors)) if per_image_errors else np.nan,
                "median_displacement_mm": float(np.median(per_image_errors)) if per_image_errors else np.nan,
                "max_displacement_mm": float(np.max(per_image_errors)) if per_image_errors else np.nan,
                "num_points_moved": len(moved_keys),
                "moved_keypoints": ";".join(moved_keys),
                "num_large_corrections": len(large_correction),
                "large_correction_keypoints": ";".join(large_correction),
                "num_qc_warnings": len(warnings),
                "qc_warning_types": ";".join(warnings),
            }
        )
        for warning in warnings:
            qc_rows.append(
                {
                    "image_name": image_name,
                    "specimen_id": specimen_id,
                    "qc_warning_type": warning,
                    "has_large_correction": bool(large_correction),
                    "large_correction_keypoints": ";".join(large_correction),
                    "num_points_moved": len(moved_keys),
                }
            )
        sample_row = sample_by_image.get(image_name)
        if sample_row is not None:
            warped_path = path_from_project(PROJECT_ROOT, str(sample_row.get("warped_image_path", "")))
            if warped_path.exists():
                image = load_image_file(warped_path)
                draw_review_comparison(
                    image,
                    pre_points,
                    corrected,
                    OUTPUT_DIR / "visual_comparisons" / f"{Path(image_name).stem}_review_compare.png",
                    title=f"{image_name} {specimen_id}",
                    mm_per_pixel=MM_PER_PIXEL,
                )

    detail_df = pd.DataFrame(rows)
    by_point = (
        detail_df.groupby("keypoint_name")
        .agg(
            num_images=("image_name", "count"),
            mean_displacement_mm=("displacement_mm", "mean"),
            median_displacement_mm=("displacement_mm", "median"),
            max_displacement_mm=("displacement_mm", "max"),
            moved_rate=("was_moved", "mean"),
        )
        .reset_index()
        if not detail_df.empty
        else pd.DataFrame()
    )
    by_image = pd.DataFrame(image_rows)
    qc_df = pd.DataFrame(qc_rows)
    time_log = pd.read_csv(time_log_path(PROJECT_ROOT), dtype=str, keep_default_na=False) if time_log_path(PROJECT_ROOT).exists() else pd.DataFrame()
    if not time_log.empty and "image_name" in time_log.columns:
        time_log = time_log.loc[time_log["image_name"].astype(str).isin(confirmed_images)].copy()

    detail_df.to_csv(OUTPUT_DIR / "manual_correction_by_point_detail.csv", index=False, encoding="utf-8-sig")
    by_point.to_csv(OUTPUT_DIR / "keypoint_error_by_point.csv", index=False, encoding="utf-8-sig")
    by_image.to_csv(OUTPUT_DIR / "keypoint_error_by_image.csv", index=False, encoding="utf-8-sig")
    by_point.to_csv(OUTPUT_DIR / "manual_correction_by_point.csv", index=False, encoding="utf-8-sig")
    by_image.to_csv(OUTPUT_DIR / "manual_correction_by_image.csv", index=False, encoding="utf-8-sig")
    qc_df.to_csv(OUTPUT_DIR / "qc_warning_accuracy.csv", index=False, encoding="utf-8-sig")
    excluded_df.to_csv(OUTPUT_DIR / "excluded_cases.csv", index=False, encoding="utf-8-sig")

    total_sample_count = int(len(sample)) if not sample.empty else len(corrected_payloads)
    excluded_count = int((sample.get("review_status", pd.Series(dtype=str)).astype(str) == "excluded").sum()) if not sample.empty else 0
    reviewed = len(corrected_payloads)
    pending_count = max(0, total_sample_count - reviewed - excluded_count)
    denominator = max(0, total_sample_count - excluded_count)
    completion_rate = (reviewed / denominator * 100.0) if denominator else np.nan
    completion_rate_text = (
        f"{completion_rate:.1f}".rstrip("0").rstrip(".") + "%"
        if np.isfinite(completion_rate)
        else ""
    )
    duration = pd.to_numeric(time_log.get("duration_sec", pd.Series(dtype=str)), errors="coerce") if not time_log.empty else pd.Series(dtype=float)
    displacements = pd.to_numeric(detail_df.get("displacement_mm", pd.Series(dtype=float)), errors="coerce") if not detail_df.empty else pd.Series(dtype=float)
    moved_counts = pd.to_numeric(by_image.get("num_points_moved", pd.Series(dtype=float)), errors="coerce") if not by_image.empty else pd.Series(dtype=float)
    moved_counter = Counter(";".join(by_image.get("moved_keypoints", [])).split(";")) if not by_image.empty else Counter()
    moved_counter.pop("", None)
    efficiency = pd.DataFrame(
        [
            {
                "total_sample_count": total_sample_count,
                "reviewed_count": reviewed,
                "excluded_count": excluded_count,
                "pending_count": pending_count,
                "completion_rate_excluding_excluded": completion_rate_text,
                "num_images_reviewed": reviewed,
                "mean_review_time_sec": duration.mean() if len(duration) else np.nan,
                "median_review_time_sec": duration.median() if len(duration) else np.nan,
                "mean_num_points_moved": moved_counts.mean() if len(moved_counts) else np.nan,
                "median_num_points_moved": moved_counts.median() if len(moved_counts) else np.nan,
                "most_frequently_moved_keypoints": ";".join(key for key, _ in moved_counter.most_common(8)),
                "mean_displacement_mm": displacements.mean() if len(displacements) else np.nan,
                "median_displacement_mm": displacements.median() if len(displacements) else np.nan,
                "estimated_time_saved_vs_manual": "requires manual baseline timing; compare against full 16-point manual annotation time",
                "notes": "Auto preannotation points are compared to user-confirmed corrected_keypoints.",
            }
        ]
    )
    efficiency.to_csv(OUTPUT_DIR / "review_efficiency_summary.csv", index=False, encoding="utf-8-sig")
    readme = f"""# Post-review Evaluation v0.4.1

Review sample images: {total_sample_count}

Confirmed images used for statistics: {reviewed}

Excluded images: {excluded_count}

Pending images: {pending_count}

本轮 review 样本 30 张，其中 29 张完成人工确认，1 张因识别错误排除。统计结果基于 29 张 confirmed images。

Mean review time sec: {efficiency.iloc[0]['mean_review_time_sec']}

Median review time sec: {efficiency.iloc[0]['median_review_time_sec']}

Most frequently moved keypoints: {efficiency.iloc[0]['most_frequently_moved_keypoints']}

This report compares saved unverified v0.4.1 preannotation points with manually confirmed corrected_keypoints.
Excluded samples are listed in `excluded_cases.csv` and are not included in keypoint or manual-correction metrics.
"""
    (OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8-sig")
    print(f"Post-review evaluation complete. Reviewed images: {reviewed}")
    print(f"Excluded images: {excluded_count}")
    print(f"Results saved to: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
