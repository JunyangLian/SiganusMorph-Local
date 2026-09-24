"""Summarize the 5.24 v0.6.5 real-world review batch after confirmation.

The script is safe to run before all images are confirmed.  It reads only the
5.24 review directory and writes post-review summary files; it never modifies
corrected keypoints.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph.config import KEYPOINT_DEFS  # noqa: E402


REVIEW_ROOT = PROJECT_ROOT / "results" / "realworld_review_5_24_v0.6.5"
OUT_DIR = REVIEW_ROOT / "post_review_evaluation"
KEYPOINT_KEYS = [f"{definition.code}_{definition.name}" for definition in KEYPOINT_DEFS]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() else pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _status_counts(sample: pd.DataFrame) -> dict[str, int]:
    if sample.empty or "review_status" not in sample.columns:
        return {"confirmed": 0, "excluded": 0, "pending": 0}
    status = sample["review_status"].astype(str)
    return {
        "confirmed": int(status.eq("corrected_and_confirmed").sum()),
        "excluded": int(status.eq("excluded").sum()),
        "needs_retake": int(status.eq("needs_retake").sum()),
        "needs_later_review": int(status.eq("needs_later_review").sum()),
        "pending": int(status.isin(["pending_review", "in_progress"]).sum()),
    }


def summarize_efficiency(sample: pd.DataFrame, time_log: pd.DataFrame, correction: pd.DataFrame) -> pd.DataFrame:
    counts = _status_counts(sample)
    durations = _to_float(time_log.get("duration_sec", pd.Series(dtype=str))) if not time_log.empty else pd.Series(dtype=float)
    moved_by_image = pd.DataFrame()
    if not correction.empty and "was_moved" in correction.columns:
        tmp = correction.copy()
        tmp["was_moved_bool"] = tmp["was_moved"].astype(str).str.lower().isin(["true", "1", "yes"])
        tmp["displacement_mm_num"] = _to_float(tmp.get("displacement_mm", pd.Series(dtype=str)))
        moved_by_image = tmp.groupby("image_name", as_index=False).agg(
            num_points_moved=("was_moved_bool", "sum"),
            mean_displacement_mm=("displacement_mm_num", "mean"),
            median_displacement_mm=("displacement_mm_num", "median"),
        )
    rows = [
        {
            "total_sample_count": len(sample),
            "confirmed_count": counts["confirmed"],
            "excluded_count": counts["excluded"],
            "needs_retake_count": counts["needs_retake"],
            "needs_later_review_count": counts["needs_later_review"],
            "pending_count": counts["pending"],
            "mean_review_time_sec": round(float(durations.mean()), 3) if not durations.empty else "",
            "median_review_time_sec": round(float(durations.median()), 3) if not durations.empty else "",
            "mean_num_points_moved": round(float(moved_by_image["num_points_moved"].mean()), 3) if not moved_by_image.empty else "",
            "median_num_points_moved": round(float(moved_by_image["num_points_moved"].median()), 3) if not moved_by_image.empty else "",
            "mean_displacement_mm": round(float(moved_by_image["mean_displacement_mm"].mean()), 3) if not moved_by_image.empty else "",
            "median_displacement_mm": round(float(moved_by_image["median_displacement_mm"].median()), 3) if not moved_by_image.empty else "",
        }
    ]
    return pd.DataFrame(rows)


def correction_by_point(correction: pd.DataFrame) -> pd.DataFrame:
    if correction.empty:
        return pd.DataFrame(columns=["keypoint_name", "num_images", "was_moved_count", "moved_rate", "mean_displacement_mm", "median_displacement_mm", "max_displacement_mm"])
    df = correction.copy()
    df["was_moved_bool"] = df["was_moved"].astype(str).str.lower().isin(["true", "1", "yes"])
    df["displacement_mm_num"] = _to_float(df.get("displacement_mm", pd.Series(dtype=str)))
    out = df.groupby("keypoint_name", as_index=False).agg(
        num_images=("image_name", "nunique"),
        was_moved_count=("was_moved_bool", "sum"),
        mean_displacement_mm=("displacement_mm_num", "mean"),
        median_displacement_mm=("displacement_mm_num", "median"),
        max_displacement_mm=("displacement_mm_num", "max"),
    )
    out["moved_rate"] = out["was_moved_count"] / out["num_images"].replace(0, pd.NA)
    return out


def correction_by_image(correction: pd.DataFrame) -> pd.DataFrame:
    if correction.empty:
        return pd.DataFrame(columns=["image_name", "num_points_moved", "mean_displacement_mm", "max_displacement_mm", "moved_keypoints"])
    df = correction.copy()
    df["was_moved_bool"] = df["was_moved"].astype(str).str.lower().isin(["true", "1", "yes"])
    df["displacement_mm_num"] = _to_float(df.get("displacement_mm", pd.Series(dtype=str)))
    moved = df[df["was_moved_bool"]].groupby("image_name")["keypoint_name"].apply(lambda s: ";".join(s.astype(str))).rename("moved_keypoints")
    out = df.groupby("image_name", as_index=False).agg(
        num_points_moved=("was_moved_bool", "sum"),
        mean_displacement_mm=("displacement_mm_num", "mean"),
        max_displacement_mm=("displacement_mm_num", "max"),
    )
    out = out.merge(moved.reset_index(), on="image_name", how="left")
    out["moved_keypoints"] = out["moved_keypoints"].fillna("")
    return out


def p6_qc_effectiveness(sample: pd.DataFrame, correction: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    p6_correction = correction[correction.get("keypoint_name", pd.Series(dtype=str)).astype(str) == "P6_caudal_fork_midpoint"].copy() if not correction.empty else pd.DataFrame()
    p6_by_image = {}
    if not p6_correction.empty:
        p6_correction["displacement_mm_num"] = _to_float(p6_correction.get("displacement_mm", pd.Series(dtype=str)))
        p6_by_image = p6_correction.set_index("image_name").to_dict("index")
    for _, row in sample.iterrows():
        image_name = str(row.get("image_name", ""))
        pre_path = PROJECT_ROOT / str(row.get("preannotation_json_path", ""))
        payload = _read_json(pre_path)
        tail_qc = payload.get("tail_qc", {}) or (payload.get("preannotation_metadata", {}) or {}).get("tail_qc", {})
        p6 = p6_by_image.get(image_name, {})
        rows.append(
            {
                "image_name": image_name,
                "specimen_id": row.get("specimen_id", ""),
                "review_status": row.get("review_status", ""),
                "P6_geometry_qc_pass": tail_qc.get("P6_geometry_qc_pass", ""),
                "P6_needs_review": tail_qc.get("P6_needs_review", ""),
                "P6_fallback_used": tail_qc.get("P6_fallback_used", ""),
                "P6_displacement_mm": p6.get("displacement_mm", ""),
                "P6_was_moved": p6.get("was_moved", ""),
                "qc_matched_large_correction": bool(str(tail_qc.get("P6_needs_review", "")).lower() == "true" and float(p6.get("displacement_mm_num", 0) or 0) > 5),
            }
        )
    return pd.DataFrame(rows)


def excluded_cases(sample: pd.DataFrame) -> pd.DataFrame:
    if sample.empty:
        return pd.DataFrame(columns=["image_name", "specimen_id", "review_status", "exclude_reason", "notes", "suggested_next_action"])
    excluded = sample[sample["review_status"].astype(str).isin(["excluded", "needs_retake"])].copy()
    if excluded.empty:
        return pd.DataFrame(columns=["image_name", "specimen_id", "review_status", "exclude_reason", "notes", "suggested_next_action"])
    excluded["suggested_next_action"] = excluded["review_status"].map({"needs_retake": "retake_photo", "excluded": "inspect_before_training"}).fillna("inspect")
    return excluded[["image_name", "specimen_id", "review_status", "exclude_reason", "notes", "suggested_next_action"]]


def write_readme(summary: pd.DataFrame, by_point: pd.DataFrame) -> None:
    row = summary.iloc[0].to_dict() if not summary.empty else {}
    top = by_point.sort_values("was_moved_count", ascending=False).head(5)["keypoint_name"].tolist() if not by_point.empty else []
    text = f"""# 5.24 v0.6.5 Post-Review Evaluation

This report summarizes the 5.24 new-fish review batch. It does not modify corrected keypoints.

## Status

- Total sample count: {row.get('total_sample_count', 0)}
- Confirmed: {row.get('confirmed_count', 0)}
- Excluded: {row.get('excluded_count', 0)}
- Needs retake: {row.get('needs_retake_count', 0)}
- Pending: {row.get('pending_count', 0)}

## Review Effort

- Mean review time sec: {row.get('mean_review_time_sec', '')}
- Median review time sec: {row.get('median_review_time_sec', '')}
- Mean moved points per image: {row.get('mean_num_points_moved', '')}
- Median moved points per image: {row.get('median_num_points_moved', '')}
- Most frequently corrected points: {', '.join(top)}

## v0.7 Candidate Guidance

Confirmed images with complete specimen IDs can enter the v0.7 candidate pool. Images with `unknown` specimen IDs, pending status, excluded status, or needs-retake status should stay out of training until resolved.
"""
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample = _read_csv(REVIEW_ROOT / "review_sample_manifest.csv")
    time_log = _read_csv(REVIEW_ROOT / "review_time_log.csv")
    correction = _read_csv(REVIEW_ROOT / "correction_summary.csv")
    summary = summarize_efficiency(sample, time_log, correction)
    by_point = correction_by_point(correction)
    by_image = correction_by_image(correction)
    p6 = p6_qc_effectiveness(sample, correction)
    excluded = excluded_cases(sample)
    summary.to_csv(OUT_DIR / "review_efficiency_summary.csv", index=False, encoding="utf-8-sig")
    by_point.to_csv(OUT_DIR / "keypoint_correction_by_point.csv", index=False, encoding="utf-8-sig")
    by_image.to_csv(OUT_DIR / "keypoint_correction_by_image.csv", index=False, encoding="utf-8-sig")
    p6.to_csv(OUT_DIR / "p6_qc_effectiveness.csv", index=False, encoding="utf-8-sig")
    excluded.to_csv(OUT_DIR / "excluded_cases.csv", index=False, encoding="utf-8-sig")
    write_readme(summary, by_point)
    counts = _status_counts(sample)
    print("Finished 5.24 post-review evaluation.")
    print(f"Confirmed: {counts['confirmed']}")
    print(f"Excluded: {counts['excluded']}")
    print(f"Pending: {counts['pending']}")
    print(f"Outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
