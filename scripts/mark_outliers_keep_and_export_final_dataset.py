"""Mark reviewed outliers as keep and export final analysis dataset.

This is non-destructive: it does not modify corrected_keypoints or the original
batch_measurements.csv. It updates only the outlier review CSV and creates
final_analysis_dataset copies with tracking columns.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BASE = PROJECT_ROOT / "results" / "batch_measurement_v0.6.4_stable"
OUTLIER_REVIEW = BASE / "outlier_review" / "outlier_grouped_review.csv"
BATCH_MEASUREMENTS = BASE / "batch_measurements.csv"
REVIEW_NOTE = "人工查看 preview 后未见明显点位错误，统计异常可能来自真实个体比例差异或姿态差异"


def main() -> None:
    if not OUTLIER_REVIEW.exists():
        raise FileNotFoundError(f"Missing outlier review file: {OUTLIER_REVIEW}")
    if not BATCH_MEASUREMENTS.exists():
        raise FileNotFoundError(f"Missing batch measurements file: {BATCH_MEASUREMENTS}")

    review = pd.read_csv(OUTLIER_REVIEW, dtype=str, keep_default_na=False)
    review["review_status"] = "keep"
    review["review_notes"] = REVIEW_NOTE
    review["reviewed_at"] = datetime.now().isoformat(timespec="seconds")
    review.to_csv(OUTLIER_REVIEW, index=False, encoding="utf-8-sig")

    measurements = pd.read_csv(BATCH_MEASUREMENTS, dtype=str, keep_default_na=False)
    outlier_lookup = review.set_index("image_name").to_dict("index")
    final = measurements.copy()
    final["outlier_flag"] = final["image_name"].map(lambda name: str(name) in outlier_lookup)
    final["outlier_type"] = final["image_name"].map(lambda name: outlier_lookup.get(str(name), {}).get("outlier_type", ""))
    final["outlier_affected_metrics"] = final["image_name"].map(lambda name: outlier_lookup.get(str(name), {}).get("affected_metrics", ""))
    final["outlier_metric_group"] = final["image_name"].map(lambda name: outlier_lookup.get(str(name), {}).get("metric_group", ""))
    final["outlier_review_status"] = final["image_name"].map(lambda name: outlier_lookup.get(str(name), {}).get("review_status", "keep"))
    final["outlier_review_notes"] = final["image_name"].map(lambda name: outlier_lookup.get(str(name), {}).get("review_notes", ""))
    final["include_in_final_analysis"] = True
    final["final_analysis_exclude_reason"] = ""

    csv_path = BASE / "final_analysis_dataset.csv"
    xlsx_path = BASE / "final_analysis_dataset.xlsx"
    final.to_csv(csv_path, index=False, encoding="utf-8-sig")
    final.to_excel(xlsx_path, index=False)

    summary_rows = [
        {"metric": "total_rows", "value": len(final), "notes": "All rows from batch_measurements.csv copied."},
        {"metric": "included_in_final_analysis", "value": int(final["include_in_final_analysis"].astype(bool).sum()), "notes": ""},
        {"metric": "outlier_flag_true", "value": int(final["outlier_flag"].astype(bool).sum()), "notes": "Reviewed outliers retained with tracking fields."},
        {"metric": "outlier_review_keep", "value": int((final["outlier_review_status"] == "keep").sum()), "notes": ""},
        {"metric": "outlier_review_note", "value": REVIEW_NOTE, "notes": ""},
    ]
    for group, count in final.loc[final["outlier_flag"].astype(bool), "outlier_metric_group"].value_counts().items():
        summary_rows.append({"metric": f"outlier_group_{group}", "value": int(count), "notes": ""})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(BASE / "final_analysis_summary.csv", index=False, encoding="utf-8-sig")

    readme = f"""# Final Analysis Dataset - v0.6.4 Stable

Generated: {datetime.now().isoformat(timespec='seconds')}

## Inputs

- Batch measurements: `{BATCH_MEASUREMENTS}`
- Outlier review: `{OUTLIER_REVIEW}`

## Rule Applied

All reviewed outliers were marked:

- `review_status = keep`
- `review_notes = {REVIEW_NOTE}`

No corrected keypoints were modified.
No batch measurement values were modified.
No models were trained.

## Outputs

- `final_analysis_dataset.csv`
- `final_analysis_dataset.xlsx`
- `final_analysis_summary.csv`

## Dataset Summary

- Total rows: {len(final)}
- Included in final analysis: {int(final['include_in_final_analysis'].astype(bool).sum())}
- Outlier rows retained: {int(final['outlier_flag'].astype(bool).sum())}

Outlier rows remain traceable via:

- `outlier_flag`
- `outlier_type`
- `outlier_affected_metrics`
- `outlier_metric_group`
- `outlier_review_status`
- `outlier_review_notes`
"""
    (BASE / "README_final_dataset.md").write_text(readme, encoding="utf-8")

    print("Final analysis dataset exported.")
    print(f"Rows: {len(final)}")
    print(f"Outliers kept: {int(final['outlier_flag'].astype(bool).sum())}")
    print(csv_path)
    print(xlsx_path)


if __name__ == "__main__":
    main()
