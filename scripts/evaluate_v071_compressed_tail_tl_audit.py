"""v0.7.1 compressed-tail TL audit with split QC and validation templates."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_v070_compressed_tail_virtual_tl as v070  # noqa: E402


OUT_DIR = ROOT / "results" / "model_eval" / "v0.7.1_compressed_tail_tl_audit"
VALIDATION_DIR = ROOT / "results" / "manual_compressed_tl_validation"


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _angle_group(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(value):
        return "unknown"
    if value <= 10:
        return "0-10"
    if value <= 20:
        return "10-20"
    if value <= 30:
        return "20-30"
    return ">30"


def load_revised_rows() -> pd.DataFrame:
    rows = v070.load_final_analysis_rows() + v070.load_524_preannotation_rows()
    if not rows:
        raise RuntimeError("No rows available for compressed-tail TL audit.")
    df = pd.DataFrame(rows)
    numeric_cols = [
        "TL_open_projection_mm",
        "TL_compressed_virtual_mm",
        "TL_difference_mm",
        "TL_difference_percent",
        "RU_mm",
        "RL_mm",
        "Rmax_mm",
        "upper_lobe_angle_to_axis_deg",
        "lower_lobe_angle_to_axis_deg",
        "inter_lobe_open_angle_deg",
        "upper_lobe_angle_deg",
        "lower_lobe_angle_deg",
        "tail_open_angle_deg",
        "tail_lobe_length_asymmetry_ratio",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "upper_lobe_angle_to_axis_deg" not in df or df["upper_lobe_angle_to_axis_deg"].isna().all():
        df["upper_lobe_angle_to_axis_deg"] = df.get("upper_lobe_angle_deg")
    if "lower_lobe_angle_to_axis_deg" not in df or df["lower_lobe_angle_to_axis_deg"].isna().all():
        df["lower_lobe_angle_to_axis_deg"] = df.get("lower_lobe_angle_deg")
    if "inter_lobe_open_angle_deg" not in df or df["inter_lobe_open_angle_deg"].isna().all():
        df["inter_lobe_open_angle_deg"] = df.get("tail_open_angle_deg")
    df["upper_lobe_angle_group"] = df["upper_lobe_angle_to_axis_deg"].apply(_angle_group)
    df["lower_lobe_angle_group"] = df["lower_lobe_angle_to_axis_deg"].apply(_angle_group)
    df["inter_lobe_angle_group"] = df["inter_lobe_open_angle_deg"].apply(_angle_group)
    df["compressed_tail_tl_valid_qc_pass"] = _bool_series(df.get("compressed_tail_tl_valid_qc_pass", df.get("compressed_tail_tl_qc_pass", pd.Series(False, index=df.index))))
    df["compressed_vs_projection_difference_flag"] = (
        df["TL_difference_mm"].abs().gt(5.0) | df["TL_difference_percent"].abs().gt(3.0)
    )
    df["compressed_tail_tl_review_required"] = ~df["compressed_tail_tl_valid_qc_pass"]
    df["compressed_tail_tl_review_reason"] = df.get("compressed_tail_tl_review_reason", "").fillna("")
    df["compressed_vs_projection_difference_reason"] = ""
    df.loc[df["TL_difference_mm"].abs().gt(5.0), "compressed_vs_projection_difference_reason"] += "TL_difference_gt_5mm"
    pct_mask = df["TL_difference_percent"].abs().gt(3.0)
    df.loc[pct_mask & df["compressed_vs_projection_difference_reason"].ne(""), "compressed_vs_projection_difference_reason"] += ";"
    df.loc[pct_mask, "compressed_vs_projection_difference_reason"] += "TL_difference_percent_gt_3"
    return df


def write_angle_audit(df: pd.DataFrame) -> None:
    text = """# Tail Angle Definition Audit

v0.7.1 separates three angle concepts:

1. `upper_lobe_angle_to_axis_deg`
   `angle(P7U - P5, tail_axis_unit)`.

2. `lower_lobe_angle_to_axis_deg`
   `angle(P7L - P5, tail_axis_unit)`.

3. `inter_lobe_open_angle_deg`
   `angle(P7U - P5, P7L - P5)`.

Only the single-lobe angles to the tail axis directly explain the projection relation:
`R_projection = R * cos(theta)`.

The inter-lobe angle describes how open the fork appears overall, but it should not be
substituted directly as `theta` in the projection equation.
"""
    (OUT_DIR / "tail_angle_definition_audit.md").write_text(text, encoding="utf-8")
    rows = []
    for angle_col, group_col in [
        ("upper_lobe_angle_to_axis_deg", "upper_lobe_angle_group"),
        ("lower_lobe_angle_to_axis_deg", "lower_lobe_angle_group"),
        ("inter_lobe_open_angle_deg", "inter_lobe_angle_group"),
    ]:
        for group, sub in df.groupby(group_col, dropna=False):
            values = pd.to_numeric(sub[angle_col], errors="coerce").dropna()
            rows.append(
                {
                    "angle_type": angle_col,
                    "angle_group": group,
                    "count": int(values.count()),
                    "mean_deg": float(values.mean()) if len(values) else math.nan,
                    "median_deg": float(values.median()) if len(values) else math.nan,
                    "min_deg": float(values.min()) if len(values) else math.nan,
                    "max_deg": float(values.max()) if len(values) else math.nan,
                }
            )
    pd.DataFrame(rows).to_csv(OUT_DIR / "tail_angle_distribution.csv", index=False, encoding="utf-8-sig")


def write_qc_outputs(df: pd.DataFrame) -> None:
    valid_count = int(df["compressed_tail_tl_valid_qc_pass"].sum())
    diff_count = int(df["compressed_vs_projection_difference_flag"].sum())
    review_count = int(df["compressed_tail_tl_review_required"].sum())
    pd.DataFrame(
        [
            {"metric": "total_images", "count": len(df), "notes": ""},
            {"metric": "valid_compressed_tail_TL", "count": valid_count, "notes": "Input geometry/data passed validity QC."},
            {"metric": "invalid_compressed_tail_TL", "count": len(df) - valid_count, "notes": "Missing/invalid inputs or lobe asymmetry issue."},
        ]
    ).to_csv(OUT_DIR / "tl_validity_qc_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"metric": "total_images", "count": len(df), "notes": ""},
            {"metric": "difference_flag_true", "count": diff_count, "notes": "Large method difference only; not an algorithm failure by itself."},
            {"metric": "difference_flag_false", "count": len(df) - diff_count, "notes": ""},
        ]
    ).to_csv(OUT_DIR / "tl_difference_flag_summary.csv", index=False, encoding="utf-8-sig")
    review_cols = [
        "image_name",
        "specimen_id",
        "subset_source",
        "TL_open_projection_mm",
        "TL_compressed_virtual_mm",
        "TL_difference_mm",
        "TL_difference_percent",
        "upper_lobe_angle_to_axis_deg",
        "lower_lobe_angle_to_axis_deg",
        "inter_lobe_open_angle_deg",
        "tail_lobe_length_asymmetry_ratio",
        "compressed_tail_tl_valid_qc_pass",
        "compressed_vs_projection_difference_flag",
        "compressed_tail_tl_review_required",
        "compressed_tail_tl_review_reason",
        "compressed_vs_projection_difference_reason",
    ]
    df.loc[df["compressed_tail_tl_review_required"], review_cols].to_csv(
        OUT_DIR / "tl_review_cases_revised.csv", index=False, encoding="utf-8-sig"
    )


def select_validation_sample(df: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    selected: list[int] = []
    work = df.copy()
    work["abs_diff"] = work["TL_difference_mm"].abs()
    work["asym"] = work["tail_lobe_length_asymmetry_ratio"].fillna(0)
    # Cover low/medium/high method differences.
    for _, sub in work.groupby(pd.qcut(work["abs_diff"].rank(method="first"), q=3, labels=False, duplicates="drop")):
        for idx in sub.sort_values(["specimen_id", "abs_diff"], ascending=[True, False]).head(5).index:
            if idx not in selected:
                selected.append(idx)
    # Add high asymmetry and both available angle groups.
    for idx in work.sort_values("asym", ascending=False).head(6).index:
        if idx not in selected:
            selected.append(idx)
    for group in ["20-30", ">30", "unknown"]:
        sub = work[work["inter_lobe_angle_group"].eq(group)]
        for idx in sub.sort_values("abs_diff", ascending=False).head(4).index:
            if idx not in selected:
                selected.append(idx)
    # Fill by unique specimens first.
    used_specimens = set(work.loc[selected, "specimen_id"].astype(str)) if selected else set()
    for idx, row in work.sort_values("abs_diff", ascending=False).iterrows():
        if len(selected) >= n:
            break
        specimen = str(row.get("specimen_id", ""))
        if specimen not in used_specimens and idx not in selected:
            selected.append(idx)
            used_specimens.add(specimen)
    for idx in work.sort_values("abs_diff", ascending=False).index:
        if len(selected) >= n:
            break
        if idx not in selected:
            selected.append(idx)
    return work.loc[selected[:n]].copy()


def write_validation_template(df: pd.DataFrame) -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    sample = select_validation_sample(df, n=30)
    template_cols = [
        "image_name",
        "specimen_id",
        "source_batch",
        "manual_compressed_TL_mm",
        "manual_open_projection_TL_mm",
        "manual_measurement_repeat_1_mm",
        "manual_measurement_repeat_2_mm",
        "TL_open_projection_mm",
        "TL_compressed_virtual_mm",
        "TL_difference_mm",
        "TL_difference_percent",
        "upper_lobe_angle_to_axis_deg",
        "lower_lobe_angle_to_axis_deg",
        "inter_lobe_open_angle_deg",
        "tail_lobe_length_asymmetry_ratio",
        "manual_notes",
    ]
    out = pd.DataFrame()
    out["image_name"] = sample["image_name"]
    out["specimen_id"] = sample["specimen_id"]
    out["source_batch"] = sample["subset_source"]
    for col in template_cols:
        if col not in out:
            out[col] = sample[col] if col in sample else ""
    out = out[template_cols]
    out.to_csv(VALIDATION_DIR / "manual_compressed_tl_template.csv", index=False, encoding="utf-8-sig")
    out.to_excel(VALIDATION_DIR / "manual_compressed_tl_template.xlsx", index=False)
    manifest = sample[
        [
            "image_name",
            "specimen_id",
            "subset_source",
            "TL_difference_mm",
            "TL_difference_percent",
            "upper_lobe_angle_to_axis_deg",
            "lower_lobe_angle_to_axis_deg",
            "inter_lobe_open_angle_deg",
            "tail_lobe_length_asymmetry_ratio",
            "compressed_vs_projection_difference_flag",
            "compressed_tail_tl_valid_qc_pass",
            "warped_image_path",
        ]
    ].rename(columns={"subset_source": "source_batch"})
    manifest.to_csv(VALIDATION_DIR / "manual_compressed_tl_validation_manifest.csv", index=False, encoding="utf-8-sig")
    readme = f"""# Manual Compressed-Tail TL Validation

This folder contains a {len(out)}-image template for manual validation of compressed-tail virtual TL.

Fill in:
- `manual_compressed_TL_mm`
- optional open-projection TL and repeat measurements
- `manual_notes`

Then run:

```bash
python scripts/analyze_manual_compressed_tl_validation.py --input results/manual_compressed_tl_validation/manual_compressed_tl_template.xlsx
```

The template includes both single-lobe-to-axis angles and the inter-lobe opening angle.
Use single-lobe angles for projection interpretation.
"""
    (VALIDATION_DIR / "README.md").write_text(readme, encoding="utf-8")


def write_readme(df: pd.DataFrame) -> None:
    valid_count = int(df["compressed_tail_tl_valid_qc_pass"].sum())
    diff_count = int(df["compressed_vs_projection_difference_flag"].sum())
    review_count = int(df["compressed_tail_tl_review_required"].sum())
    text = f"""# v0.7.1 Compressed-Tail TL Audit

## What changed from v0.7.0

QC is split into:

1. `compressed_tail_tl_valid_qc_pass`: input geometry/data validity.
2. `compressed_vs_projection_difference_flag`: large difference between historical projection TL and compressed-tail candidate TL.
3. `compressed_tail_tl_review_required`: true manual-review cases. A large method difference alone is not treated as an algorithm failure.

## Counts

- Total images: `{len(df)}`
- Valid compressed-tail TL: `{valid_count} / {len(df)}`
- Difference flag: `{diff_count} / {len(df)}`
- Manual-review-required: `{review_count} / {len(df)}`

## Angle definitions

See `tail_angle_definition_audit.md`. Only `upper_lobe_angle_to_axis_deg` and `lower_lobe_angle_to_axis_deg` should be used for `R_projection = R * cos(theta)` interpretation. `inter_lobe_open_angle_deg` is the fork opening angle, not the single-lobe projection angle.

## Recommendation

Do not replace `TL_final_mm` yet. Use `TL_compressed_virtual_mm` as a parallel candidate and complete manual measuring-board validation first.
"""
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")
    summary = {
        "total_images": int(len(df)),
        "valid_compressed_tail_tl": valid_count,
        "difference_flag": diff_count,
        "manual_review_required": review_count,
        "manual_validation_template_generated": True,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_revised_rows()
    df.to_csv(OUT_DIR / "tl_method_comparison_revised.csv", index=False, encoding="utf-8-sig")
    write_angle_audit(df)
    write_qc_outputs(df)
    write_validation_template(df)
    write_readme(df)
    print("Finished v0.7.1 compressed-tail TL audit.")
    print("Tail-angle definition audited: yes")
    print("QC split completed: yes")
    print(f"Valid compressed-tail TL: {int(df['compressed_tail_tl_valid_qc_pass'].sum())} / {len(df)}")
    print(f"Difference flag: {int(df['compressed_vs_projection_difference_flag'].sum())} / {len(df)}")
    print(f"Manual-review-required: {int(df['compressed_tail_tl_review_required'].sum())} / {len(df)}")
    print("Manual validation template generated: yes")
    print("Historical results unchanged: yes")


if __name__ == "__main__":
    main()
