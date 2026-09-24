"""Analyze manual compressed-tail TL validation measurements."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "manual_compressed_tl_validation" / "manual_compressed_tl_template.xlsx"
DEFAULT_OUT = ROOT / "results" / "manual_compressed_tl_validation" / "analysis"


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _pearson(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    mask = a.notna() & b.notna()
    if mask.sum() < 2:
        return math.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def _icc_two_way_single(method: pd.Series, manual: pd.Series) -> float:
    data = pd.concat(
        [pd.to_numeric(method, errors="coerce"), pd.to_numeric(manual, errors="coerce")],
        axis=1,
    ).dropna()
    if len(data) < 2:
        return math.nan
    values = data.to_numpy(dtype=float)
    n, k = values.shape
    grand_mean = values.mean()
    row_means = values.mean(axis=1)
    col_means = values.mean(axis=0)
    ss_rows = k * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    ss_total = np.sum((values - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols
    ms_rows = ss_rows / max(n - 1, 1)
    ms_cols = ss_cols / max(k - 1, 1)
    ms_error = ss_error / max((n - 1) * (k - 1), 1)
    denom = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / max(n, 1)
    if abs(denom) <= 1e-12:
        return math.nan
    return float((ms_rows - ms_error) / denom)


def _metrics(df: pd.DataFrame, method_col: str, manual_col: str) -> dict[str, float | str | int]:
    method = pd.to_numeric(df[method_col], errors="coerce")
    manual = pd.to_numeric(df[manual_col], errors="coerce")
    mask = method.notna() & manual.notna()
    diff = method[mask] - manual[mask]
    abs_diff = diff.abs()
    pct = abs_diff / manual[mask].abs().replace(0, np.nan) * 100.0
    bias = float(diff.mean()) if len(diff) else math.nan
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else math.nan
    return {
        "method": method_col,
        "manual_reference": manual_col,
        "N": int(mask.sum()),
        "mean_difference": bias,
        "MAE": float(abs_diff.mean()) if len(abs_diff) else math.nan,
        "RMSE": float(np.sqrt(np.mean(diff**2))) if len(diff) else math.nan,
        "MAPE": float(pct.mean()) if len(pct) else math.nan,
        "Pearson_r": _pearson(method, manual),
        "ICC": _icc_two_way_single(method, manual),
        "Bland_Altman_bias": bias,
        "Bland_Altman_LOA_lower": bias - 1.96 * sd if math.isfinite(sd) else math.nan,
        "Bland_Altman_LOA_upper": bias + 1.96 * sd if math.isfinite(sd) else math.nan,
    }


def _agreement_by_image(df: pd.DataFrame, manual_col: str, methods: Iterable[str]) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        manual = pd.to_numeric(pd.Series([row.get(manual_col)]), errors="coerce").iloc[0]
        for method in methods:
            value = pd.to_numeric(pd.Series([row.get(method)]), errors="coerce").iloc[0]
            diff = value - manual if pd.notna(value) and pd.notna(manual) else np.nan
            rows.append(
                {
                    "image_name": row.get("image_name"),
                    "specimen_id": row.get("specimen_id"),
                    "method": method,
                    "manual_compressed_TL_mm": manual,
                    "method_TL_mm": value,
                    "difference_mm": diff,
                    "absolute_difference_mm": abs(diff) if pd.notna(diff) else np.nan,
                    "percent_error": abs(diff) / abs(manual) * 100.0 if pd.notna(diff) and pd.notna(manual) and manual != 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _write_plots(bland: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir = out_dir / "agreement_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for method, sub in bland.groupby("method"):
        sub = sub.dropna(subset=["mean_mm", "difference_mm"])
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(sub["mean_mm"], sub["difference_mm"], s=24)
        bias = sub["difference_mm"].mean()
        sd = sub["difference_mm"].std(ddof=1)
        ax.axhline(bias, color="red", label="bias")
        if pd.notna(sd):
            ax.axhline(bias + 1.96 * sd, color="gray", linestyle="--")
            ax.axhline(bias - 1.96 * sd, color="gray", linestyle="--")
        ax.set_title(f"Bland-Altman: {method}")
        ax.set_xlabel("Mean of method and manual (mm)")
        ax.set_ylabel("Method - manual (mm)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"bland_altman_{method}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Manual validation CSV/XLSX.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT), help="Output analysis directory.")
    args = parser.parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _read_table(input_path)
    manual_col = "manual_compressed_TL_mm"
    methods = ["TL_open_projection_mm", "TL_compressed_virtual_mm"]
    summary = pd.DataFrame([_metrics(df, method, manual_col) for method in methods])
    by_image = _agreement_by_image(df, manual_col, methods)
    bland = by_image.copy()
    bland["mean_mm"] = (pd.to_numeric(bland["manual_compressed_TL_mm"], errors="coerce") + pd.to_numeric(bland["method_TL_mm"], errors="coerce")) / 2.0

    summary.to_csv(out_dir / "agreement_summary.csv", index=False, encoding="utf-8-sig")
    by_image.to_csv(out_dir / "agreement_by_image.csv", index=False, encoding="utf-8-sig")
    bland.to_csv(out_dir / "bland_altman_data.csv", index=False, encoding="utf-8-sig")
    _write_plots(bland, out_dir)
    readme = f"""# Manual Compressed-Tail TL Validation Analysis

Input: `{input_path}`

Outputs:
- `agreement_summary.csv`
- `agreement_by_image.csv`
- `bland_altman_data.csv`
- `agreement_plots/` when matplotlib is available

Reference column: `{manual_col}`.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Manual compressed-tail TL validation analysis saved to: {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"analysis_failed: {exc}", file=sys.stderr)
        raise
