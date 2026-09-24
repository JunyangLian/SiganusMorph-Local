"""Evaluate compressed-tail virtual total length without altering historical data."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from siganusmorph.compressed_tail_tl import derive_tail_length_comparison
from siganusmorph.config import KEYPOINT_DEFS
from siganusmorph.measurements import calculate_measurements
from siganusmorph.realworld_review import full_to_short_keypoints

OUT_DIR = ROOT / "results" / "model_eval" / "v0.7.0_compressed_tail_virtual_tl"
VIS_DIR = OUT_DIR / "visual_comparisons"
MM_PER_PIXEL_DEFAULT = 0.1


def _repo_path(path: str | Path | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p


def _as_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        value = float(value)
        if math.isfinite(value):
            return value
    except (TypeError, ValueError):
        pass
    return None


def keypoints_from_final_row(row: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    points: dict[str, tuple[float, float]] = {}
    for definition in KEYPOINT_DEFS:
        x = _as_float(row.get(f"{definition.code}_{definition.name}_x"))
        y = _as_float(row.get(f"{definition.code}_{definition.name}_y"))
        if x is not None and y is not None:
            points[definition.name] = (x, y)
    return points


def _point_from_mapping(mapping: Mapping[str, Any], x_key: str, y_key: str) -> tuple[float, float] | None:
    x = _as_float(mapping.get(x_key))
    y = _as_float(mapping.get(y_key))
    if x is None or y is None:
        return None
    return x, y


def _draw_star(draw: ImageDraw.ImageDraw, point: tuple[float, float], fill: tuple[int, int, int], scale: float, label: str) -> None:
    x, y = point[0] * scale, point[1] * scale
    r = 9
    draw.line((x - r, y, x + r, y), fill=fill, width=3)
    draw.line((x, y - r, x, y + r), fill=fill, width=3)
    draw.line((x - r * 0.7, y - r * 0.7, x + r * 0.7, y + r * 0.7), fill=fill, width=2)
    draw.line((x - r * 0.7, y + r * 0.7, x + r * 0.7, y - r * 0.7), fill=fill, width=2)
    draw.text((x + r + 3, y - r), label, fill=fill, font=ImageFont.load_default())


def draw_visual(row: Mapping[str, Any], image_path: Path | None) -> None:
    if image_path is None or not image_path.exists():
        return
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return
    max_w = 1100
    scale = min(1.0, max_w / max(1, img.width))
    canvas = img.resize((int(img.width * scale), int(img.height * scale)))
    draw = ImageDraw.Draw(canvas)

    def p(name: str) -> tuple[float, float] | None:
        if name == "P5":
            return _point_from_mapping(row, "P5_x", "P5_y")
        if name == "P7U":
            return _point_from_mapping(row, "P7U_x", "P7U_y")
        if name == "P7L":
            return _point_from_mapping(row, "P7L_x", "P7L_y")
        return None

    p5, p7u, p7l = p("P5"), p("P7U"), p("P7L")
    for point, color, label in ((p5, (255, 255, 255), "P5"), (p7u, (255, 180, 80), "P7U"), (p7l, (255, 180, 80), "P7L")):
        if point:
            x, y = point[0] * scale, point[1] * scale
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
            draw.text((x + 7, y - 7), label, fill=color, font=ImageFont.load_default())
    if p5 and p7u:
        draw.line((p5[0] * scale, p5[1] * scale, p7u[0] * scale, p7u[1] * scale), fill=(120, 220, 120), width=2)
    if p5 and p7l:
        draw.line((p5[0] * scale, p5[1] * scale, p7l[0] * scale, p7l[1] * scale), fill=(120, 220, 120), width=2)

    p_open = _point_from_mapping(row, "P7V_open_projection_x", "P7V_open_projection_y")
    p_comp = _point_from_mapping(row, "P7V_compressed_virtual_x", "P7V_compressed_virtual_y")
    if p_open:
        _draw_star(draw, p_open, (255, 165, 60), scale, "P7V_open")
    if p_comp:
        _draw_star(draw, p_comp, (50, 245, 80), scale, "P7V_comp")
        if p5:
            draw.line((p5[0] * scale, p5[1] * scale, p_comp[0] * scale, p_comp[1] * scale), fill=(50, 245, 80), width=3)

    label = (
        f"open={_as_float(row.get('TL_open_projection_mm')):.2f}  "
        f"compressed={_as_float(row.get('TL_compressed_virtual_mm')):.2f}  "
        f"diff={_as_float(row.get('TL_difference_mm')):.2f} mm"
    )
    draw.rectangle((8, 8, 720, 42), fill=(0, 0, 0))
    draw.text((16, 16), label, fill=(255, 255, 255), font=ImageFont.load_default())
    out_name = f"{row.get('image_name', 'image')}_tl_compare.png".replace("/", "_").replace("\\", "_")
    canvas.save(VIS_DIR / out_name)


def load_final_analysis_rows() -> list[dict[str, Any]]:
    path = ROOT / "results" / "batch_measurement_v0.6.4_stable" / "final_analysis_dataset.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    manifest_path = ROOT / "results" / "batch_measurement_v0.6.4_stable" / "batch_measurement_manifest.csv"
    image_paths: dict[str, str] = {}
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        image_paths = dict(zip(manifest["image_name"].astype(str), manifest["warped_image_path"].astype(str)))
    rows: list[dict[str, Any]] = []
    for _, src in df.iterrows():
        row = src.to_dict()
        keypoints = keypoints_from_final_row(row)
        if len(keypoints) < 16:
            continue
        mm_per_pixel = _as_float(row.get("mm_per_pixel")) or MM_PER_PIXEL_DEFAULT
        try:
            measurements = calculate_measurements(keypoints, mm_per_pixel, str(row.get("axis_mode_selected") or "auto"))
        except Exception:
            measurements = derive_tail_length_comparison(
                keypoints,
                mm_per_pixel,
                sl_mm=_as_float(row.get("SL_final_mm")),
                open_projection_endpoint={
                    "x": row.get("P7V_caudal_fin_posterior_endpoint_x"),
                    "y": row.get("P7V_caudal_fin_posterior_endpoint_y"),
                },
                open_projection_extension_px=(_as_float(row.get("caudal_extension_axis_mm")) or 0.0) / mm_per_pixel,
                open_projection_tl_mm=_as_float(row.get("TL_final_mm")),
            )
        out = {
            "image_name": row.get("image_name"),
            "specimen_id": row.get("specimen_id"),
            "subset_source": "final_analysis_96",
            "mm_per_pixel": mm_per_pixel,
            "P5_x": row.get("P5_caudal_base_midpoint_x"),
            "P5_y": row.get("P5_caudal_base_midpoint_y"),
            "P7U_x": row.get("P7U_caudal_fin_upper_tip_x"),
            "P7U_y": row.get("P7U_caudal_fin_upper_tip_y"),
            "P7L_x": row.get("P7L_caudal_fin_lower_tip_x"),
            "P7L_y": row.get("P7L_caudal_fin_lower_tip_y"),
            "historical_TL_final_mm": row.get("TL_final_mm"),
            "historical_tail_axis_source": row.get("tail_axis_source"),
            "warped_image_path": image_paths.get(str(row.get("image_name")), ""),
        }
        out.update({key: measurements.get(key) for key in OUTPUT_FIELDS})
        rows.append(out)
    return rows


def load_524_preannotation_rows() -> list[dict[str, Any]]:
    qc_path = ROOT / "results" / "realworld_review_5_24_v0.6.5" / "preannotation_qc.csv"
    if not qc_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    df = pd.read_csv(qc_path)
    for _, src in df.iterrows():
        if str(src.get("preannotation_success", "")).lower() not in {"true", "1"}:
            continue
        json_path = _repo_path(src.get("preannotation_json_path"))
        if json_path is None or not json_path.exists():
            continue
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        full = payload.get("corrected_keypoints") or payload.get("hybrid_keypoints") or payload.get("v06_keypoints") or {}
        if not isinstance(full, Mapping):
            continue
        short = full_to_short_keypoints(dict(full))
        if len(short) < 16:
            continue
        mm_per_pixel = _as_float(payload.get("mm_per_pixel")) or MM_PER_PIXEL_DEFAULT
        try:
            measurements = calculate_measurements(short, mm_per_pixel, "auto")
        except Exception:
            continue
        p5 = full.get("P5_caudal_base_midpoint", [None, None])
        p7u = full.get("P7U_caudal_fin_upper_tip", [None, None])
        p7l = full.get("P7L_caudal_fin_lower_tip", [None, None])
        out = {
            "image_name": payload.get("image_name") or src.get("image_name"),
            "specimen_id": payload.get("specimen_id") or src.get("specimen_id"),
            "subset_source": "5_24_preannotation_success",
            "mm_per_pixel": mm_per_pixel,
            "P5_x": p5[0] if isinstance(p5, list) else "",
            "P5_y": p5[1] if isinstance(p5, list) else "",
            "P7U_x": p7u[0] if isinstance(p7u, list) else "",
            "P7U_y": p7u[1] if isinstance(p7u, list) else "",
            "P7L_x": p7l[0] if isinstance(p7l, list) else "",
            "P7L_y": p7l[1] if isinstance(p7l, list) else "",
            "historical_TL_final_mm": "",
            "historical_tail_axis_source": ((payload.get("preannotation_metadata") or {}).get("tail_axis_source") if isinstance(payload.get("preannotation_metadata"), Mapping) else ""),
            "warped_image_path": payload.get("warped_image_path", ""),
        }
        out.update({key: measurements.get(key) for key in OUTPUT_FIELDS})
        rows.append(out)
    return rows


OUTPUT_FIELDS = [
    "P7V_open_projection_x",
    "P7V_open_projection_y",
    "P7V_compressed_virtual_x",
    "P7V_compressed_virtual_y",
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
    "compressed_tail_tl_valid_qc_pass",
    "compressed_tail_tl_qc_pass",
    "compressed_vs_projection_difference_flag",
    "compressed_vs_projection_difference_reason",
    "compressed_tail_tl_review_required",
    "compressed_tail_tl_review_reason",
    "tail_axis_source_compressed",
]


def angle_group(angle: Any) -> str:
    value = _as_float(angle)
    if value is None:
        return "unknown"
    if value <= 10:
        return "0-10"
    if value <= 20:
        return "10-20"
    if value <= 30:
        return "20-30"
    return ">30"


def write_audit_doc() -> None:
    text = """# Current TL Formula Audit

## Current historical formula

The current historical `P7V` / `TL_final_mm` implementation is projection-based.

- `siganusmorph.axis_utils.derive_caudal_posterior_endpoint()` derives `P7V` by projecting `P7U` and `P7L` onto a tail axis and choosing the larger projection:
  `extension = max(dot(P7U - P5, tail_axis), dot(P7L - P5, tail_axis))`.
- `siganusmorph.hybrid_point_selector.derive_hybrid_p7v()` uses the same projection formula for v0.4/v0.6 preannotation, with a newer tail axis source: `P5 -> midpoint(P7U, P7L)` first, then lobe bisector / C4 fallback.
- `siganusmorph.measurements.calculate_measurements()` stores historical `TL_final_mm` as `SL_final_mm + caudal_extension_axis_mm`, where `caudal_extension_axis_mm` is the selected projection distance.

## Formula class

Current historical TL method: `orthogonal_projection`.

The older generic measurement helper in `axis_utils` may use `P5 -> P6` as its tail-axis source, while the v0.4+ hybrid preannotation path avoids P6 and uses lobe geometry. In both cases the caudal-tip length contribution itself is an axis projection.

## Systematic underestimation risk

For an opened caudal lobe with physical length `R` and angle `theta` from the tail axis, projection TL uses `R * cos(theta)`. A manual measuring board often compresses the tail lobes toward the body axis, closer to preserving the physical lobe radius `R`. Therefore projection TL can underestimate compressed-tail virtual TL, especially when `tail_open_angle_deg` is large.

## New fields

This evaluation adds `TL_compressed_virtual_mm` and related QC fields. It does not overwrite historical `TL_final_mm`.
"""
    (OUT_DIR / "current_tl_formula_audit.md").write_text(text, encoding="utf-8")


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.copy()
    for col in ("TL_difference_mm", "TL_difference_percent"):
        numeric[col] = pd.to_numeric(numeric[col], errors="coerce")
    rows = []
    for metric in ("TL_difference_mm", "TL_difference_percent"):
        series = numeric[metric].dropna()
        rows.append(
            {
                "metric": metric,
                "num_valid": int(series.count()),
                "mean": float(series.mean()) if len(series) else math.nan,
                "median": float(series.median()) if len(series) else math.nan,
                "max": float(series.max()) if len(series) else math.nan,
            }
        )
    grouped = numeric.groupby("tail_angle_group", dropna=False)
    for group, group_df in grouped:
        series = pd.to_numeric(group_df["TL_difference_mm"], errors="coerce").dropna()
        pct = pd.to_numeric(group_df["TL_difference_percent"], errors="coerce").dropna()
        rows.append(
            {
                "metric": f"tail_angle_group_{group}",
                "num_valid": int(series.count()),
                "mean": float(series.mean()) if len(series) else math.nan,
                "median": float(series.median()) if len(series) else math.nan,
                "max": float(series.max()) if len(series) else math.nan,
                "percent_mean": float(pct.mean()) if len(pct) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def write_readme(df: pd.DataFrame, summary: pd.DataFrame, qc_df: pd.DataFrame) -> None:
    diff = pd.to_numeric(df["TL_difference_mm"], errors="coerce").dropna()
    pct = pd.to_numeric(df["TL_difference_percent"], errors="coerce").dropna()
    qc_failed = len(qc_df)
    recommendation = "yes" if len(diff) and diff.median() > 0.5 and qc_failed / max(len(df), 1) < 0.25 else "review_more"
    text = f"""# v0.7.0 Compressed-Tail Virtual TL Evaluation

## Formula audit

Current historical `TL_final_mm` is projection-based (`orthogonal_projection`). It is preserved unchanged for compatibility.

## Difference summary

- Images evaluated: `{len(df)}`
- TL difference mean / median / max: `{diff.mean():.3f}` / `{diff.median():.3f}` / `{diff.max():.3f}` mm
- TL difference percent mean / median / max: `{pct.mean():.3f}` / `{pct.median():.3f}` / `{pct.max():.3f}` %
- QC failed / review cases: `{qc_failed}`

## Interpretation

`TL_compressed_virtual_mm` is closer to the measuring-board practice of gently closing caudal lobes to the body axis because it preserves the longer lobe radius from P5. `TL_open_projection_mm` remains useful as an image-state/open-tail reference.

## Recommendations

- Use `TL_compressed_virtual_mm` as future default TL: `{recommendation}`
- Keep `TL_open_projection_mm` as reference: `yes`
- Keep FL and SL outputs: `yes`
- Historical results unchanged: `yes`

See `tl_difference_summary.csv` for tail-angle groups and `tl_qc_cases.csv` for review cases.
"""
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    write_audit_doc()
    rows = load_final_analysis_rows() + load_524_preannotation_rows()
    if not rows:
        raise SystemExit("No rows available for v0.7.0 TL evaluation.")
    df = pd.DataFrame(rows)
    numeric_output_fields = [field for field in OUTPUT_FIELDS if field not in {"compressed_tail_tl_qc_pass", "compressed_tail_tl_review_reason", "tail_axis_source_compressed"}]
    for col in numeric_output_fields:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["tail_angle_group"] = df["tail_open_angle_deg"].apply(angle_group)
    df.to_csv(OUT_DIR / "tl_method_comparison.csv", index=False, encoding="utf-8-sig")

    summary = make_summary(df)
    summary.to_csv(OUT_DIR / "tl_difference_summary.csv", index=False, encoding="utf-8-sig")

    qc_mask = (
        df["compressed_tail_tl_qc_pass"].astype(str).str.lower().isin(["false", "0"])
        | (pd.to_numeric(df["TL_difference_mm"], errors="coerce").abs() > 5.0)
        | (pd.to_numeric(df["TL_difference_percent"], errors="coerce").abs() > 3.0)
        | (pd.to_numeric(df["tail_lobe_length_asymmetry_ratio"], errors="coerce") > 0.20)
    )
    qc_df = df.loc[qc_mask].copy()
    qc_df.to_csv(OUT_DIR / "tl_qc_cases.csv", index=False, encoding="utf-8-sig")

    # Generate visual comparisons for all rows with available images; skip silently
    # if image files are unavailable because this evaluation must not mutate inputs.
    for _, row in df.iterrows():
        image_path = _repo_path(row.get("warped_image_path"))
        draw_visual(row.to_dict(), image_path)

    write_readme(df, summary, qc_df)
    diff = pd.to_numeric(df["TL_difference_mm"], errors="coerce").dropna()
    pct = pd.to_numeric(df["TL_difference_percent"], errors="coerce").dropna()
    print("Finished v0.7.0 compressed-tail virtual TL evaluation.")
    print("Current historical TL method: orthogonal_projection")
    print(f"difference mean / median / max: {diff.mean():.3f} / {diff.median():.3f} / {diff.max():.3f} mm")
    print(f"difference percent mean / median / max: {pct.mean():.3f} / {pct.median():.3f} / {pct.max():.3f} %")
    print(f"QC failed cases: {len(qc_df)}")
    print(f"Results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
