"""Stable v0.6.4 batch morphometric measurement export.

Reads confirmed corrected annotations, computes v0.6.4 dual-axis measurement
fields, writes measurement/QC reports, and does not modify saved labels.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph.config import KEYPOINT_DEFS, RESULT_COLUMNS
from siganusmorph.dual_axis_measurement import MEASUREMENT_AXIS_MODEL, compute_measurement_axis_metadata
from siganusmorph.image_utils import ensure_rgb, load_image_file
from siganusmorph.measurements import build_result_row, calculate_measurements
from siganusmorph.visualization import draw_enhanced_preannotation_overlay, draw_keypoints_and_measurements


OUT = PROJECT_ROOT / "results" / "batch_measurement_v0.6.4_stable"
PLANNING_MANIFEST = PROJECT_ROOT / "results" / "v0.5_dataset_planning" / "v05_candidate_label_manifest.csv"
E2E_README = PROJECT_ROOT / "results" / "e2e_regression_v0.6.4" / "README.md"
V06_SAME_SET = PROJECT_ROOT / "results" / "model_eval" / "v0.6_keypointwise_hybrid_selector" / "same_set_comparison"
V063 = PROJECT_ROOT / "results" / "model_eval" / "v0.6.3_dual_axis_measurement"
MM_PER_PIXEL = 0.1

FULL_TO_SHORT = {f"{kp.code}_{kp.name}": kp.name for kp in KEYPOINT_DEFS}
SHORT_TO_FULL = {kp.name: f"{kp.code}_{kp.name}" for kp in KEYPOINT_DEFS}
FULL_KEYPOINTS = tuple(FULL_TO_SHORT.keys())

CORE_METRICS = [
    "TL_final_mm",
    "SL_final_mm",
    "body_depth_mm",
    "head_length_mm",
    "snout_length_mm",
    "caudal_peduncle_length_mm",
    "caudal_peduncle_depth_mm",
    "TL_curve_selected_mm",
    "SL_curve_selected_mm",
    "curvature_index_selected",
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip() or value == "nan":
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _xy(value: Any) -> list[float] | None:
    if isinstance(value, Mapping):
        if "x" in value and "y" in value:
            return [float(value["x"]), float(value["y"])]
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return [float(value[0]), float(value[1])]
    return None


def _point_distance(a: Any, b: Any) -> float:
    aa = _xy(a)
    bb = _xy(b)
    if aa is None or bb is None:
        return float("nan")
    return math.hypot(aa[0] - bb[0], aa[1] - bb[1])


def _corrected_keypoints(payload: Mapping[str, Any]) -> dict[str, list[float]]:
    points = payload.get("corrected_keypoints") or payload.get("keypoints") or {}
    out: dict[str, list[float]] = {}
    if not isinstance(points, Mapping):
        return out
    for key in FULL_KEYPOINTS:
        point = _xy(points.get(key))
        if point is not None:
            out[key] = point
    return out


def _full_to_short(full_points: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for full, short in FULL_TO_SHORT.items():
        point = _xy(full_points.get(full))
        if point is not None:
            out[short] = {"x": point[0], "y": point[1]}
    return out


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _as_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _load_label_manifest() -> pd.DataFrame:
    if not PLANNING_MANIFEST.exists():
        raise FileNotFoundError(f"Missing planning manifest: {PLANNING_MANIFEST}")
    frame = pd.read_csv(PLANNING_MANIFEST, dtype=str, keep_default_na=False)
    rows = []
    for record in frame.to_dict("records"):
        json_path = _resolve_path(record.get("source_json_path", ""))
        include = str(record.get("include_in_v05_candidate", "")).lower() == "true"
        exclude_reason = str(record.get("exclude_reason", ""))
        annotation_status = ""
        has_16 = False
        if json_path and json_path.exists():
            try:
                payload = _load_json(json_path)
                annotation_status = str(payload.get("annotation_status", record.get("review_status", "")))
                has_16 = len(_corrected_keypoints(payload)) == len(FULL_KEYPOINTS)
            except Exception as exc:  # noqa: BLE001
                annotation_status = "json_read_failed"
                exclude_reason = f"json_read_failed:{exc}"
        else:
            include = False
            exclude_reason = exclude_reason or "missing_json"
        if record.get("image_name") == "real_037.png":
            include = False
            exclude_reason = exclude_reason or "failed_preannotation"
        if annotation_status != "corrected_and_confirmed":
            include = False
            exclude_reason = exclude_reason or "not_corrected_and_confirmed"
        if not has_16:
            include = False
            exclude_reason = exclude_reason or "missing_16_keypoints"
        rows.append(
            {
                "image_name": record.get("image_name", ""),
                "specimen_id": record.get("specimen_id", ""),
                "source_set": record.get("source_set", ""),
                "corrected_json_path": "" if json_path is None else str(json_path),
                "warped_image_path": record.get("warped_image_path", ""),
                "crop_image_path": record.get("crop_image_path", ""),
                "annotation_status": annotation_status,
                "include_in_batch_measurement": bool(include),
                "exclude_reason": "" if include else exclude_reason,
                "notes": record.get("notes", ""),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUT / "batch_measurement_manifest.csv", index=False, encoding="utf-8-sig")
    return manifest


def _compute_one(record: Mapping[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str]:
    json_path = _resolve_path(record.get("corrected_json_path", ""))
    warped_path = _resolve_path(record.get("warped_image_path", ""))
    if json_path is None or not json_path.exists():
        return None, None, "missing_json"
    if warped_path is None or not warped_path.exists():
        return None, None, "missing_warped_image"
    payload = _load_json(json_path)
    full_points = _corrected_keypoints(payload)
    if len(full_points) != len(FULL_KEYPOINTS):
        return None, None, "missing_16_keypoints"
    image = load_image_file(warped_path)
    short_points = _full_to_short(full_points)
    measurements = calculate_measurements(short_points, MM_PER_PIXEL, axis_mode_selected="auto")
    axis_payload = compute_measurement_axis_metadata(
        image,
        full_points,
        mm_per_pixel=MM_PER_PIXEL,
        measurement_axis_mode=MEASUREMENT_AXIS_MODEL,
    )
    measurements.update(axis_payload.get("measurement_axis_row_fields", {}))
    # Stable baseline keeps model_axis as selected/compatibility axis.
    measurements["TL_curve_mm"] = measurements.get("TL_curve_selected_mm", measurements.get("TL_curve_mm"))
    measurements["SL_curve_mm"] = measurements.get("SL_curve_selected_mm", measurements.get("SL_curve_mm"))
    measurements["curvature_index"] = measurements.get("curvature_index_selected", measurements.get("curvature_index"))
    p7v = measurements.get("derived_points", {}).get("P7V_caudal_fin_posterior_endpoint")
    row = build_result_row(
        image_name=str(record.get("image_name", "")),
        specimen_id=str(record.get("specimen_id", "")),
        scale_method="board_mm_per_pixel",
        mm_per_pixel=MM_PER_PIXEL,
        keypoints=short_points,
        measurements=measurements,
        needs_review=bool(measurements.get("measurements_needs_review", False)),
        notes="v0.6.4_stable_batch_measurement",
        source_type="real",
    )
    row.update(
        {
            "source_set": record.get("source_set", ""),
            "head_length_mm": row.get("head_length_straight_mm", ""),
            "snout_length_mm": row.get("snout_length_straight_mm", ""),
            "caudal_peduncle_length_mm": row.get("caudal_peduncle_length_straight_mm", ""),
            "p7v_valid": bool(p7v),
            "preannotation_reliability_level": "confirmed_manual",
            "curvature_qc_level": axis_payload.get("measurement_axes", {}).get("model_axis", {}).get("curvature_qc_level", ""),
            "high_curvature_review_required": str(axis_payload.get("measurement_axes", {}).get("model_axis", {}).get("curvature_qc_level", "")).lower() == "high",
            "recommended_action": "review_measurements" if measurements.get("measurements_needs_review") else "ready_for_analysis",
        }
    )
    metadata = {
        "image_name": record.get("image_name", ""),
        "specimen_id": record.get("specimen_id", ""),
        "warped_image_path": str(warped_path),
        "full_keypoints": full_points,
        "short_keypoints": short_points,
        "measurements": measurements,
        "measurement_axes": axis_payload.get("measurement_axes", {}),
        "measurement_axis_row_fields": axis_payload.get("measurement_axis_row_fields", {}),
    }
    return row, metadata, ""


def _iqr_outlier_mask(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    if not math.isfinite(float(iqr)) or iqr <= 0:
        return pd.Series(False, index=series.index)
    return (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)


def _z3_outlier_mask(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    sd = values.std(ddof=1)
    if not math.isfinite(float(sd)) or sd <= 0:
        return pd.Series(False, index=series.index)
    return ((values - values.mean()).abs() / sd) > 3.0


def _qc_summary(measurements: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in CORE_METRICS:
        values = pd.to_numeric(measurements.get(metric, pd.Series(dtype=float)), errors="coerce")
        rows.append(
            {
                "metric": metric,
                "num_valid": int(values.notna().sum()),
                "num_missing": int(values.isna().sum()),
                "mean": values.mean(),
                "sd": values.std(ddof=1),
                "median": values.median(),
                "min": values.min(),
                "max": values.max(),
                "num_outliers_iqr": int(_iqr_outlier_mask(values).sum()),
                "num_outliers_z3": int(_z3_outlier_mask(values).sum()),
                "notes": "IQR is sensitive because curvature values are tightly clustered; use >1.03 as practical review threshold."
                if metric == "curvature_index_selected"
                else "",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "measurement_qc_summary.csv", index=False, encoding="utf-8-sig")
    return frame


def _ratios(measurements: pd.DataFrame) -> pd.DataFrame:
    def div(a: str, b: str) -> pd.Series:
        aa = pd.to_numeric(measurements.get(a, pd.Series(dtype=float)), errors="coerce")
        bb = pd.to_numeric(measurements.get(b, pd.Series(dtype=float)), errors="coerce")
        return aa / bb.replace(0, np.nan)

    ratios = pd.DataFrame(
        {
            "image_name": measurements["image_name"],
            "specimen_id": measurements["specimen_id"],
            "TL_SL_ratio": div("TL_final_mm", "SL_final_mm"),
            "body_depth_SL_ratio": div("body_depth_mm", "SL_final_mm"),
            "head_length_SL_ratio": div("head_length_mm", "SL_final_mm"),
            "snout_length_head_length_ratio": div("snout_length_mm", "head_length_mm"),
            "caudal_peduncle_length_SL_ratio": div("caudal_peduncle_length_mm", "SL_final_mm"),
            "caudal_peduncle_depth_SL_ratio": div("caudal_peduncle_depth_mm", "SL_final_mm"),
            "curvature_index_selected": pd.to_numeric(measurements.get("curvature_index_selected", pd.Series(dtype=float)), errors="coerce"),
            "measurements_needs_review": measurements.get("measurements_needs_review", ""),
        }
    )
    ratios.to_csv(OUT / "batch_measurement_ratios.csv", index=False, encoding="utf-8-sig")
    return ratios


def _outlier_cases(measurements: pd.DataFrame, ratios: pd.DataFrame) -> pd.DataFrame:
    metric_outlier: dict[str, set[str]] = {}
    for metric in CORE_METRICS:
        values = pd.to_numeric(measurements.get(metric, pd.Series(dtype=float)), errors="coerce")
        mask = _iqr_outlier_mask(values) | _z3_outlier_mask(values)
        metric_outlier[metric] = set(measurements.loc[mask, "image_name"].astype(str))

    rows = []
    ratio_lookup = ratios.set_index("image_name").to_dict("index")
    for record in measurements.to_dict("records"):
        image_name = str(record["image_name"])
        issues: list[str] = []
        affected: list[str] = []
        values: dict[str, Any] = {}
        if str(record.get("p7v_valid", "")).lower() not in {"true", "1"}:
            issues.append("invalid_p7v")
        if str(record.get("measurements_needs_review", "")).lower() in {"true", "1"}:
            issues.append("measurements_need_review")
        if str(record.get("dual_axis_disagreement", "")).lower() in {"true", "1"}:
            issues.append("dual_axis_disagreement")
        if str(record.get("high_curvature_review_required", "")).lower() in {"true", "1"}:
            issues.append("high_curvature_review_required")
        if pd.isna(pd.to_numeric(pd.Series([record.get("TL_final_mm")]), errors="coerce").iloc[0]):
            issues.append("missing_TL_final")
        if pd.isna(pd.to_numeric(pd.Series([record.get("SL_final_mm")]), errors="coerce").iloc[0]):
            issues.append("missing_SL_final")
        ratio = ratio_lookup.get(image_name, {})
        checks = {
            "TL_SL_ratio": (1.00, 1.35),
            "body_depth_SL_ratio": (0.15, 0.65),
            "caudal_peduncle_depth_SL_ratio": (0.02, 0.18),
        }
        for key, (lo, hi) in checks.items():
            value = _as_float(ratio.get(key))
            if math.isfinite(value) and not (lo <= value <= hi):
                issues.append(f"abnormal_{key}")
                affected.append(key)
                values[key] = round(value, 4)
        for metric, images in metric_outlier.items():
            if metric == "curvature_index_selected":
                value = _as_float(record.get(metric))
                if math.isfinite(value) and value > 1.03:
                    issues.append("curvature_review_threshold")
                    affected.append(metric)
                    values[metric] = record.get(metric)
                continue
            if image_name in images:
                issues.append("core_metric_outlier")
                affected.append(metric)
                values[metric] = record.get(metric)
        if issues:
            rows.append(
                {
                    "image_name": image_name,
                    "specimen_id": record.get("specimen_id", ""),
                    "source_set": record.get("source_set", ""),
                    "outlier_type": ";".join(dict.fromkeys(issues)),
                    "affected_metrics": ";".join(dict.fromkeys(affected)),
                    "metric_values": json.dumps(_json_safe(values), ensure_ascii=False),
                    "recommended_action": "manual_measurement_review",
                    "notes": record.get("measurement_axis_review_reason", ""),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "measurement_outlier_cases.csv", index=False, encoding="utf-8-sig")
    return frame


def _draw_warning_banner(image: np.ndarray, text: str, color: tuple[int, int, int]) -> np.ndarray:
    canvas = Image.fromarray(ensure_rgb(image))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((10, 10, min(canvas.size[0] - 10, 720), 55), fill=(0, 0, 0))
    draw.text((20, 22), text, fill=color)
    return np.asarray(canvas)


def _previews(metadata_rows: list[dict[str, Any]], outlier_images: set[str]) -> None:
    preview_dir = OUT / "measurement_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for meta in metadata_rows:
        image_name = str(meta["image_name"])
        image = load_image_file(Path(meta["warped_image_path"]))
        measurements = dict(meta["measurements"])
        keypoints = meta["short_keypoints"]
        base = draw_keypoints_and_measurements(
            image,
            keypoints,
            measurements,
            image_name=image_name,
            specimen_id=str(meta.get("specimen_id", "")),
        )
        overlay_meta = {
            "corrected_keypoints": meta["full_keypoints"],
            "measurement_axes": meta["measurement_axes"],
            "qc_results": {
                "p7v_valid": measurements.get("p7v_valid", True),
                "p7v_suggestion": measurements.get("derived_points", {}).get("P7V_caudal_fin_posterior_endpoint"),
            },
            "hybrid_qc_results": {
                "p7v_valid": measurements.get("p7v_valid", True),
                "p7v": {"point": measurements.get("derived_points", {}).get("P7V_caudal_fin_posterior_endpoint")},
            },
            "show_model_axis": True,
            "show_body_midline_axis": True,
            "show_dual_axis_comparison": True,
            "show_selected_measurement_axis_only": False,
            "show_qc_warnings": True,
        }
        preview = draw_enhanced_preannotation_overlay(base, overlay_meta)
        if image_name in outlier_images:
            preview = _draw_warning_banner(preview, "WARNING: measurement QC review recommended", (255, 220, 0))
        Image.fromarray(ensure_rgb(preview)).save(preview_dir / f"{Path(image_name).stem}_measurement_preview.png")


def _version_snapshot(included: int, excluded: int) -> None:
    text = f"""# VERSION SNAPSHOT - v0.6.4_stable_batch_measurement

Generated: {datetime.now().isoformat(timespec='seconds')}

## Frozen Baseline

1. Default preannotation mode: `v0.4.1 hybrid heatmap + geometry`
2. Experimental preannotation mode: `v0.6 keypoint-wise hybrid selector`
3. Default measurement axis: `model_axis`
4. Optional measurement axes: `body_midline_axis`, `auto_qc_gated`

## Model Weights

- `models/siganusmorph_heatmap_unet_v0.1/preannotation_candidate.pt`
- `models/siganusmorph_heatmap_unet_v0.5/preannotation_candidate.pt`

## Validation References

- e2e regression: {E2E_README}
- real-world review: 29 confirmed / 1 excluded
- strict same-set comparison: {V06_SAME_SET}
- dual-axis comparison: {V063}

## Batch Scope

- Included images: {included}
- Excluded manifest rows: {excluded}

This batch run does not train models and does not modify corrected labels.
"""
    (OUT / "VERSION_SNAPSHOT.md").write_text(text, encoding="utf-8")


def _readme(measurements: pd.DataFrame, manifest: pd.DataFrame, qc: pd.DataFrame, outliers: pd.DataFrame) -> None:
    included = int(manifest["include_in_batch_measurement"].sum())
    excluded = int((~manifest["include_in_batch_measurement"]).sum())
    p7v_valid = int(measurements["p7v_valid"].astype(str).str.lower().isin(["true", "1"]).sum())
    needs_review = int(measurements["measurements_needs_review"].astype(str).str.lower().isin(["true", "1"]).sum())
    dual_disagree = int(measurements["dual_axis_disagreement"].astype(str).str.lower().isin(["true", "1"]).sum())
    excluded_reasons = manifest.loc[~manifest["include_in_batch_measurement"], "exclude_reason"].value_counts().to_dict()
    stats_table = qc.to_markdown(index=False)
    major_outliers = outliers.head(15).to_markdown(index=False) if not outliers.empty else "No outlier cases flagged."
    text = f"""# v0.6.4 Stable Batch Measurement

## Summary

- Version: `v0.6.4_stable_batch_measurement`
- Included images: {included}
- Excluded rows: {excluded}
- P7V valid: {p7v_valid} / {included}
- Measurements needing review: {needs_review}
- Dual-axis disagreement: {dual_disagree}
- Outlier cases: {len(outliers)}

Excluded reasons:

```json
{json.dumps(excluded_reasons, ensure_ascii=False, indent=2)}
```

## Measurement QC Summary

{stats_table}

## Main Outlier Cases

{major_outliers}

## Interpretation

The batch export uses confirmed corrected_keypoints only. The stable selected
measurement axis is `model_axis`, while body-midline axis fields are exported
for QC and comparison. No labels or model defaults were changed.

## Recommended Next Step

Proceed to manual measurement agreement validation / formal analysis. Review
rows in `measurement_outlier_cases.csv` before using measurements in final
statistics.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = _load_label_manifest()
    rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    for record in manifest[manifest["include_in_batch_measurement"]].to_dict("records"):
        row, metadata, error = _compute_one(record)
        if error:
            manifest.loc[manifest["image_name"].eq(record["image_name"]), "include_in_batch_measurement"] = False
            manifest.loc[manifest["image_name"].eq(record["image_name"]), "exclude_reason"] = error
            continue
        assert row is not None and metadata is not None
        rows.append(row)
        metadata_rows.append(metadata)

    measurements = pd.DataFrame(rows)
    # Keep configured columns first, then extra reporting fields.
    for col in RESULT_COLUMNS:
        if col not in measurements.columns:
            measurements[col] = ""
    extra_cols = [col for col in measurements.columns if col not in RESULT_COLUMNS]
    measurements = measurements[list(RESULT_COLUMNS) + extra_cols]
    measurements.to_csv(OUT / "batch_measurements.csv", index=False, encoding="utf-8-sig")
    measurements.to_excel(OUT / "batch_measurements.xlsx", index=False)
    manifest.to_csv(OUT / "batch_measurement_manifest.csv", index=False, encoding="utf-8-sig")

    qc = _qc_summary(measurements)
    ratios = _ratios(measurements)
    outliers = _outlier_cases(measurements, ratios)
    _previews(metadata_rows, set(outliers["image_name"].astype(str)) if not outliers.empty else set())
    _version_snapshot(int(manifest["include_in_batch_measurement"].sum()), int((~manifest["include_in_batch_measurement"]).sum()))
    _readme(measurements, manifest, qc, outliers)

    print("Finished v0.6.4 stable batch measurement.")
    print("")
    print(f"Included images: {int(manifest['include_in_batch_measurement'].sum())}")
    print(f"Excluded images: {int((~manifest['include_in_batch_measurement']).sum())}")
    print("")
    p7v_valid = int(measurements["p7v_valid"].astype(str).str.lower().isin(["true", "1"]).sum())
    needs_review = int(measurements["measurements_needs_review"].astype(str).str.lower().isin(["true", "1"]).sum())
    dual_disagree = int(measurements["dual_axis_disagreement"].astype(str).str.lower().isin(["true", "1"]).sum())
    print(f"P7V valid: {p7v_valid} / {len(measurements)}")
    print(f"Measurements needing review: {needs_review}")
    print(f"Dual-axis disagreement: {dual_disagree}")
    print(f"Outlier cases: {len(outliers)}")
    print("")
    print("Outputs:")
    print("batch_measurements.csv")
    print("batch_measurements.xlsx")
    print("measurement_qc_summary.csv")
    print("measurement_outlier_cases.csv")
    print("batch_measurement_ratios.csv")
    print("measurement_previews/")
    print("")
    print("Recommended next step:")
    print("manual measurement agreement validation / formal analysis")


if __name__ == "__main__":
    main()
