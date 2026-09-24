"""Export confirmed 5.24 corrected labels and measurements.

Run after manual review.  The script reads only confirmed corrected JSON files
from the 5.24 batch and writes final_corrected_dataset CSV/XLSX. It does not
modify labels.
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
from siganusmorph.dual_axis_measurement import MEASUREMENT_AXIS_AUTO_QC, compute_measurement_axis_metadata  # noqa: E402
from siganusmorph.image_utils import load_image_file  # noqa: E402
from siganusmorph.measurements import calculate_measurements  # noqa: E402
from siganusmorph.realworld_review import full_to_short_keypoints, xy_from_payload  # noqa: E402


REVIEW_ROOT = PROJECT_ROOT / "results" / "realworld_review_5_24_v0.6.5"
KEYPOINT_KEYS = [f"{definition.code}_{definition.name}" for definition in KEYPOINT_DEFS]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_sample() -> pd.DataFrame:
    path = REVIEW_ROOT / "review_sample_manifest.csv"
    return pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() else pd.DataFrame()


def _axis_fields(payload: dict[str, Any], image: Any | None, corrected: dict[str, Any], mm_per_pixel: float) -> dict[str, Any]:
    axes = payload.get("measurement_axes", {}) if isinstance(payload.get("measurement_axes", {}), dict) else {}
    if axes:
        out = dict(axes)
        for axis in ("model_axis", "body_midline_axis"):
            data = axes.get(axis, {}) if isinstance(axes.get(axis, {}), dict) else {}
            prefix = "model_axis" if axis == "model_axis" else "body_midline_axis"
            out[f"TL_curve_{prefix}_mm"] = data.get("TL_curve_mm", "")
            out[f"SL_curve_{prefix}_mm"] = data.get("SL_curve_mm", "")
            out[f"curvature_index_{prefix}"] = data.get("curvature_index", "")
            out[f"axis_smoothness_{prefix}"] = data.get("axis_smoothness", "")
        return out
    if image is None:
        return {}
    try:
        axis_payload = compute_measurement_axis_metadata(
            image,
            corrected,
            mm_per_pixel=mm_per_pixel,
            measurement_axis_mode=MEASUREMENT_AXIS_AUTO_QC,
        )
        return dict(axis_payload.get("measurement_axis_row_fields", {}) or {})
    except Exception:
        return {}


def main() -> None:
    sample = _read_sample()
    sample_by_image = {str(row["image_name"]): row for _, row in sample.iterrows()} if not sample.empty else {}
    rows: list[dict[str, Any]] = []
    for json_path in sorted((REVIEW_ROOT / "corrected_keypoints").glob("*_corrected.json")):
        payload = _read_json(json_path)
        if payload.get("annotation_status") != "corrected_and_confirmed":
            continue
        image_name = str(payload.get("image_name", json_path.stem.replace("_corrected", ".png")))
        row = sample_by_image.get(image_name, {})
        corrected = payload.get("corrected_keypoints", {}) if isinstance(payload.get("corrected_keypoints", {}), dict) else {}
        if len([key for key in KEYPOINT_KEYS if key in corrected]) < len(KEYPOINT_KEYS):
            continue
        mm_per_pixel = float(payload.get("mm_per_pixel", 0.1) or 0.1)
        short = full_to_short_keypoints(corrected)
        measurements = calculate_measurements(short, mm_per_pixel) if len(short) == len(KEYPOINT_DEFS) else {}
        image = None
        warped_path = str((payload.get("preannotation", {}) or {}).get("warped_image_path", "") or row.get("warped_image_path", ""))
        if warped_path:
            path = Path(warped_path)
            image_path = path if path.is_absolute() else PROJECT_ROOT / path
            if image_path.exists():
                image = load_image_file(image_path)
        axis = _axis_fields(payload, image, corrected, mm_per_pixel)
        out: dict[str, Any] = {
            "image_name": image_name,
            "specimen_id": payload.get("specimen_id", row.get("specimen_id", "")),
            "source_batch": row.get("source_batch", "5_24_new_fish"),
            "corrected_json_path": str(json_path),
            "annotation_status": payload.get("annotation_status", ""),
        }
        for key in KEYPOINT_KEYS:
            xy = xy_from_payload(corrected.get(key))
            out[f"{key}_x"] = xy[0] if xy else ""
            out[f"{key}_y"] = xy[1] if xy else ""
        derived = payload.get("preannotation", {}).get("derived_points", {}) if isinstance(payload.get("preannotation", {}), dict) else {}
        p7v = derived.get("P7V_caudal_fin_posterior_endpoint") or payload.get("derived_points", {}).get("P7V_caudal_fin_posterior_endpoint")
        p7v_xy = xy_from_payload(p7v)
        out["P7V_x"] = p7v_xy[0] if p7v_xy else ""
        out["P7V_y"] = p7v_xy[1] if p7v_xy else ""
        out.update(measurements)
        out.update(axis)
        out["selected_measurement_axis"] = axis.get("selected_measurement_axis", payload.get("measurement_axes", {}).get("selected_measurement_axis", ""))
        tail_qc = payload.get("tail_qc", {}) if isinstance(payload.get("tail_qc", {}), dict) else {}
        out["P6_geometry_qc_pass"] = tail_qc.get("P6_geometry_qc_pass", "")
        out["measurements_needs_review"] = axis.get("measurements_needs_review", payload.get("measurement_axes", {}).get("measurements_needs_review", ""))
        out["recommended_action"] = "manual_review" if str(out["measurements_needs_review"]).lower() == "true" else "include_in_analysis"
        rows.append(out)
    df = pd.DataFrame(rows)
    csv_path = REVIEW_ROOT / "final_corrected_dataset.csv"
    xlsx_path = REVIEW_ROOT / "final_corrected_dataset.xlsx"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(xlsx_path) as writer:
        df.to_excel(writer, index=False, sheet_name="final_corrected")
    print(f"Exported {len(df)} confirmed 5.24 rows.")
    print(csv_path)
    print(xlsx_path)


if __name__ == "__main__":
    main()
