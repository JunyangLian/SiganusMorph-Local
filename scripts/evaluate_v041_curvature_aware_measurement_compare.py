"""Evaluate v0.4.1 curvature-aware local-normal measurement suggestions.

This script reuses the v0.4 30-image sample. It does not train models, does
not overwrite corrected labels, and does not alter manual annotations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_v031_preannotation_compare import (  # noqa: E402
    DEFAULT_MM_PER_PIXEL,
    collect_available_records,
    distance_px,
    records_from_sample_manifest,
)
from siganusmorph.hybrid_preannotation import preannotate_warped_image_hybrid_v04  # noqa: E402
from siganusmorph.image_utils import load_image_file  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "results" / "model_eval" / "v0.4.1_curvature_aware_measurement_compare_30"
V04_DIR = PROJECT_ROOT / "results" / "model_eval" / "v0.4_hybrid_heatmap_geometry_compare_30"
TARGET_KEYS = (
    "P8_body_depth_dorsal",
    "P9_body_depth_ventral",
    "P10_peduncle_depth_dorsal",
    "P11_peduncle_depth_ventral",
)


def _xy(point: Any) -> tuple[float, float] | None:
    if isinstance(point, Mapping):
        if "x" in point and "y" in point:
            return float(point["x"]), float(point["y"])
        return None
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        return float(point[0]), float(point[1])
    return None


def _point_error_mm(point: Any, manual: Any, mm_per_pixel: float) -> float:
    err_px = distance_px(point, manual)
    if err_px is None:
        return float("nan")
    return float(err_px) * float(mm_per_pixel)


def _local_point(metadata: Mapping[str, Any], key: str) -> Any:
    local = metadata.get("local_normal_measurements", {})
    if not isinstance(local, Mapping):
        return None
    body = local.get("body_depth", {}) if isinstance(local.get("body_depth", {}), Mapping) else {}
    ped = local.get("peduncle_depth", {}) if isinstance(local.get("peduncle_depth", {}), Mapping) else {}
    if key == "P8_body_depth_dorsal":
        return body.get("P8_refined")
    if key == "P9_body_depth_ventral":
        return body.get("P9_refined")
    if key == "P10_peduncle_depth_dorsal":
        return ped.get("P10_refined")
    if key == "P11_peduncle_depth_ventral":
        return ped.get("P11_refined")
    return None


def _recommend(local_median: float, hybrid_median: float, better_count: int, total: int) -> str:
    if not np.isfinite(local_median) or not np.isfinite(hybrid_median):
        return "no"
    if local_median < hybrid_median * 0.85 and better_count >= max(1, int(total * 0.6)):
        return "yes"
    return "no"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "debug_json").mkdir(parents=True, exist_ok=True)

    manifest = V04_DIR / "sample_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing sample manifest: {manifest}")

    records = collect_available_records()
    selected = records_from_sample_manifest(records, manifest)
    pd.DataFrame(
        [
            {
                "image_name": record["image_name"],
                "specimen_id": record["specimen_id"],
                "split": record["split"],
                "warped_image_path": str(record["warped_image_path"]),
                "corrected_json_path": str(record["corrected_json_path"]),
            }
            for record in selected
        ]
    ).to_csv(OUTPUT_DIR / "sample_manifest.csv", index=False, encoding="utf-8-sig")

    comparison_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    curvature_rows: list[dict[str, Any]] = []

    for index, record in enumerate(selected, start=1):
        image_name = str(record["image_name"])
        print(f"[{index}/{len(selected)}] v0.4.1 local-normal measurement {image_name}")
        payload = record["payload"]
        manual_points = payload.get("corrected_keypoints") or payload.get("keypoints") or {}
        mm_per_pixel = float(payload.get("mm_per_pixel") or DEFAULT_MM_PER_PIXEL)
        image = load_image_file(record["warped_image_path"])
        result = preannotate_warped_image_hybrid_v04(
            image,
            image_name,
            PROJECT_ROOT,
            mm_per_pixel=mm_per_pixel,
        )
        hybrid_points = result.get("keypoints", {})
        metadata = result.get("metadata", {})
        local = metadata.get("local_normal_measurements", {}) if isinstance(metadata.get("local_normal_measurements", {}), Mapping) else {}
        body = local.get("body_depth", {}) if isinstance(local.get("body_depth", {}), Mapping) else {}
        ped = local.get("peduncle_depth", {}) if isinstance(local.get("peduncle_depth", {}), Mapping) else {}
        curvature = metadata.get("curvature_qc", {}) if isinstance(metadata.get("curvature_qc", {}), Mapping) else {}

        image_debug: dict[str, Any] = {
            "image_name": image_name,
            "specimen_id": record["specimen_id"],
            "split": record["split"],
            "manual_points": {key: manual_points.get(key) for key in TARGET_KEYS},
            "hybrid_points": {key: hybrid_points.get(key) for key in TARGET_KEYS},
            "local_normal_measurements": local,
            "curvature_qc": curvature,
        }
        _write_json(OUTPUT_DIR / "debug_json" / f"{Path(image_name).stem}_debug.json", image_debug)

        for key in TARGET_KEYS:
            manual = manual_points.get(key)
            hybrid = hybrid_points.get(key)
            local_point = _local_point(metadata, key)
            hybrid_error = _point_error_mm(hybrid, manual, mm_per_pixel)
            local_error = _point_error_mm(local_point, manual, mm_per_pixel)
            point_rows.append(
                {
                    "image_name": image_name,
                    "specimen_id": record["specimen_id"],
                    "split": record["split"],
                    "keypoint_name": key,
                    "hybrid_x": (_xy(hybrid) or (np.nan, np.nan))[0],
                    "hybrid_y": (_xy(hybrid) or (np.nan, np.nan))[1],
                    "local_normal_x": (_xy(local_point) or (np.nan, np.nan))[0],
                    "local_normal_y": (_xy(local_point) or (np.nan, np.nan))[1],
                    "manual_x": (_xy(manual) or (np.nan, np.nan))[0],
                    "manual_y": (_xy(manual) or (np.nan, np.nan))[1],
                    "hybrid_error_mm": hybrid_error,
                    "local_normal_error_mm": local_error,
                    "local_normal_better": bool(np.isfinite(local_error) and np.isfinite(hybrid_error) and local_error < hybrid_error),
                }
            )

        curvature_rows.append(
            {
                "image_name": image_name,
                "specimen_id": record["specimen_id"],
                "split": record["split"],
                "curvature_index": curvature.get("curvature_index"),
                "curvature_qc_level": curvature.get("curvature_qc_level"),
                "max_axis_deviation_mm": curvature.get("max_axis_deviation_mm"),
                "axis_bend_angle_deg": curvature.get("axis_bend_angle_deg"),
                "body_depth_qc_pass": body.get("qc_pass"),
                "body_depth_review_reason": body.get("review_reason"),
                "peduncle_depth_qc_pass": ped.get("qc_pass"),
                "peduncle_depth_review_reason": ped.get("review_reason"),
                "measurements_needs_review": curvature.get("measurements_needs_review"),
                "curvature_review_points": ";".join(curvature.get("curvature_review_points", []) or []),
            }
        )

    point_df = pd.DataFrame(point_rows)
    point_df.to_csv(OUTPUT_DIR / "local_normal_measurement_errors.csv", index=False, encoding="utf-8-sig")

    for key, group in point_df.groupby("keypoint_name"):
        hybrid_median = float(group["hybrid_error_mm"].median())
        local_median = float(group["local_normal_error_mm"].median())
        hybrid_mean = float(group["hybrid_error_mm"].mean())
        local_mean = float(group["local_normal_error_mm"].mean())
        local_better = int(group["local_normal_better"].sum())
        total = int(group["local_normal_better"].notna().sum())
        comparison_rows.append(
            {
                "keypoint_name": key,
                "num_images": len(group),
                "hybrid_median_error_mm": hybrid_median,
                "local_normal_median_error_mm": local_median,
                "hybrid_mean_error_mm": hybrid_mean,
                "local_normal_mean_error_mm": local_mean,
                "local_normal_better_count": local_better,
                "hybrid_better_count": len(group) - local_better,
                "recommend_use_local_normal_as_default": _recommend(local_median, hybrid_median, local_better, total),
            }
        )

    comparison_df = pd.DataFrame(comparison_rows).sort_values("keypoint_name")
    curvature_df = pd.DataFrame(curvature_rows)
    comparison_df.to_csv(OUTPUT_DIR / "local_normal_measurement_comparison.csv", index=False, encoding="utf-8-sig")
    curvature_df.to_csv(OUTPUT_DIR / "curvature_qc_summary.csv", index=False, encoding="utf-8-sig")

    body_keys = ["P8_body_depth_dorsal", "P9_body_depth_ventral"]
    ped_keys = ["P10_peduncle_depth_dorsal", "P11_peduncle_depth_ventral"]
    body_hybrid = float(point_df[point_df["keypoint_name"].isin(body_keys)]["hybrid_error_mm"].median())
    body_local = float(point_df[point_df["keypoint_name"].isin(body_keys)]["local_normal_error_mm"].median())
    ped_hybrid = float(point_df[point_df["keypoint_name"].isin(ped_keys)]["hybrid_error_mm"].median())
    ped_local = float(point_df[point_df["keypoint_name"].isin(ped_keys)]["local_normal_error_mm"].median())
    body_recommend = "yes" if body_local < body_hybrid * 0.85 else "no"
    ped_recommend = "yes" if ped_local < ped_hybrid * 0.85 else "no"
    level_counts = curvature_df["curvature_qc_level"].fillna("unknown").value_counts().to_dict()

    readme = f"""# v0.4.1 Curvature-Aware Measurement Compare 30

This evaluation reuses the v0.4 sample manifest and does not modify corrected labels.

## Body Depth

P8/P9 hybrid median error: {body_hybrid:.3f} mm

P8/P9 local-normal median error: {body_local:.3f} mm

Recommend local-normal body depth as default: {body_recommend}

## Caudal Peduncle Depth

P10/P11 hybrid median error: {ped_hybrid:.3f} mm

P10/P11 local-normal median error: {ped_local:.3f} mm

Recommend local-normal peduncle depth as default: {ped_recommend}

## Curvature QC

Level counts: {level_counts}

The initial thresholds are straight/mild <= 1.03, moderate <= 1.08, and high > 1.08.

## Files

- `local_normal_measurement_comparison.csv`
- `local_normal_measurement_errors.csv`
- `curvature_qc_summary.csv`
- `debug_json/`
"""
    (OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    print("Finished v0.4.1 curvature-aware measurement evaluation.")
    print("Body depth:")
    print(f"P8/P9 hybrid median error: {body_hybrid:.3f} mm")
    print(f"P8/P9 local-normal median error: {body_local:.3f} mm")
    print(f"Recommend local-normal body depth as default: {body_recommend}")
    print("Peduncle depth:")
    print(f"P10/P11 hybrid median error: {ped_hybrid:.3f} mm")
    print(f"P10/P11 local-normal median error: {ped_local:.3f} mm")
    print(f"Recommend local-normal peduncle depth as default: {ped_recommend}")
    print("Curvature QC:")
    print(f"straight/mild: {level_counts.get('straight_or_mild', 0)}")
    print(f"moderate: {level_counts.get('moderate', 0)}")
    print(f"high: {level_counts.get('high', 0)}")
    print(f"Results saved to: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
