"""Build combined ground truth and crop metadata for v0.5 evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph.config import KEYPOINT_DEFS  # noqa: E402


KEYPOINT_KEYS = [f"{definition.code}_{definition.name}" for definition in KEYPOINT_DEFS]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "siganusmorph_heatmap_unet_v0.5_candidate"
PLANNING_DIR = PROJECT_ROOT / "results" / "v0.5_dataset_planning"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "model_eval" / "heatmap_unet_v0.5_predictions"
GT_DIR = OUTPUT_ROOT / "ground_truth_keypoints"
CROP_METADATA = OUTPUT_ROOT / "crop_metadata.csv"
SPLIT_MANIFEST = DATASET_ROOT / "split_manifest.csv"
OLD_SPLIT = PROJECT_ROOT / "datasets" / "siganusmorph_heatmap_unet_v0.1" / "split_manifest.csv"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _xy(value: Any) -> list[float] | None:
    if isinstance(value, Mapping) and "x" in value and "y" in value:
        return [float(value["x"]), float(value["y"])]
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return [float(value[0]), float(value[1])]
    return None


def _extract_points(payload: Mapping[str, Any]) -> dict[str, list[float]]:
    raw = payload.get("corrected_keypoints")
    if not isinstance(raw, Mapping):
        raw = payload.get("keypoints")
    if not isinstance(raw, Mapping):
        return {}
    out = {}
    for key in KEYPOINT_KEYS:
        xy = _xy(raw.get(key))
        if xy is not None:
            out[key] = xy
    return out


def _project_path(value: str | Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    GT_DIR.mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(SPLIT_MANIFEST, dtype=str, keep_default_na=False)
    manifest = pd.read_csv(PLANNING_DIR / "v05_candidate_label_manifest.csv", dtype=str, keep_default_na=False)
    manifest = manifest.loc[manifest["include_in_v05_candidate"].astype(str).str.lower() == "true"].copy()
    labels = _read_json(DATASET_ROOT / "labels.json")
    sample_by_image = {str(sample["image_name"]): sample for sample in labels["samples"]}
    old_split = pd.read_csv(OLD_SPLIT, dtype=str, keep_default_na=False) if OLD_SPLIT.exists() else pd.DataFrame()
    old_by_image = {str(row["image_name"]): row.to_dict() for _, row in old_split.iterrows()}
    manifest_by_image_source = {
        (str(row["image_name"]), str(row["source_set"])): row
        for _, row in manifest.iterrows()
    }
    crop_rows = []
    for _, row in split.iterrows():
        image_name = str(row["image_name"])
        source_set = str(row["source_set"])
        manifest_row = manifest_by_image_source[(image_name, source_set)]
        payload = _read_json(_project_path(manifest_row["source_json_path"]))
        points = _extract_points(payload)
        out_payload = {
            "image_name": image_name,
            "specimen_id": str(row["specimen_id"]),
            "mm_per_pixel": float(payload.get("mm_per_pixel") or 0.1),
            "corrected_keypoints": points,
            "source_set": source_set,
            "source_json_path": str(manifest_row["source_json_path"]),
        }
        _write_json(GT_DIR / f"{Path(image_name).stem}_keypoints.json", out_payload)
        if source_set == "original_corrected":
            old = old_by_image[image_name]
            x1 = float(old["bbox_x1"])
            y1 = float(old["bbox_y1"])
            x2 = float(old["bbox_x2"])
            y2 = float(old["bbox_y2"])
        else:
            crop_box = sample_by_image[image_name].get("crop_box", [])
            x1, y1, x2, y2 = [float(v) for v in crop_box]
        crop_rows.append(
            {
                "image_name": image_name,
                "specimen_id": str(row["specimen_id"]),
                "split": str(row["split"]),
                "source_set": source_set,
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "crop_width": x2 - x1,
                "crop_height": y2 - y1,
                "source_warped_image_path": str(row.get("warped_image_path", "")),
            }
        )
    pd.DataFrame(crop_rows).to_csv(CROP_METADATA, index=False, encoding="utf-8-sig")
    print(f"Wrote {GT_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {CROP_METADATA.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
