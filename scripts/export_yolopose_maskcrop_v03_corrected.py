"""Export v0.3 YOLO-pose data from corrected_keypoints only."""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph.config import DERIVED_POINT_DEFS, KEYPOINT_DEFS  # noqa: E402
from siganusmorph.real_dataset import CLASS_ID, CLASS_NAME, corrected_annotation_dir, relpath  # noqa: E402


SOURCE_DATASET = PROJECT_ROOT / "datasets" / "siganusmorph_real_5_15_yolopose_maskcrop"
OUTPUT_DATASET = PROJECT_ROOT / "datasets" / "siganusmorph_real_5_15_yolopose_maskcrop_v0.3_corrected"
CORRECTED_DIR = corrected_annotation_dir(PROJECT_ROOT) / "keypoints"
KEY_ORDER = [f"{definition.code}_{definition.name}" for definition in KEYPOINT_DEFS]


def base_stem(image_name: str) -> str:
    return Path(image_name).stem.removesuffix("_warped")


def corrected_json_path(image_name: str) -> Path:
    return CORRECTED_DIR / f"{base_stem(image_name)}_keypoints.json"


def warped_image_path(image_name: str) -> Path:
    return PROJECT_ROOT / "data" / "real_images_warped" / f"{base_stem(image_name)}_warped.png"


def as_xy(value: Any) -> tuple[float, float] | None:
    if isinstance(value, Mapping):
        try:
            return float(value["x"]), float(value["y"])
        except Exception:
            return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return None


def load_corrected_points(path: Path) -> tuple[list[tuple[float, float]] | None, str]:
    if not path.exists():
        return None, "missing_corrected_json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload.get("corrected_keypoints") or {}
    if not isinstance(source, Mapping):
        return None, "missing_corrected_keypoints"
    points: list[tuple[float, float]] = []
    missing: list[str] = []
    for key in KEY_ORDER:
        point = as_xy(source.get(key))
        if point is None:
            missing.append(key)
        else:
            points.append(point)
    if missing:
        return None, "missing_keypoints:" + ";".join(missing)
    return points, "ok"


def expand_box_to_points(
    box: tuple[int, int, int, int],
    points: list[tuple[float, float]],
    width: int,
    height: int,
    margin_px: int = 20,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    min_x = min(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_x = max(point[0] for point in points)
    max_y = max(point[1] for point in points)
    return (
        max(0, min(x0, int(math.floor(min_x)) - margin_px)),
        max(0, min(y0, int(math.floor(min_y)) - margin_px)),
        min(width, max(x1, int(math.ceil(max_x)) + margin_px)),
        min(height, max(y1, int(math.ceil(max_y)) + margin_px)),
    )


def bbox_from_points(points: list[tuple[float, float]], width: int, height: int, padding: float = 0.05) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x0, x1 = max(0.0, min(xs)), min(float(width), max(xs))
    y0, y1 = max(0.0, min(ys)), min(float(height), max(ys))
    pad_x = (x1 - x0) * padding
    pad_y = (y1 - y0) * padding
    x0 = max(0.0, x0 - pad_x)
    x1 = min(float(width), x1 + pad_x)
    y0 = max(0.0, y0 - pad_y)
    y1 = min(float(height), y1 + pad_y)
    return ((x0 + x1) / 2 / width, (y0 + y1) / 2 / height, (x1 - x0) / width, (y1 - y0) / height)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    split_path = SOURCE_DATASET / "split_manifest.csv"
    crop_path = SOURCE_DATASET / "crop_metadata.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"source split manifest not found: {split_path}")
    if not crop_path.exists():
        raise FileNotFoundError(f"source crop metadata not found: {crop_path}")

    if OUTPUT_DATASET.exists():
        backup = OUTPUT_DATASET.with_name(f"{OUTPUT_DATASET.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        shutil.move(str(OUTPUT_DATASET), str(backup))

    for split in ("train", "val", "test"):
        (OUTPUT_DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)

    split_df = pd.read_csv(split_path, dtype=str).fillna("")
    crop_df = pd.read_csv(crop_path, dtype=str).fillna("")
    crop_by_image = {str(row["image_name"]): row for _, row in crop_df.iterrows()}

    split_rows: list[dict[str, Any]] = []
    crop_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []

    for _, row in split_df.iterrows():
        image_name = str(row["image_name"])
        specimen_id = str(row["specimen_id"])
        split = str(row["split"])
        points, status = load_corrected_points(corrected_json_path(image_name))
        if points is None:
            report_rows.append({"image_name": image_name, "specimen_id": specimen_id, "split": split, "included": False, "reason": status})
            continue

        image_path = warped_image_path(image_name)
        if not image_path.exists():
            report_rows.append({"image_name": image_name, "specimen_id": specimen_id, "split": split, "included": False, "reason": "missing_warped_image"})
            continue

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            crop_meta = crop_by_image.get(image_name)
            if crop_meta is not None:
                x0 = int(float(crop_meta["bbox_x1"]))
                y0 = int(float(crop_meta["bbox_y1"]))
                x1 = int(float(crop_meta["bbox_x2"]))
                y1 = int(float(crop_meta["bbox_y2"]))
            else:
                x0, y0, x1, y1 = 0, 0, width, height
            x0, y0, x1, y1 = expand_box_to_points((x0, y0, x1, y1), points, width, height)
            crop = image.crop((x0, y0, x1, y1))

        crop_width, crop_height = crop.size
        crop_points = [(x - x0, y - y0) for x, y in points]
        if any(x < 0 or y < 0 or x > crop_width or y > crop_height for x, y in crop_points):
            report_rows.append({"image_name": image_name, "specimen_id": specimen_id, "split": split, "included": False, "reason": "keypoints_outside_crop"})
            continue

        out_name = f"{base_stem(image_name)}_warped_maskcrop.png"
        out_image = OUTPUT_DATASET / "images" / split / out_name
        out_label = OUTPUT_DATASET / "labels" / split / f"{Path(out_name).stem}.txt"
        crop.save(out_image)
        bbox = bbox_from_points(crop_points, crop_width, crop_height)
        label_values = [str(CLASS_ID), *(f"{value:.8f}" for value in bbox)]
        for x, y in crop_points:
            label_values.extend([f"{x / crop_width:.8f}", f"{y / crop_height:.8f}", "2"])
        out_label.write_text(" ".join(label_values) + "\n", encoding="utf-8")

        split_rows.append(
            {
                "image_name": image_name,
                "specimen_id": specimen_id,
                "split": split,
                "source_type": "real",
                "warped_image_path": relpath(out_image, PROJECT_ROOT),
                "label_path": relpath(out_label, PROJECT_ROOT),
                "annotation_source": "corrected_keypoints",
            }
        )
        crop_rows.append(
            {
                "image_name": image_name,
                "specimen_id": specimen_id,
                "split": split,
                "source_warped_image_path": relpath(image_path, PROJECT_ROOT),
                "crop_image_path": relpath(out_image, PROJECT_ROOT),
                "bbox_x1": x0,
                "bbox_y1": y0,
                "bbox_x2": x1,
                "bbox_y2": y1,
                "crop_width": crop_width,
                "crop_height": crop_height,
                "annotation_source": "corrected_keypoints",
            }
        )
        report_rows.append({"image_name": image_name, "specimen_id": specimen_id, "split": split, "included": True, "reason": "ok"})

    split_manifest = pd.DataFrame(split_rows)
    split_manifest.to_csv(OUTPUT_DATASET / "split_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(crop_rows).to_csv(OUTPUT_DATASET / "crop_metadata.csv", index=False, encoding="utf-8-sig")
    write_csv(OUTPUT_DATASET / "export_report.csv", report_rows, ["image_name", "specimen_id", "split", "included", "reason"])

    summary_rows = []
    for split in ("train", "val", "test"):
        group = split_manifest[split_manifest["split"] == split] if not split_manifest.empty else pd.DataFrame()
        ids = sorted(group["specimen_id"].unique().tolist()) if not group.empty else []
        summary_rows.append({"split": split, "num_specimens": len(ids), "num_images": len(group), "specimen_ids": ";".join(ids)})
    pd.DataFrame(summary_rows).to_csv(OUTPUT_DATASET / "dataset_summary.csv", index=False, encoding="utf-8-sig")

    schema = {
        "class_id": CLASS_ID,
        "class_name": CLASS_NAME,
        "kpt_shape": [len(KEYPOINT_DEFS), 3],
        "keypoints": [{"index": index, "code": definition.code, "name": definition.name} for index, definition in enumerate(KEYPOINT_DEFS)],
        "excluded_derived_points": [f"{definition.code}_{definition.name}" for definition in DERIVED_POINT_DEFS],
        "annotation_source": "corrected_keypoints",
        "split_source": relpath(split_path, PROJECT_ROOT),
    }
    (OUTPUT_DATASET / "keypoint_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    data_yaml = f"""path: {relpath(OUTPUT_DATASET, PROJECT_ROOT)}
train: images/train
val: images/val
test: images/test

names:
  0: {CLASS_NAME}

kpt_shape: [{len(KEYPOINT_DEFS)}, 3]

flip_idx:
{chr(10).join(f"  - {i}" for i in range(len(KEYPOINT_DEFS)))}
"""
    (OUTPUT_DATASET / "data.yaml").write_text(data_yaml, encoding="utf-8")
    print(f"Exported v0.3 corrected maskcrop dataset: {relpath(OUTPUT_DATASET, PROJECT_ROOT)}")
    print(f"Images included: {len(split_rows)} / {len(split_df)}")
    print(f"Specimens included: {split_manifest['specimen_id'].nunique() if not split_manifest.empty else 0}")


if __name__ == "__main__":
    main()
