"""Export a mask-derived fish-crop YOLO-pose dataset using the v0.1 split."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph.config import DERIVED_POINT_DEFS, KEYPOINT_DEFS  # noqa: E402
from siganusmorph.real_dataset import CLASS_ID, CLASS_NAME, relpath  # noqa: E402


SOURCE_DATASET = PROJECT_ROOT / "datasets" / "siganusmorph_real_5_15_yolopose"
MASK_REPORT = PROJECT_ROOT / "results" / "model_eval" / "v0.1_mask_correction" / "mask_correction_report.csv"
ANNOTATION_DIR = PROJECT_ROOT / "results" / "real_annotation_5_15" / "keypoints"
OUTPUT_DATASET = PROJECT_ROOT / "datasets" / "siganusmorph_real_5_15_yolopose_maskcrop"
KEY_ORDER = [f"{definition.code}_{definition.name}" for definition in KEYPOINT_DEFS]


def warped_path(image_name: str) -> Path:
    return PROJECT_ROOT / "data" / "real_images_warped" / f"{Path(image_name).stem}_warped.png"


def annotation_path(image_name: str) -> Path:
    return ANNOTATION_DIR / f"{Path(image_name).stem}_keypoints.json"


def load_keypoints(image_name: str) -> list[tuple[float, float]]:
    data = json.loads(annotation_path(image_name).read_text(encoding="utf-8"))
    points = data["keypoints"]
    return [(float(points[key][0]), float(points[key][1])) for key in KEY_ORDER]


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
    split_df = pd.read_csv(SOURCE_DATASET / "split_manifest.csv", dtype=str).fillna("")
    report_df = pd.read_csv(MASK_REPORT, dtype=str).fillna("")
    report_by_image = {row["image_name"]: row for _, row in report_df.iterrows()}

    if OUTPUT_DATASET.exists():
        import shutil

        backup = OUTPUT_DATASET.with_name(f"{OUTPUT_DATASET.name}_backup")
        if backup.exists():
            shutil.rmtree(backup)
        shutil.move(str(OUTPUT_DATASET), str(backup))

    for split in ("train", "val", "test"):
        (OUTPUT_DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)

    split_rows: list[dict[str, Any]] = []
    crop_rows: list[dict[str, Any]] = []
    for _, row in split_df.iterrows():
        image_name = str(row["image_name"])
        specimen_id = str(row["specimen_id"])
        split = str(row["split"])
        report = report_by_image[image_name]
        image_path = warped_path(image_name)
        points = load_keypoints(image_name)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            x0 = int(float(report["fish_bbox_x1"]))
            y0 = int(float(report["fish_bbox_y1"]))
            x1 = int(float(report["fish_bbox_x2"]))
            y1 = int(float(report["fish_bbox_y2"]))
            min_x = min(point[0] for point in points)
            min_y = min(point[1] for point in points)
            max_x = max(point[0] for point in points)
            max_y = max(point[1] for point in points)
            x0 = max(0, min(x0, int(min_x) - 20))
            y0 = max(0, min(y0, int(min_y) - 20))
            x1 = min(width, max(x1, int(max_x) + 20))
            y1 = min(height, max(y1, int(max_y) + 20))
            crop = image.crop((x0, y0, x1, y1))
        crop_width, crop_height = crop.size
        crop_points = [(x - x0, y - y0) for x, y in points]
        if any(x < 0 or y < 0 or x > crop_width or y > crop_height for x, y in crop_points):
            raise RuntimeError(f"Keypoints outside mask crop for {image_name}")

        out_name = f"{Path(image_name).stem}_warped_maskcrop.png"
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
            }
        )
        crop_rows.append(
            {
                "image_name": image_name,
                "specimen_id": specimen_id,
                "split": split,
                "warped_image_path": relpath(image_path, PROJECT_ROOT),
                "crop_image_path": relpath(out_image, PROJECT_ROOT),
                "bbox_x1": x0,
                "bbox_y1": y0,
                "bbox_x2": x1,
                "bbox_y2": y1,
                "crop_width": crop_width,
                "crop_height": crop_height,
                "segmentation_success": report["segmentation_success"],
                "segmentation_quality": report["segmentation_quality"],
            }
        )

    split_manifest = pd.DataFrame(split_rows)
    split_manifest.to_csv(OUTPUT_DATASET / "split_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(crop_rows).to_csv(OUTPUT_DATASET / "crop_metadata.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for split in ("train", "val", "test"):
        group = split_manifest[split_manifest["split"] == split]
        ids = sorted(group["specimen_id"].unique().tolist())
        summary_rows.append({"split": split, "num_specimens": len(ids), "num_images": len(group), "specimen_ids": ";".join(ids)})
    pd.DataFrame(summary_rows).to_csv(OUTPUT_DATASET / "dataset_summary.csv", index=False, encoding="utf-8-sig")

    schema = {
        "class_id": CLASS_ID,
        "class_name": CLASS_NAME,
        "kpt_shape": [len(KEYPOINT_DEFS), 3],
        "keypoints": [{"index": index, "code": definition.code, "name": definition.name} for index, definition in enumerate(KEYPOINT_DEFS)],
        "excluded_derived_points": [f"{definition.code}_{definition.name}" for definition in DERIVED_POINT_DEFS],
        "crop_source": "fish_mask_bbox",
        "split_source": relpath(SOURCE_DATASET / "split_manifest.csv", PROJECT_ROOT),
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
    print(f"Exported maskcrop dataset: {relpath(OUTPUT_DATASET, PROJECT_ROOT)}")
    print(f"Images: {len(split_rows)}")
    print(f"Specimens: {split_manifest['specimen_id'].nunique()}")


if __name__ == "__main__":
    main()

