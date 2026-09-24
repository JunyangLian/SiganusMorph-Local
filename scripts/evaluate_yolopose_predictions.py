from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO-pose prediction labels against a YOLO-pose dataset split."
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--prediction-label-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--mm-per-pixel", default=0.1, type=float)
    return parser.parse_args()


def parse_label_file(path: Path, num_keypoints: int) -> list[list[float]]:
    if not path.exists():
        return []
    rows: list[list[float]] = []
    min_fields = 5 + num_keypoints * 3
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = [float(item) for item in line.split()]
        if len(values) >= min_fields:
            rows.append(values)
    return rows


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1 = box_a[0] - box_a[2] / 2
    ay1 = box_a[1] - box_a[3] / 2
    ax2 = box_a[0] + box_a[2] / 2
    ay2 = box_a[1] + box_a[3] / 2
    bx1 = box_b[0] - box_b[2] / 2
    by1 = box_b[1] - box_b[3] / 2
    bx2 = box_b[0] + box_b[2] / 2
    by2 = box_b[1] + box_b[3] / 2
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    union += max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union -= inter
    return inter / union if union else 0.0


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    prediction_label_dir = args.prediction_label_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    schema = json.loads((dataset_root / "keypoint_schema.json").read_text(encoding="utf-8"))
    keypoints = schema["keypoints"]
    keypoint_names = [f"{item['code']}_{item['name']}" for item in keypoints]
    num_keypoints = len(keypoint_names)

    split_rows = list(
        csv.DictReader((dataset_root / "split_manifest.csv").open(encoding="utf-8-sig"))
    )
    error_rows: list[dict[str, object]] = []

    for split_row in split_rows:
        if split_row["split"] != args.split:
            continue
        image_path = dataset_root.parent.parent / split_row["warped_image_path"]
        if not image_path.exists():
            image_path = Path(split_row["warped_image_path"])
        label_path = dataset_root.parent.parent / split_row["label_path"]
        if not label_path.exists():
            label_path = Path(split_row["label_path"])

        with Image.open(image_path) as image:
            width, height = image.size

        ground_truth = parse_label_file(label_path, num_keypoints)
        if not ground_truth:
            continue
        gt = ground_truth[0]

        pred_path = prediction_label_dir / f"{label_path.stem}.txt"
        predictions = parse_label_file(pred_path, num_keypoints)
        best_prediction = None
        best_iou = 0.0
        if predictions:
            best_prediction = max(
                predictions, key=lambda pred: bbox_iou(gt[1:5], pred[1:5])
            )
            best_iou = bbox_iou(gt[1:5], best_prediction[1:5])

        for index, keypoint_name in enumerate(keypoint_names):
            gt_x = gt[5 + index * 3]
            gt_y = gt[6 + index * 3]
            if best_prediction is None:
                error_rows.append(
                    {
                        "image_name": split_row["image_name"],
                        "specimen_id": split_row["specimen_id"],
                        "keypoint_name": keypoint_name,
                        "error_px": "",
                        "error_mm": "",
                        "confidence": "",
                        "matched_box_iou": 0.0,
                        "num_predictions": 0,
                    }
                )
                continue

            pred_x = best_prediction[5 + index * 3]
            pred_y = best_prediction[6 + index * 3]
            confidence = best_prediction[7 + index * 3]
            error_px = math.hypot((pred_x - gt_x) * width, (pred_y - gt_y) * height)
            error_rows.append(
                {
                    "image_name": split_row["image_name"],
                    "specimen_id": split_row["specimen_id"],
                    "keypoint_name": keypoint_name,
                    "error_px": round(error_px, 3),
                    "error_mm": round(error_px * args.mm_per_pixel, 3),
                    "confidence": round(confidence, 4),
                    "matched_box_iou": round(best_iou, 4),
                    "num_predictions": len(predictions),
                }
            )

    summary_path = output_dir / "keypoint_error_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_name",
                "specimen_id",
                "keypoint_name",
                "error_px",
                "error_mm",
                "confidence",
                "matched_box_iou",
                "num_predictions",
            ],
        )
        writer.writeheader()
        writer.writerows(error_rows)

    aggregate_rows: list[dict[str, object]] = []
    for keypoint_name in keypoint_names:
        point_rows = [
            row
            for row in error_rows
            if row["keypoint_name"] == keypoint_name and row["error_px"] != ""
        ]
        errors = sorted(float(row["error_px"]) for row in point_rows)
        confidences = [float(row["confidence"]) for row in point_rows]
        aggregate_rows.append(
            {
                "keypoint_name": keypoint_name,
                "mean_error_px": round(sum(errors) / len(errors), 3) if errors else "",
                "median_error_px": round(errors[len(errors) // 2], 3) if errors else "",
                "max_error_px": round(max(errors), 3) if errors else "",
                "mean_confidence": round(sum(confidences) / len(confidences), 4)
                if confidences
                else "",
            }
        )

    by_point_path = output_dir / "keypoint_error_by_point.csv"
    with by_point_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "keypoint_name",
                "mean_error_px",
                "median_error_px",
                "max_error_px",
                "mean_confidence",
            ],
        )
        writer.writeheader()
        writer.writerows(aggregate_rows)

    print(f"Wrote {summary_path}")
    print(f"Wrote {by_point_path}")
    print(f"Evaluated {len(error_rows)} keypoints from split={args.split}")


if __name__ == "__main__":
    main()
