"""Evaluate v0.2/v0.3 pose weights on the v0.3 corrected test split."""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph.config import KEYPOINT_DEFS  # noqa: E402
from siganusmorph.real_dataset import relpath  # noqa: E402


DATASET_ROOT = PROJECT_ROOT / "datasets" / "siganusmorph_real_5_15_yolopose_maskcrop_v0.3_corrected"
MODEL_ROOT = PROJECT_ROOT / "models" / "siganusmorph_yolopose_v0.3_corrected_real_5_15"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "model_eval" / "v0.3_corrected_real_5_15"
PREDICTION_DIR = OUTPUT_ROOT / "test_predictions"
MM_PER_PIXEL = 0.1
KEY_NAMES = [f"{definition.code}_{definition.name}" for definition in KEYPOINT_DEFS]
FOCUS_KEYS = [
    "P1_snout_tip",
    "P3_operculum_posterior",
    "P5_caudal_base_midpoint",
    "P7U_caudal_fin_upper_tip",
    "P7L_caudal_fin_lower_tip",
    "P10_peduncle_depth_dorsal",
    "P11_peduncle_depth_ventral",
]


MODELS = [
    {
        "model_name": "v0.2_preannotation_candidate",
        "input_type": "maskcrop",
        "path": PROJECT_ROOT / "models" / "siganusmorph_yolopose_v0.2_maskcrop_real_5_15" / "preannotation_candidate.pt",
    },
    {
        "model_name": "v0.3_best",
        "input_type": "maskcrop_corrected",
        "path": MODEL_ROOT / "weights" / "best.pt",
    },
    {
        "model_name": "v0.3_last",
        "input_type": "maskcrop_corrected",
        "path": MODEL_ROOT / "weights" / "last.pt",
    },
]


def parse_gt_label(path: Path, width: int, height: int) -> list[tuple[float, float]]:
    values = [float(item) for item in path.read_text(encoding="utf-8").strip().split()]
    points = []
    for index in range(len(KEY_NAMES)):
        x = values[5 + index * 3] * width
        y = values[6 + index * 3] * height
        points.append((x, y))
    return points


def predict_points(model: Any, image_path: Path) -> tuple[list[tuple[float, float]] | None, list[float]]:
    results = model.predict(str(image_path), imgsz=1024, conf=0.05, max_det=1, verbose=False)
    if not results:
        return None, []
    result = results[0]
    if result.keypoints is None or len(result.keypoints) == 0:
        return None, []
    if result.boxes is not None and len(result.boxes) > 0 and hasattr(result.boxes, "conf"):
        import numpy as np

        det_index = int(np.argmax(result.boxes.conf.cpu().numpy()))
    else:
        det_index = 0
    xy = result.keypoints.xy.cpu().numpy()[det_index]
    if getattr(result.keypoints, "conf", None) is not None:
        conf = result.keypoints.conf.cpu().numpy()[det_index]
    else:
        conf = [1.0] * len(KEY_NAMES)
    points = [(float(xy[index][0]), float(xy[index][1])) for index in range(min(len(KEY_NAMES), len(xy)))]
    confidences = [float(conf[index]) if index < len(conf) else 1.0 for index in range(len(points))]
    if len(points) != len(KEY_NAMES):
        return None, []
    return points, confidences


def draw_prediction(
    image_path: Path,
    gt: list[tuple[float, float]],
    pred: list[tuple[float, float]] | None,
    out_path: Path,
    title: str,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    radius = max(4, round(min(image.size) / 180))
    for index, definition in enumerate(KEYPOINT_DEFS):
        gx, gy = gt[index]
        draw.ellipse((gx - radius, gy - radius, gx + radius, gy + radius), fill=(0, 220, 80), outline=(0, 60, 0), width=2)
        draw.text((gx + radius + 1, gy - radius), definition.code, fill=(0, 220, 80), font=font)
        if pred is not None:
            px, py = pred[index]
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=(255, 140, 0), outline=(50, 20, 0), width=2)
            draw.line((gx, gy, px, py), fill=(255, 220, 0), width=2)
    draw.rectangle((6, 6, min(image.size[0] - 6, 1000), 36), fill=(0, 0, 0))
    draw.text((12, 15), title, fill=(255, 255, 255), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def summarize_model(model_name: str, input_type: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    model_rows = [row for row in rows if row["model_name"] == model_name and row["error_mm"] != ""]
    errors = [float(row["error_mm"]) for row in model_rows]
    output = {
        "model_name": model_name,
        "input_type": input_type,
        "mean_keypoint_error_mm": round(mean(errors), 3) if errors else "",
        "median_keypoint_error_mm": round(median(errors), 3) if errors else "",
        "num_test_images": len({row["image_name"] for row in model_rows}),
        "selected_as_preannotation_candidate": False,
    }
    for key in ("P1_snout_tip", "P3_operculum_posterior", "P5_caudal_base_midpoint"):
        key_errors = [float(row["error_mm"]) for row in model_rows if row["keypoint_name"] == key]
        prefix = key.split("_", 1)[0].lower()
        output[f"{prefix}_mean_error_mm"] = round(mean(key_errors), 3) if key_errors else ""
        output[f"{prefix}_median_error_mm"] = round(median(key_errors), 3) if key_errors else ""
    for key in ("P7U_caudal_fin_upper_tip", "P7L_caudal_fin_lower_tip", "P10_peduncle_depth_dorsal", "P11_peduncle_depth_ventral"):
        key_errors = [float(row["error_mm"]) for row in model_rows if row["keypoint_name"] == key]
        output[f"{key}_mean_error_mm"] = round(mean(key_errors), 3) if key_errors else ""
        output[f"{key}_median_error_mm"] = round(median(key_errors), 3) if key_errors else ""
    return output


def point_aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for model in MODELS:
        name = model["model_name"]
        for key in KEY_NAMES:
            subset = [row for row in rows if row["model_name"] == name and row["keypoint_name"] == key and row["error_mm"] != ""]
            errors = [float(row["error_mm"]) for row in subset]
            confs = [float(row["confidence"]) for row in subset if row["confidence"] != ""]
            out.append(
                {
                    "model_name": name,
                    "keypoint_name": key,
                    "mean_error_mm": round(mean(errors), 3) if errors else "",
                    "median_error_mm": round(median(errors), 3) if errors else "",
                    "max_error_mm": round(max(errors), 3) if errors else "",
                    "mean_confidence": round(mean(confs), 4) if confs else "",
                    "num_images": len(errors),
                }
            )
    return out


def main() -> None:
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("ultralytics is required for evaluation") from exc

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    split_df = pd.read_csv(DATASET_ROOT / "split_manifest.csv", dtype=str).fillna("")
    test_rows = split_df[split_df["split"] == "test"].copy()
    error_rows: list[dict[str, Any]] = []

    for model_info in MODELS:
        model_path = Path(model_info["path"])
        if not model_path.exists():
            raise FileNotFoundError(f"model not found: {model_path}")
        model = YOLO(str(model_path))
        model_name = str(model_info["model_name"])
        for _, row in test_rows.iterrows():
            image_name = str(row["image_name"])
            image_path = PROJECT_ROOT / str(row["warped_image_path"])
            label_path = PROJECT_ROOT / str(row["label_path"])
            with Image.open(image_path) as image:
                width, height = image.size
            gt_points = parse_gt_label(label_path, width, height)
            pred_points, confidences = predict_points(model, image_path)
            draw_prediction(
                image_path,
                gt_points,
                pred_points,
                PREDICTION_DIR / model_name / f"{Path(image_name).stem}_prediction.png",
                f"{model_name} {image_name}",
            )
            for index, key in enumerate(KEY_NAMES):
                if pred_points is None:
                    error_px: Any = ""
                    error_mm: Any = ""
                    confidence: Any = ""
                else:
                    error_px_value = math.hypot(pred_points[index][0] - gt_points[index][0], pred_points[index][1] - gt_points[index][1])
                    error_px = round(error_px_value, 3)
                    error_mm = round(error_px_value * MM_PER_PIXEL, 3)
                    confidence = round(confidences[index], 4) if index < len(confidences) else ""
                error_rows.append(
                    {
                        "model_name": model_name,
                        "image_name": image_name,
                        "specimen_id": str(row["specimen_id"]),
                        "keypoint_name": key,
                        "error_px": error_px,
                        "error_mm": error_mm,
                        "confidence": confidence,
                    }
                )

    summary_path = OUTPUT_ROOT / "keypoint_error_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(error_rows[0].keys()))
        writer.writeheader()
        writer.writerows(error_rows)

    by_point = point_aggregate(error_rows)
    with (OUTPUT_ROOT / "keypoint_error_by_point.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(by_point[0].keys()))
        writer.writeheader()
        writer.writerows(by_point)

    comparison = [summarize_model(str(model["model_name"]), str(model["input_type"]), error_rows) for model in MODELS]
    candidates = [row for row in comparison if str(row["model_name"]).startswith("v0.3") and row["median_keypoint_error_mm"] != ""]
    selected = min(candidates, key=lambda row: (float(row["median_keypoint_error_mm"]), float(row.get("p1_median_error_mm") or 9999))) if candidates else None
    if selected is not None:
        for row in comparison:
            row["selected_as_preannotation_candidate"] = row["model_name"] == selected["model_name"]
        selected_model = next(model for model in MODELS if model["model_name"] == selected["model_name"])
        shutil.copy2(Path(selected_model["path"]), MODEL_ROOT / "preannotation_candidate.pt")

    with (OUTPUT_ROOT / "model_comparison_report.csv").open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = list(comparison[0].keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison)

    selection_rows = []
    for row in comparison:
        selection_rows.append(
            {
                "model_name": row["model_name"],
                "median_keypoint_error_mm": row["median_keypoint_error_mm"],
                "p1_median_error_mm": row.get("p1_median_error_mm", ""),
                "selected": row["selected_as_preannotation_candidate"],
                "selection_rule": "lowest v0.3 median_keypoint_error_mm; P1 median as tie-breaker",
            }
        )
    with (MODEL_ROOT / "model_selection_report.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(selection_rows[0].keys()))
        writer.writeheader()
        writer.writerows(selection_rows)

    for file_name in ("data.yaml", "keypoint_schema.json", "split_manifest.csv", "dataset_summary.csv"):
        source = DATASET_ROOT / file_name
        if source.exists():
            shutil.copy2(source, MODEL_ROOT / file_name)
    training_config = {
        "model": "yolo11n-pose.pt",
        "dataset": relpath(DATASET_ROOT, PROJECT_ROOT),
        "imgsz": 1024,
        "epochs": 100,
        "batch": 2,
        "patience": 30,
        "device": 0,
        "workers": 2,
        "cache": False,
        "seed": 20260515,
        "fliplr": 0.0,
    }
    (MODEL_ROOT / "training_config.yaml").write_text("\n".join(f"{k}: {v}" for k, v in training_config.items()) + "\n", encoding="utf-8")
    readme = f"""# SiganusMorph YOLO-pose v0.3 corrected real 5.15

Training data: {relpath(DATASET_ROOT, PROJECT_ROOT)}
Images/specimens: 67 images, 34 specimens
Keypoints: 16 manual keypoints, P7V excluded from training
Split: grouped by specimen_id, inherited from v0.2 maskcrop split
Pretrained weight: yolo11n-pose.pt
GPU: NVIDIA RTX 3050 Ti Laptop GPU
Horizontal flip augmentation: disabled (fliplr=0.0)
Selected preannotation candidate: {selected['model_name'] if selected else 'not selected'}
Selection rule: lowest v0.3 median keypoint error on the shared test split.
"""
    (MODEL_ROOT / "README_model.md").write_text(readme, encoding="utf-8")

    print(f"Wrote {relpath(OUTPUT_ROOT / 'model_comparison_report.csv', PROJECT_ROOT)}")
    if selected is not None:
        print(f"Selected {selected['model_name']} as v0.3 preannotation_candidate.pt")


if __name__ == "__main__":
    main()
