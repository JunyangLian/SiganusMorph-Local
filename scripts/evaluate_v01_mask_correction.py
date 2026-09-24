"""Evaluate v0.1 YOLO-pose predictions with mask-aware P1 correction."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph.config import KEYPOINT_DEFS  # noqa: E402
from siganusmorph.image_utils import ensure_rgb  # noqa: E402
from siganusmorph.mask_keypoint_correction import correct_keypoints_with_mask  # noqa: E402
from siganusmorph.measurements import calculate_measurements  # noqa: E402
from siganusmorph.segmentation import (  # noqa: E402
    padded_bbox,
    save_mask_png,
    save_segmentation_preview,
    segment_fish_from_blue_board,
)


DATASET_ROOT = PROJECT_ROOT / "datasets" / "siganusmorph_real_5_15_yolopose"
MODEL_PATH = PROJECT_ROOT / "models" / "siganusmorph_yolopose_v0.1_real_5_15" / "weights" / "preannotation_candidate.pt"
ANNOTATION_DIR = PROJECT_ROOT / "results" / "real_annotation_5_15" / "keypoints"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "model_eval" / "v0.1_mask_correction"
MM_PER_PIXEL = 0.1


KEY_ORDER = [f"{definition.code}_{definition.name}" for definition in KEYPOINT_DEFS]
FULL_TO_SHORT = {f"{definition.code}_{definition.name}": definition.name for definition in KEYPOINT_DEFS}


def warped_image_path(image_name: str) -> Path:
    stem = Path(image_name).stem
    return PROJECT_ROOT / "data" / "real_images_warped" / f"{stem}_warped.png"


def annotation_path(image_name: str) -> Path:
    return ANNOTATION_DIR / f"{Path(image_name).stem}_keypoints.json"


def load_ground_truth(image_name: str) -> dict[str, tuple[float, float]]:
    data = json.loads(annotation_path(image_name).read_text(encoding="utf-8"))
    points = data["keypoints"]
    return {
        key: (float(points[key][0]), float(points[key][1]))
        for key in KEY_ORDER
        if key in points
    }


def predict_rows(split_df: pd.DataFrame) -> dict[str, tuple[dict[str, tuple[float, float]], dict[str, float]]]:
    model = YOLO(str(MODEL_PATH))
    crop_paths = [PROJECT_ROOT / str(path) for path in split_df["warped_image_path"].tolist()]
    results = model.predict(
        source=[str(path) for path in crop_paths],
        imgsz=1024,
        conf=0.05,
        max_det=1,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(OUTPUT_ROOT),
        name="raw_predictions",
        exist_ok=True,
        verbose=False,
    )
    predictions: dict[str, tuple[dict[str, tuple[float, float]], dict[str, float]]] = {}
    for result, (_, row) in zip(results, split_df.iterrows()):
        image_name = str(row["image_name"])
        crop_x0 = float(row.get("crop_x0", 0))
        crop_y0 = float(row.get("crop_y0", 0))
        pred_points: dict[str, tuple[float, float]] = {}
        confidences: dict[str, float] = {}
        if result.keypoints is not None and len(result.keypoints) > 0:
            xy = result.keypoints.xy[0].cpu().numpy()
            conf = result.keypoints.conf[0].cpu().numpy() if result.keypoints.conf is not None else np.ones(len(KEY_ORDER))
            for index, key in enumerate(KEY_ORDER):
                pred_points[key] = (float(xy[index][0] + crop_x0), float(xy[index][1] + crop_y0))
                confidences[key] = float(conf[index])
        predictions[image_name] = (pred_points, confidences)
    return predictions


def point_error(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_by_point(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    aggregate: dict[str, dict[str, float]] = {}
    for key in KEY_ORDER:
        name = f"{key}"
        values = [float(row["error_mm"]) for row in rows if row["keypoint_name"] == name and row["error_mm"] != ""]
        if not values:
            aggregate[name] = {"mean": math.nan, "median": math.nan, "count": 0}
            continue
        values_sorted = sorted(values)
        aggregate[name] = {
            "mean": float(sum(values) / len(values)),
            "median": float(values_sorted[len(values_sorted) // 2]),
            "count": len(values),
        }
    return aggregate


def make_visual_comparison(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    gt_p1: tuple[float, float],
    raw_p1: tuple[float, float],
    corrected_p1: tuple[float, float],
    raw_error_mm: float,
    corrected_error_mm: float,
    output_path: Path,
) -> None:
    preview = ensure_rgb(image).copy()
    overlay = preview.copy()
    overlay[mask > 0] = (255, 80, 0)
    preview = cv2.addWeighted(overlay, 0.25, preview, 0.75, 0)
    x0, y0, x1, y1 = bbox
    if x1 > x0 and y1 > y0:
        cv2.rectangle(preview, (x0, y0), (x1, y1), (255, 220, 0), 8)
    points = [
        (gt_p1, (0, 255, 0), "GT P1"),
        (raw_p1, (255, 0, 0), "raw P1"),
        (corrected_p1, (255, 255, 0), "mask P1"),
    ]
    for point, color, label in points:
        x, y = int(round(point[0])), int(round(point[1]))
        cv2.circle(preview, (x, y), 18, color, -1, cv2.LINE_AA)
        cv2.putText(preview, label, (x + 20, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)
    text = f"P1 raw={raw_error_mm:.2f} mm  corrected={corrected_error_mm:.2f} mm"
    cv2.putText(preview, text, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 7, cv2.LINE_AA)
    cv2.putText(preview, text, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 2, cv2.LINE_AA)

    max_width = 1800
    if preview.shape[1] > max_width:
        scale = max_width / preview.shape[1]
        preview = cv2.resize(preview, (max_width, int(round(preview.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(preview).save(output_path)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    split_df = pd.read_csv(DATASET_ROOT / "split_manifest.csv", dtype=str).fillna("")
    predictions = predict_rows(split_df)

    raw_error_rows: list[dict[str, Any]] = []
    corrected_error_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    measurement_rows: list[dict[str, Any]] = []

    mask_dir = OUTPUT_ROOT / "masks"
    preview_dir = OUTPUT_ROOT / "segmentation_previews"
    visual_dir = OUTPUT_ROOT / "visual_comparisons"

    for _, row in split_df.iterrows():
        image_name = str(row["image_name"])
        specimen_id = str(row["specimen_id"])
        split = str(row["split"])
        warped = warped_image_path(image_name)
        image = np.array(Image.open(warped).convert("RGB"))
        gt = load_ground_truth(image_name)

        mask, bbox, quality = segment_fish_from_blue_board(image)
        success = bool(quality.get("segmentation_success", False))
        quality_text = str(quality.get("segmentation_quality", "unknown"))
        if success:
            crop_bbox = padded_bbox(bbox, image.shape, padding_ratio=0.08)
        else:
            crop_bbox = (0, 0, 0, 0)
        save_mask_png(mask, mask_dir / f"{Path(image_name).stem}_fish_mask.png")
        save_segmentation_preview(image, mask, crop_bbox if success else bbox, quality, preview_dir / f"{Path(image_name).stem}_segmentation_preview.png")

        raw_pred, confidences = predictions.get(image_name, ({}, {}))
        corrected_pred = dict(raw_pred)
        correction_log: dict[str, Any] = {
            "p1_source": "model",
            "p1_correction_applied": False,
            "p1_model_x": raw_pred.get("P1_snout_tip", ("", ""))[0] if raw_pred else "",
            "p1_model_y": raw_pred.get("P1_snout_tip", ("", ""))[1] if raw_pred else "",
            "p1_mask_x": "",
            "p1_mask_y": "",
        }
        needs_review = False
        review_reason = ""
        if success and raw_pred:
            corrected_pred, correction_log = correct_keypoints_with_mask(raw_pred, mask, crop_bbox, confidences)
            needs_review = bool(correction_log.get("p1_correction_applied", False))
            review_reason = "p1_mask_corrected" if needs_review else ""
        elif not success:
            needs_review = True
            review_reason = "segmentation_failed"

        for key in KEY_ORDER:
            gt_point = gt.get(key)
            raw_point = raw_pred.get(key)
            corrected_point = corrected_pred.get(key)
            confidence = confidences.get(key, "")
            for target_rows, point in ((raw_error_rows, raw_point), (corrected_error_rows, corrected_point)):
                if gt_point is None or point is None:
                    error_px = ""
                    error_mm = ""
                else:
                    error_px_value = point_error(point, gt_point)
                    error_px = round(error_px_value, 3)
                    error_mm = round(error_px_value * MM_PER_PIXEL, 3)
                target_rows.append(
                    {
                        "image_name": image_name,
                        "specimen_id": specimen_id,
                        "split": split,
                        "keypoint_name": key,
                        "error_px": error_px,
                        "error_mm": error_mm,
                        "confidence": round(float(confidence), 4) if confidence != "" else "",
                    }
                )

        p1_gt = gt.get("P1_snout_tip")
        p1_raw = raw_pred.get("P1_snout_tip")
        p1_corrected = corrected_pred.get("P1_snout_tip")
        p1_raw_error = point_error(p1_raw, p1_gt) * MM_PER_PIXEL if p1_gt and p1_raw else math.nan
        p1_corrected_error = point_error(p1_corrected, p1_gt) * MM_PER_PIXEL if p1_gt and p1_corrected else math.nan
        p1_improvement = p1_raw_error - p1_corrected_error if not math.isnan(p1_raw_error) and not math.isnan(p1_corrected_error) else math.nan
        report_rows.append(
            {
                "image_name": image_name,
                "specimen_id": specimen_id,
                "split": split,
                "segmentation_success": success,
                "segmentation_quality": quality_text,
                "fish_bbox_x1": crop_bbox[0],
                "fish_bbox_y1": crop_bbox[1],
                "fish_bbox_x2": crop_bbox[2],
                "fish_bbox_y2": crop_bbox[3],
                "mask_area_px": quality.get("mask_area_px", 0),
                "p1_model_x": correction_log.get("p1_model_x", ""),
                "p1_model_y": correction_log.get("p1_model_y", ""),
                "p1_mask_x": correction_log.get("p1_mask_x", ""),
                "p1_mask_y": correction_log.get("p1_mask_y", ""),
                "p1_corrected_x": p1_corrected[0] if p1_corrected else "",
                "p1_corrected_y": p1_corrected[1] if p1_corrected else "",
                "p1_correction_applied": bool(correction_log.get("p1_correction_applied", False)),
                "p1_source": correction_log.get("p1_source", "model"),
                "p1_error_raw_mm": round(p1_raw_error, 3) if not math.isnan(p1_raw_error) else "",
                "p1_error_corrected_mm": round(p1_corrected_error, 3) if not math.isnan(p1_corrected_error) else "",
                "p1_improvement_mm": round(p1_improvement, 3) if not math.isnan(p1_improvement) else "",
                "needs_review": needs_review,
                "review_reason": review_reason,
                "keypoint_correction_log": json.dumps(correction_log, ensure_ascii=False),
            }
        )

        if raw_pred and corrected_pred:
            try:
                raw_short = {FULL_TO_SHORT[key]: value for key, value in raw_pred.items()}
                corrected_short = {FULL_TO_SHORT[key]: value for key, value in corrected_pred.items()}
                raw_m = calculate_measurements(raw_short, MM_PER_PIXEL, "auto")
                corr_m = calculate_measurements(corrected_short, MM_PER_PIXEL, "auto")
                measurement_rows.append(
                    {
                        "image_name": image_name,
                        "specimen_id": specimen_id,
                        "split": split,
                        "TL_final_raw_mm": raw_m.get("TL_final_mm", ""),
                        "TL_final_corrected_mm": corr_m.get("TL_final_mm", ""),
                        "SL_final_raw_mm": raw_m.get("SL_final_mm", ""),
                        "SL_final_corrected_mm": corr_m.get("SL_final_mm", ""),
                        "head_length_raw_mm": raw_m.get("head_length_straight_mm", ""),
                        "head_length_corrected_mm": corr_m.get("head_length_straight_mm", ""),
                        "snout_length_raw_mm": raw_m.get("snout_length_straight_mm", ""),
                        "snout_length_corrected_mm": corr_m.get("snout_length_straight_mm", ""),
                    }
                )
            except Exception as exc:  # pragma: no cover - report-only fallback
                measurement_rows.append({"image_name": image_name, "specimen_id": specimen_id, "split": split, "error": str(exc)})

        if split == "test" and p1_gt and p1_raw and p1_corrected:
            make_visual_comparison(
                image,
                mask,
                crop_bbox if success else bbox,
                p1_gt,
                p1_raw,
                p1_corrected,
                p1_raw_error,
                p1_corrected_error,
                visual_dir / f"{Path(image_name).stem}_p1_mask_comparison.png",
            )

    test_raw = [row for row in raw_error_rows if row["split"] == "test"]
    test_corrected = [row for row in corrected_error_rows if row["split"] == "test"]
    write_csv(
        OUTPUT_ROOT / "keypoint_error_summary_raw.csv",
        test_raw,
        ["image_name", "specimen_id", "split", "keypoint_name", "error_px", "error_mm", "confidence"],
    )
    write_csv(
        OUTPUT_ROOT / "keypoint_error_summary_mask_corrected.csv",
        test_corrected,
        ["image_name", "specimen_id", "split", "keypoint_name", "error_px", "error_mm", "confidence"],
    )
    raw_agg = aggregate_by_point(test_raw)
    corr_agg = aggregate_by_point(test_corrected)
    comparison_rows = []
    for key in KEY_ORDER:
        raw = raw_agg[key]
        corr = corr_agg[key]
        comparison_rows.append(
            {
                "keypoint_name": key,
                "mean_error_raw_mm": round(raw["mean"], 3) if not math.isnan(raw["mean"]) else "",
                "median_error_raw_mm": round(raw["median"], 3) if not math.isnan(raw["median"]) else "",
                "mean_error_mask_corrected_mm": round(corr["mean"], 3) if not math.isnan(corr["mean"]) else "",
                "median_error_mask_corrected_mm": round(corr["median"], 3) if not math.isnan(corr["median"]) else "",
                "mean_improvement_mm": round(raw["mean"] - corr["mean"], 3) if not math.isnan(raw["mean"]) and not math.isnan(corr["mean"]) else "",
                "median_improvement_mm": round(raw["median"] - corr["median"], 3) if not math.isnan(raw["median"]) and not math.isnan(corr["median"]) else "",
                "num_images": raw["count"],
            }
        )
    write_csv(
        OUTPUT_ROOT / "keypoint_error_by_point_comparison.csv",
        comparison_rows,
        [
            "keypoint_name",
            "mean_error_raw_mm",
            "median_error_raw_mm",
            "mean_error_mask_corrected_mm",
            "median_error_mask_corrected_mm",
            "mean_improvement_mm",
            "median_improvement_mm",
            "num_images",
        ],
    )
    write_csv(
        OUTPUT_ROOT / "mask_correction_report.csv",
        report_rows,
        [
            "image_name",
            "specimen_id",
            "split",
            "segmentation_success",
            "segmentation_quality",
            "fish_bbox_x1",
            "fish_bbox_y1",
            "fish_bbox_x2",
            "fish_bbox_y2",
            "mask_area_px",
            "p1_model_x",
            "p1_model_y",
            "p1_mask_x",
            "p1_mask_y",
            "p1_corrected_x",
            "p1_corrected_y",
            "p1_correction_applied",
            "p1_source",
            "p1_error_raw_mm",
            "p1_error_corrected_mm",
            "p1_improvement_mm",
            "needs_review",
            "review_reason",
            "keypoint_correction_log",
        ],
    )
    if measurement_rows:
        pd.DataFrame(measurement_rows).to_csv(OUTPUT_ROOT / "measurement_comparison.csv", index=False, encoding="utf-8-sig")

    report_df = pd.DataFrame(report_rows)
    success_rate = report_df["segmentation_success"].astype(bool).mean() if not report_df.empty else 0.0
    test_report = report_df[report_df["split"] == "test"]
    p1_raw_mean = pd.to_numeric(test_report["p1_error_raw_mm"], errors="coerce").mean()
    p1_corr_mean = pd.to_numeric(test_report["p1_error_corrected_mm"], errors="coerce").mean()
    print(f"Segmentation success rate: {success_rate:.3f}")
    print(f"Test P1 raw mean error mm: {p1_raw_mean:.3f}")
    print(f"Test P1 corrected mean error mm: {p1_corr_mean:.3f}")
    print(f"Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()

