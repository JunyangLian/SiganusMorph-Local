"""Compare v0.2 enhanced preannotation against migrated manual ground truth."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph.config import KEYPOINT_DEFS  # noqa: E402
from siganusmorph.image_utils import ensure_rgb  # noqa: E402
from siganusmorph.preannotation import preannotate_warped_image  # noqa: E402
from siganusmorph.real_dataset import corrected_annotation_dir, relpath  # noqa: E402


MODEL_PATH = PROJECT_ROOT / "models" / "siganusmorph_yolopose_v0.2_maskcrop_real_5_15" / "preannotation_candidate.pt"
SPLIT_MANIFEST = PROJECT_ROOT / "datasets" / "siganusmorph_real_5_15_yolopose_maskcrop" / "split_manifest.csv"
CORRECTED_DIR = corrected_annotation_dir(PROJECT_ROOT) / "keypoints"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "model_eval" / "v0.2_vs_manual_original"
VIS_DIR = OUTPUT_ROOT / "visual_comparisons"
MM_PER_PIXEL = 0.1
LARGE_ERROR_MM = 10.0
KEY_ORDER = [f"{definition.code}_{definition.name}" for definition in KEYPOINT_DEFS]


def base_stem(image_name: str) -> str:
    return Path(image_name).stem.removesuffix("_warped")


def warped_path(image_name: str) -> Path:
    return PROJECT_ROOT / "data" / "real_images_warped" / f"{base_stem(image_name)}_warped.png"


def corrected_json_path(image_name: str) -> Path:
    return CORRECTED_DIR / f"{base_stem(image_name)}_keypoints.json"


def as_xy(payload: Any) -> tuple[float, float] | None:
    if isinstance(payload, Mapping):
        try:
            return float(payload["x"]), float(payload["y"])
        except Exception:
            return None
    if isinstance(payload, (list, tuple)) and len(payload) >= 2:
        return float(payload[0]), float(payload[1])
    return None


def load_manual_points(path: Path) -> dict[str, tuple[float, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    points = payload.get("corrected_keypoints") or payload.get("keypoints") or {}
    manual: dict[str, tuple[float, float]] = {}
    for key in KEY_ORDER:
        point = as_xy(points.get(key) if isinstance(points, Mapping) else None)
        if point is not None:
            manual[key] = point
    return manual


def serialize_points(points: Mapping[str, Any]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for key, value in points.items():
        point = as_xy(value)
        if point is not None:
            out[str(key)] = [float(point[0]), float(point[1])]
    return out


def update_corrected_json(
    path: Path,
    result: Mapping[str, Any],
    manual_points: Mapping[str, tuple[float, float]],
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = dict(result.get("metadata", {}) or {})
    payload["model_keypoints_raw"] = serialize_points(metadata.get("model_keypoints_raw", {}))
    payload["mask_suggestions"] = serialize_points(metadata.get("mask_suggestions", {}))
    payload["model_confidences"] = dict(result.get("confidences", {}) or {})
    payload["corrected_keypoints"] = {key: [float(x), float(y)] for key, (x, y) in manual_points.items()}
    payload["annotation_mode"] = payload.get("annotation_mode") or "manual_original"
    payload["annotation_status"] = "corrected_and_confirmed"
    payload["preannotation"] = {
        **dict(payload.get("preannotation", {}) or {}),
        **metadata,
        "corrected_keypoints": payload["corrected_keypoints"],
        "annotation_status": "corrected_and_confirmed",
    }
    payload["qc_results"] = metadata.get("qc_results", payload.get("qc_results", {}))
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def draw_comparison(
    image: np.ndarray,
    manual: Mapping[str, tuple[float, float]],
    model: Mapping[str, Any],
    out_path: Path,
    title: str,
) -> None:
    canvas = Image.fromarray(ensure_rgb(image)).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    radius = max(5, round(min(canvas.size) / 320))
    for definition in KEYPOINT_DEFS:
        key = f"{definition.code}_{definition.name}"
        m = manual.get(key)
        p = as_xy(model.get(key) if isinstance(model, Mapping) else None)
        if m is not None:
            draw.ellipse((m[0] - radius, m[1] - radius, m[0] + radius, m[1] + radius), fill=(0, 220, 80), outline=(0, 50, 0), width=2)
            draw.text((m[0] + radius + 2, m[1] - radius), definition.code, fill=(0, 220, 80), font=font)
        if p is not None:
            draw.ellipse((p[0] - radius, p[1] - radius, p[0] + radius, p[1] + radius), fill=(255, 140, 0), outline=(40, 20, 0), width=2)
            draw.text((p[0] + radius + 2, p[1] + 2), f"{definition.code}m", fill=(255, 140, 0), font=font)
        if m is not None and p is not None:
            error_mm = math.hypot(p[0] - m[0], p[1] - m[1]) * MM_PER_PIXEL
            line_color = (255, 0, 0) if error_mm > LARGE_ERROR_MM else (255, 210, 0)
            draw.line((m[0], m[1], p[0], p[1]), fill=line_color, width=2)
    draw.rectangle((8, 8, 900, 42), fill=(0, 0, 0))
    draw.text((14, 17), f"{title} | green=manual corrected, orange=v0.2 model", fill=(255, 255, 255), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key in KEY_ORDER:
        values = [float(row["error_mm"]) for row in rows if row["keypoint_name"] == key and row["error_mm"] != ""]
        px_values = [float(row["error_px"]) for row in rows if row["keypoint_name"] == key and row["error_px"] != ""]
        confs = [float(row["model_confidence"]) for row in rows if row["keypoint_name"] == key and row["model_confidence"] != ""]
        out.append(
            {
                "keypoint_name": key,
                "mean_error_px": round(mean(px_values), 3) if px_values else "",
                "median_error_px": round(median(px_values), 3) if px_values else "",
                "max_error_px": round(max(px_values), 3) if px_values else "",
                "mean_error_mm": round(mean(values), 3) if values else "",
                "median_error_mm": round(median(values), 3) if values else "",
                "max_error_mm": round(max(values), 3) if values else "",
                "mean_confidence": round(mean(confs), 4) if confs else "",
                "num_images": len(values),
                "num_large_errors": sum(1 for value in values if value > LARGE_ERROR_MM),
            }
        )
    return out


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"v0.2 preannotation model not found: {MODEL_PATH}")
    if not SPLIT_MANIFEST.exists():
        raise FileNotFoundError(f"split manifest not found: {SPLIT_MANIFEST}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    split_df = pd.read_csv(SPLIT_MANIFEST, dtype=str).fillna("")
    rows: list[dict[str, Any]] = []
    for _, split_row in split_df.iterrows():
        image_name = str(split_row["image_name"])
        specimen_id = str(split_row["specimen_id"])
        split = str(split_row["split"])
        json_path = corrected_json_path(image_name)
        image_path = warped_path(image_name)
        if not json_path.exists() or not image_path.exists():
            for key in KEY_ORDER:
                rows.append(
                    {
                        "image_name": image_name,
                        "specimen_id": specimen_id,
                        "split": split,
                        "keypoint_name": key,
                        "model_x": "",
                        "model_y": "",
                        "manual_x": "",
                        "manual_y": "",
                        "error_px": "",
                        "error_mm": "",
                        "model_confidence": "",
                        "is_large_error": "",
                        "status": "missing_json_or_image",
                    }
                )
            continue
        manual = load_manual_points(json_path)
        image = np.asarray(Image.open(image_path).convert("RGB"))
        try:
            result = preannotate_warped_image(image, image_name, PROJECT_ROOT, MODEL_PATH)
            metadata = dict(result.get("metadata", {}) or {})
            model_points = metadata.get("model_keypoints_raw", {})
            confidences = dict(result.get("confidences", {}) or {})
            update_corrected_json(json_path, result, manual)
            draw_comparison(
                image,
                manual,
                model_points if isinstance(model_points, Mapping) else {},
                VIS_DIR / f"{base_stem(image_name)}_v02_vs_manual.png",
                f"{image_name} {specimen_id} {split}",
            )
            status = "ok"
        except Exception as exc:  # noqa: BLE001 - evaluation should continue per image.
            model_points = {}
            confidences = {}
            status = f"prediction_failed:{exc}"

        for key in KEY_ORDER:
            manual_xy = manual.get(key)
            model_xy = as_xy(model_points.get(key) if isinstance(model_points, Mapping) else None)
            if manual_xy is None or model_xy is None:
                error_px: Any = ""
                error_mm: Any = ""
                is_large: Any = ""
            else:
                error_px_value = math.hypot(model_xy[0] - manual_xy[0], model_xy[1] - manual_xy[1])
                error_px = round(error_px_value, 3)
                error_mm = round(error_px_value * MM_PER_PIXEL, 3)
                is_large = bool(error_mm > LARGE_ERROR_MM)
            rows.append(
                {
                    "image_name": image_name,
                    "specimen_id": specimen_id,
                    "split": split,
                    "keypoint_name": key,
                    "model_x": round(model_xy[0], 3) if model_xy else "",
                    "model_y": round(model_xy[1], 3) if model_xy else "",
                    "manual_x": round(manual_xy[0], 3) if manual_xy else "",
                    "manual_y": round(manual_xy[1], 3) if manual_xy else "",
                    "error_px": error_px,
                    "error_mm": error_mm,
                    "model_confidence": round(float(confidences.get(key, "")), 4) if key in confidences else "",
                    "is_large_error": is_large,
                    "status": status,
                }
            )

    columns = [
        "image_name",
        "specimen_id",
        "split",
        "keypoint_name",
        "model_x",
        "model_y",
        "manual_x",
        "manual_y",
        "error_px",
        "error_mm",
        "model_confidence",
        "is_large_error",
        "status",
    ]
    for name in ("keypoint_error_summary.csv", "model_vs_manual_displacement.csv"):
        with (OUTPUT_ROOT / name).open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    by_point = aggregate(rows)
    with (OUTPUT_ROOT / "keypoint_error_by_point.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(by_point[0].keys()))
        writer.writeheader()
        writer.writerows(by_point)

    ok_rows = [row for row in rows if row["error_mm"] != ""]
    mm_errors = [float(row["error_mm"]) for row in ok_rows]
    print(f"Evaluated {len(split_df)} images / {len(ok_rows)} keypoint comparisons.")
    if mm_errors:
        print(f"Mean error: {mean(mm_errors):.3f} mm")
        print(f"Median error: {median(mm_errors):.3f} mm")
    print(f"Output directory: {relpath(OUTPUT_ROOT, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
