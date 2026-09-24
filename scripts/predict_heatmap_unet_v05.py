"""Predict v0.5 candidate dataset with heatmap U-Net checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph.config import KEYPOINT_DEFS  # noqa: E402
from siganusmorph.heatmap_unet import HeatmapDataset, LightweightUNet, heatmaps_to_points  # noqa: E402


KEYPOINT_KEYS = tuple(f"{definition.code}_{definition.name}" for definition in KEYPOINT_DEFS)
DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "siganusmorph_heatmap_unet_v0.5_candidate"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "model_eval" / "heatmap_unet_v0.5_predictions"


def load_model(model_path: Path, device: torch.device) -> tuple[LightweightUNet, dict[str, Any]]:
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get("config", {})
    model = LightweightUNet(out_channels=int(config.get("num_keypoints", 16)), base_channels=int(config.get("base_channels", 24))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def font(size: int = 18) -> ImageFont.ImageFont:
    for candidate in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def save_prediction_visual(dataset_root: Path, output_root: Path, sample: dict[str, Any], pred_points: np.ndarray, gt_points: np.ndarray, tag: str) -> None:
    image = Image.open(dataset_root / sample["image_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    errors = np.sqrt(np.sum((pred_points - gt_points) ** 2, axis=1))
    large = []
    for idx, key in enumerate(KEYPOINT_KEYS):
        gx, gy = gt_points[idx]
        px, py = pred_points[idx]
        radius = 5
        draw.ellipse((gx - radius, gy - radius, gx + radius, gy + radius), fill=(60, 220, 90), outline=(0, 80, 0), width=2)
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), outline=(255, 120, 0), width=3)
        draw.line((gx, gy, px, py), fill=(255, 255, 255), width=1)
        if errors[idx] > 100:
            large.append(key)
            draw.ellipse((px - 14, py - 14, px + 14, py + 14), outline=(255, 230, 0), width=4)
    text = f"{tag} mean_px={errors.mean():.1f} median_px={np.median(errors):.1f} large={';'.join(large[:4])}"
    draw.rectangle((8, 8, min(510, 20 + len(text) * 8), 44), fill=(0, 0, 0))
    draw.text((16, 14), text, fill=(255, 255, 255), font=font(16))
    out = output_root / "visual_predictions" / f"{Path(str(sample['image_name'])).stem}_{tag}_prediction.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out, optimize=True)


def predict_split(model: torch.nn.Module, dataset_root: Path, output_root: Path, split: str, tag: str, device: torch.device) -> Path:
    dataset = HeatmapDataset(dataset_root, split, augment=False)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=device.type == "cuda")
    labels = json.loads((dataset_root / "labels.json").read_text(encoding="utf-8"))
    samples_by_name = {item["image_name"]: item for item in labels["samples"] if item["split"] == split}
    rows = []
    debug = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            preds = torch.sigmoid(model(images))
            points, confidences = heatmaps_to_points(preds)
            gt_points = batch["keypoints"].numpy()
            names = list(batch["image_name"])
            specimens = list(batch["specimen_id"])
            crop_w = batch["source_crop_width"].numpy()
            crop_h = batch["source_crop_height"].numpy()
            for i, image_name in enumerate(names):
                sx = float(crop_w[i]) / 512.0
                sy = float(crop_h[i]) / 512.0
                pred_crop = points[i].copy()
                pred_crop[:, 0] *= sx
                pred_crop[:, 1] *= sy
                for k, key in enumerate(KEYPOINT_KEYS):
                    rows.append(
                        {
                            "image_name": image_name,
                            "specimen_id": specimens[i],
                            "split": split,
                            "keypoint_name": key,
                            "x": float(pred_crop[k, 0]),
                            "y": float(pred_crop[k, 1]),
                            "confidence": float(confidences[i, k]),
                            "coordinate_space": "crop",
                        }
                    )
                debug.append({"image_name": image_name, "split": split, "pred_points_512": points[i].tolist(), "confidences": confidences[i].tolist()})
                if split == "test":
                    save_prediction_visual(dataset_root, output_root, samples_by_name[image_name], points[i], gt_points[i], tag)
    suffix = "" if tag == "best" else f"_{tag}"
    out_path = output_root / f"predictions_{split}{suffix}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    debug_path = output_root / "debug_json" / f"predictions_{split}{suffix}_debug.json"
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--tag", default="best")
    parser.add_argument("--splits", default="val,test", help="Comma-separated splits to predict.")
    args = parser.parse_args()
    dataset_root = Path(args.dataset)
    output_root = Path(args.output_root)
    if not dataset_root.is_absolute():
        dataset_root = PROJECT_ROOT / dataset_root
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _config = load_model(Path(args.model), device)
    for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
        path = predict_split(model, dataset_root, output_root, split, args.tag, device)
        print(f"Wrote {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
