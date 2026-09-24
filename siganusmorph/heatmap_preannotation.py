"""Heatmap-U-Net preannotation helpers.

This module is intentionally separate from the existing YOLO/geometry
preannotation flow. It loads the lightweight heatmap model, predicts on the
same mask-crop coordinate system used for training, and maps points back to
warped-image coordinates for hybrid evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

from .config import KEYPOINT_DEFS
from .heatmap_unet import LightweightUNet, heatmaps_to_points
from .image_utils import ensure_rgb
from .segmentation import padded_bbox, segment_fish_from_blue_board


DEFAULT_DATASET_ROOT = Path("datasets") / "siganusmorph_heatmap_unet_v0.1"
DEFAULT_MODEL_PATH = Path("models") / "siganusmorph_heatmap_unet_v0.1" / "preannotation_candidate.pt"

_MODEL_CACHE: dict[str, tuple[LightweightUNet, torch.device, dict[str, Any]]] = {}
_CROP_MANIFEST_CACHE: dict[str, pd.DataFrame] = {}


def clear_heatmap_model_cache(model_path: str | Path | None = None) -> None:
    """Release cached heatmap models without discarding crop manifests."""
    if model_path is None:
        _MODEL_CACHE.clear()
        return
    _MODEL_CACHE.pop(str(Path(model_path).resolve()), None)


@dataclass(frozen=True)
class CropTransform:
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    crop_width: float
    crop_height: float
    input_size: int

    def resized_to_warped(self, point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        return (
            self.bbox_x1 + x * self.crop_width / float(self.input_size),
            self.bbox_y1 + y * self.crop_height / float(self.input_size),
        )


def default_heatmap_model(project_root: Path) -> Path:
    """Return the configured heatmap preannotation candidate path."""
    return project_root / DEFAULT_MODEL_PATH


def _base_from_image_name(image_name: str) -> str:
    stem = Path(image_name).stem
    if stem.endswith("_warped"):
        stem = stem[: -len("_warped")]
    return stem


def _load_model(model_path: Path) -> tuple[LightweightUNet, torch.device, dict[str, Any]]:
    resolved = str(model_path.resolve())
    if resolved in _MODEL_CACHE:
        return _MODEL_CACHE[resolved]
    checkpoint = torch.load(model_path, map_location="cpu")
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    base_channels = int(config.get("base_channels", 24))
    num_keypoints = int(config.get("num_keypoints", len(KEYPOINT_DEFS)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightweightUNet(out_channels=num_keypoints, base_channels=base_channels)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    _MODEL_CACHE[resolved] = (model, device, config)
    return model, device, config


def _load_crop_manifest(project_root: Path, dataset_root: Path) -> pd.DataFrame:
    manifest_path = project_root / dataset_root / "split_manifest.csv"
    resolved = str(manifest_path.resolve())
    if resolved not in _CROP_MANIFEST_CACHE:
        if not manifest_path.exists():
            raise FileNotFoundError(f"heatmap crop manifest not found: {manifest_path}")
        df = pd.read_csv(manifest_path)
        df["_base"] = df["image_name"].astype(str).map(_base_from_image_name)
        _CROP_MANIFEST_CACHE[resolved] = df
    return _CROP_MANIFEST_CACHE[resolved]


def crop_transform_for_image(project_root: Path, image_name: str, dataset_root: Path = DEFAULT_DATASET_ROOT) -> CropTransform:
    """Find the mask-crop bbox used by the heatmap dataset for an image."""
    df = _load_crop_manifest(project_root, dataset_root)
    base = _base_from_image_name(image_name)
    rows = df[df["_base"] == base]
    if rows.empty:
        raise KeyError(f"no heatmap crop metadata found for {image_name}")
    row = rows.iloc[0]
    width = float(row["crop_width"])
    height = float(row["crop_height"])
    return CropTransform(
        bbox_x1=float(row["bbox_x1"]),
        bbox_y1=float(row["bbox_y1"]),
        bbox_x2=float(row["bbox_x2"]),
        bbox_y2=float(row["bbox_y2"]),
        crop_width=width,
        crop_height=height,
        input_size=512,
    )


def _clip_bbox(transform: CropTransform, image_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x1 = max(0, min(width - 1, int(round(transform.bbox_x1))))
    y1 = max(0, min(height - 1, int(round(transform.bbox_y1))))
    x2 = max(x1 + 1, min(width, int(round(transform.bbox_x2))))
    y2 = max(y1 + 1, min(height, int(round(transform.bbox_y2))))
    return x1, y1, x2, y2


def predict_keypoints_heatmap_unet(
    crop_image: np.ndarray,
    model_path: str | Path,
    keypoint_schema: list[str] | tuple[str, ...] | None = None,
) -> tuple[dict[str, tuple[float, float]], dict[str, float], dict[str, Any]]:
    """Predict heatmap keypoints in resized crop coordinates.

    The returned coordinates are in the 512x512 model input space. Use
    :func:`predict_warped_keypoints_heatmap_unet` when warped coordinates are
    needed.
    """
    model, device, config = _load_model(Path(model_path))
    input_size = int(config.get("image_size", 512))
    schema = tuple(keypoint_schema or (f"{kp.code}_{kp.name}" for kp in KEYPOINT_DEFS))
    rgb = ensure_rgb(crop_image)
    resized = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized.transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0).to(device)
    with torch.no_grad():
        heatmaps = model(tensor)
    points_np, conf_np = heatmaps_to_points(heatmaps.cpu())
    points = points_np[0]
    confidences = conf_np[0]
    keypoints = {
        key: (float(points[index][0]), float(points[index][1]))
        for index, key in enumerate(schema)
        if index < len(points)
    }
    conf = {
        key: float(confidences[index])
        for index, key in enumerate(schema)
        if index < len(confidences)
    }
    debug = {
        "input_size": input_size,
        "model_path": str(model_path),
        "heatmap_shape": list(heatmaps.shape),
        "coordinate_space": "resized_maskcrop",
    }
    return keypoints, conf, debug


def predict_warped_keypoints_heatmap_unet(
    warped_image: np.ndarray,
    image_name: str,
    project_root: Path,
    model_path: str | Path | None = None,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    keypoint_schema: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Predict heatmap keypoints and map them back to warped-image coordinates."""
    selected_model = Path(model_path) if model_path is not None else default_heatmap_model(project_root)
    if not selected_model.is_absolute():
        selected_model = project_root / selected_model
    if not selected_model.exists():
        raise FileNotFoundError(f"heatmap model not found: {selected_model}")
    rgb = ensure_rgb(warped_image)
    try:
        transform = crop_transform_for_image(project_root, image_name, dataset_root)
        x1, y1, x2, y2 = _clip_bbox(transform, rgb.shape)
        crop_source = "dataset_manifest_maskcrop"
    except Exception:
        fish_mask, fish_bbox, _quality = segment_fish_from_blue_board(rgb)
        if fish_bbox is None:
            x1, y1, x2, y2 = 0, 0, rgb.shape[1], rgb.shape[0]
            crop_source = "full_image_fallback"
        else:
            x1, y1, x2, y2 = padded_bbox(fish_bbox, rgb.shape, padding_ratio=0.08)
            crop_source = "segmentation_maskcrop_fallback"
    crop = rgb[y1:y2, x1:x2]
    crop_points, confidences, debug = predict_keypoints_heatmap_unet(crop, selected_model, keypoint_schema)
    # If the manifest bbox was clipped by image bounds, use the actual clipped crop
    # size while preserving the original origin.
    actual_transform = CropTransform(
        bbox_x1=float(x1),
        bbox_y1=float(y1),
        bbox_x2=float(x2),
        bbox_y2=float(y2),
        crop_width=float(x2 - x1),
        crop_height=float(y2 - y1),
        input_size=int(debug.get("input_size", 512)),
    )
    warped_points = {
        key: actual_transform.resized_to_warped(point)
        for key, point in crop_points.items()
    }
    unreliable = {
        key: bool(
            point[0] < 0
            or point[1] < 0
            or point[0] >= actual_transform.input_size
            or point[1] >= actual_transform.input_size
            or confidences.get(key, 0.0) < 0.03
        )
        for key, point in crop_points.items()
    }
    debug.update(
        {
            "image_name": image_name,
            "crop_box": [x1, y1, x2, y2],
            "crop_source": crop_source,
            "coordinate_transform_info": {
                "resized_maskcrop_to_warped": {
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "crop_width": x2 - x1,
                    "crop_height": y2 - y1,
                    "input_size": actual_transform.input_size,
                }
            },
            "unreliable_keypoints": unreliable,
        }
    )
    return {
        "heatmap_keypoints_crop": crop_points,
        "heatmap_keypoints_warped": warped_points,
        "heatmap_confidences": confidences,
        "heatmap_debug_info": debug,
    }
