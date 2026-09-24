"""Lightweight heatmap U-Net for SiganusMorph landmark experiments."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset


KEYPOINT_COUNT = 16


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LightweightUNet(nn.Module):
    def __init__(self, out_channels: int = KEYPOINT_COUNT, base_channels: int = 24) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(3, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c * 8, c * 12)
        self.up4 = nn.ConvTranspose2d(c * 12, c * 8, 2, stride=2)
        self.dec4 = ConvBlock(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = ConvBlock(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)
        self.head = nn.Conv2d(c, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def gaussian_heatmaps(points: np.ndarray, size: int, sigma: float) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    heatmaps = []
    for x, y in points:
        heatmaps.append(np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma * sigma)).astype(np.float32))
    return np.stack(heatmaps, axis=0)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class HeatmapSample:
    image_name: str
    specimen_id: str
    split: str
    image_path: Path
    heatmap_path: Path
    keypoints: np.ndarray
    source_crop_width: int
    source_crop_height: int


def load_samples(dataset_root: Path, split: str | None = None) -> list[HeatmapSample]:
    import json

    payload = json.loads((dataset_root / "labels.json").read_text(encoding="utf-8"))
    samples: list[HeatmapSample] = []
    for item in payload["samples"]:
        if split is not None and item["split"] != split:
            continue
        keypoints = np.asarray(list(item["keypoints"].values()), dtype=np.float32)
        samples.append(
            HeatmapSample(
                image_name=str(item["image_name"]),
                specimen_id=str(item["specimen_id"]),
                split=str(item["split"]),
                image_path=dataset_root / item["image_path"],
                heatmap_path=dataset_root / item["heatmap_path"],
                keypoints=keypoints,
                source_crop_width=int(item["source_crop_width"]),
                source_crop_height=int(item["source_crop_height"]),
            )
        )
    return samples


class HeatmapDataset(Dataset):
    def __init__(self, dataset_root: Path, split: str, image_size: int = 512, sigma: float = 4.0, augment: bool = False) -> None:
        self.dataset_root = dataset_root
        self.samples = load_samples(dataset_root, split)
        self.image_size = image_size
        self.sigma = sigma
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, image: np.ndarray, keypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        size = self.image_size
        # Photometric jitter.
        alpha = random.uniform(0.85, 1.15)
        beta = random.uniform(-12, 12)
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        if random.random() < 0.35:
            noise = np.random.normal(0, random.uniform(1.0, 4.0), image.shape).astype(np.float32)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Small affine transform. No horizontal flip.
        angle = random.uniform(-5.0, 5.0)
        scale = random.uniform(0.92, 1.08)
        tx = random.uniform(-0.04, 0.04) * size
        ty = random.uniform(-0.04, 0.04) * size
        center = (size / 2.0, size / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, scale).astype(np.float32)
        matrix[:, 2] += (tx, ty)
        image = cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        ones = np.ones((keypoints.shape[0], 1), dtype=np.float32)
        keypoints = np.concatenate([keypoints, ones], axis=1) @ matrix.T
        keypoints[:, 0] = np.clip(keypoints[:, 0], 0, size - 1)
        keypoints[:, 1] = np.clip(keypoints[:, 1], 0, size - 1)
        return image, keypoints.astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = np.asarray(Image.open(sample.image_path).convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR))
        keypoints = sample.keypoints.copy()
        if self.augment:
            image, keypoints = self._augment(image, keypoints)
            heatmaps = gaussian_heatmaps(keypoints, self.image_size, self.sigma)
        else:
            heatmaps = np.load(sample.heatmap_path)["heatmaps"].astype(np.float32)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32) / 255.0)
        return {
            "image": image_tensor,
            "heatmaps": torch.from_numpy(heatmaps),
            "keypoints": torch.from_numpy(keypoints.astype(np.float32)),
            "image_name": sample.image_name,
            "specimen_id": sample.specimen_id,
            "split": sample.split,
            "source_crop_width": sample.source_crop_width,
            "source_crop_height": sample.source_crop_height,
        }


def heatmaps_to_points(heatmaps: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Argmax heatmaps to resized-image coordinates and peak confidences."""
    if heatmaps.ndim == 3:
        heatmaps = heatmaps.unsqueeze(0)
    batch, channels, height, width = heatmaps.shape
    flat = heatmaps.reshape(batch, channels, -1)
    values, indices = torch.max(flat, dim=2)
    ys = torch.div(indices, width, rounding_mode="floor").float()
    xs = (indices % width).float()
    points = torch.stack([xs, ys], dim=2)
    return points.cpu().numpy(), values.cpu().numpy()
