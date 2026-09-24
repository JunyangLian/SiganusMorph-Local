"""Train lightweight heatmap U-Net for SiganusMorph landmarks."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph.heatmap_unet import HeatmapDataset, LightweightUNet, set_seed


DATASET_ROOT = PROJECT_ROOT / "datasets" / "siganusmorph_heatmap_unet_v0.1"
MODEL_DIR = PROJECT_ROOT / "models" / "siganusmorph_heatmap_unet_v0.1"


CONFIG = {
    "image_size": 512,
    "epochs": 150,
    "batch_size": 4,
    "patience": 30,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "optimizer": "AdamW",
    "loss": "WeightedMSELoss",
    "positive_heatmap_weight": 200.0,
    "heatmap_sigma": 4.0,
    "seed": 20260515,
    "base_channels": 24,
    "num_keypoints": 16,
    "augmentation": "brightness_contrast, gaussian_noise, rotation_pm5, translation_pm4pct, scale_0.92_1.08, no_horizontal_flip",
}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    lines = []
    for key, value in payload.items():
        if isinstance(value, str):
            lines.append(f"{key}: {value}")
        else:
            lines.append(f"{key}: {json.dumps(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, val_loss: float, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "config": config,
        },
        path,
    )


def weighted_mse_loss(preds: torch.Tensor, targets: torch.Tensor, positive_weight: float) -> torch.Tensor:
    weights = 1.0 + positive_weight * targets
    return torch.mean(weights * (preds - targets) ** 2)


def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device, positive_weight: float, optimizer: torch.optim.Optimizer | None = None, scaler: torch.cuda.amp.GradScaler | None = None) -> float:
    train = optimizer is not None
    model.train(train)
    losses = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["heatmaps"].to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            if scaler is not None and train:
                with torch.cuda.amp.autocast():
                    preds = torch.sigmoid(model(images))
                    loss = weighted_mse_loss(preds, targets, positive_weight)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = torch.sigmoid(model(images))
                loss = weighted_mse_loss(preds, targets, positive_weight)
                if train:
                    loss.backward()
                    optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def save_loss_curve(log_rows: list[dict[str, Any]], path: Path) -> None:
    width, height = 900, 520
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    if not log_rows:
        image.save(path)
        return
    train = np.asarray([row["train_loss"] for row in log_rows], dtype=float)
    val = np.asarray([row["val_loss"] for row in log_rows], dtype=float)
    finite = np.concatenate([train[np.isfinite(train)], val[np.isfinite(val)]])
    ymin = float(finite.min()) if finite.size else 0.0
    ymax = float(finite.max()) if finite.size else 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0
    left, top, right, bottom = 70, 40, width - 30, height - 60
    draw.rectangle((left, top, right, bottom), outline=(0, 0, 0), width=2)

    def map_point(i: int, y: float) -> tuple[float, float]:
        x = left + (right - left) * i / max(1, len(log_rows) - 1)
        yy = bottom - (bottom - top) * (y - ymin) / (ymax - ymin)
        return x, yy

    for values, color in ((train, (40, 120, 255)), (val, (255, 90, 40))):
        pts = [map_point(i, float(v)) for i, v in enumerate(values)]
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=3)
    draw.text((left, height - 40), "blue=train_loss  orange=val_loss", fill=(0, 0, 0))
    draw.text((left, 12), f"loss range {ymin:.6f} - {ymax:.6f}", fill=(0, 0, 0))
    image.save(path)


def main() -> None:
    set_seed(int(CONFIG["seed"]))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DATASET_ROOT / "keypoint_schema.json", MODEL_DIR / "keypoint_schema.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(CONFIG["batch_size"])
    train_dataset = HeatmapDataset(DATASET_ROOT, "train", int(CONFIG["image_size"]), float(CONFIG["heatmap_sigma"]), augment=True)
    val_dataset = HeatmapDataset(DATASET_ROOT, "val", int(CONFIG["image_size"]), float(CONFIG["heatmap_sigma"]), augment=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=device.type == "cuda")
    model = LightweightUNet(out_channels=int(CONFIG["num_keypoints"]), base_channels=int(CONFIG["base_channels"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(CONFIG["learning_rate"]), weight_decay=float(CONFIG["weight_decay"]))
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    run_config = dict(CONFIG)
    run_config.update(
        {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
            "num_train_images": len(train_dataset),
            "num_val_images": len(val_dataset),
            "num_test_images": len(HeatmapDataset(DATASET_ROOT, "test")),
        }
    )
    write_yaml(MODEL_DIR / "train_config.yaml", run_config)
    log_rows: list[dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = 0
    patience_counter = 0
    for epoch in range(1, int(CONFIG["epochs"]) + 1):
        try:
            train_loss = run_epoch(model, train_loader, device, float(CONFIG["positive_heatmap_weight"]), optimizer, scaler)
        except torch.cuda.OutOfMemoryError:
            if batch_size > 2:
                torch.cuda.empty_cache()
                batch_size = 2
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
                run_config["batch_size"] = batch_size
                train_loss = run_epoch(model, train_loader, device, float(CONFIG["positive_heatmap_weight"]), optimizer, scaler)
            else:
                raise
        val_loss = run_epoch(model, val_loader, device, float(CONFIG["positive_heatmap_weight"]))
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "best_val_loss": min(best_val, val_loss)}
        log_rows.append(row)
        print(f"epoch {epoch:03d} train={train_loss:.6f} val={val_loss:.6f}")
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(MODEL_DIR / "best_model.pt", model, optimizer, epoch, val_loss, run_config)
        else:
            patience_counter += 1
        save_checkpoint(MODEL_DIR / "last_model.pt", model, optimizer, epoch, val_loss, run_config)
        if patience_counter >= int(CONFIG["patience"]):
            print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}.")
            break

    with (MODEL_DIR / "training_log.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch", "train_loss", "val_loss", "best_val_loss"])
        writer.writeheader()
        writer.writerows(log_rows)
    save_loss_curve(log_rows, MODEL_DIR / "loss_curve.png")
    (MODEL_DIR / "README_model.md").write_text(
        f"""# SiganusMorph Heatmap U-Net v0.1

Data: `datasets/siganusmorph_heatmap_unet_v0.1/`

Corrected real maskcrop images: {len(train_dataset) + len(val_dataset) + len(HeatmapDataset(DATASET_ROOT, "test"))}

Train/val/test images: {len(train_dataset)} / {len(val_dataset)} / {len(HeatmapDataset(DATASET_ROOT, "test"))}

Keypoints: 16 manual landmarks. P7V is derived and excluded.

Image size: {CONFIG['image_size']}

Heatmap sigma: {CONFIG['heatmap_sigma']} px

Loss: {CONFIG['loss']}

Optimizer: {CONFIG['optimizer']} lr={CONFIG['learning_rate']}

Augmentation: {CONFIG['augmentation']}

Device: {run_config['device']} {run_config['gpu_name']}

Best epoch by val loss: {best_epoch}

Evaluation results are appended after running prediction/evaluation scripts.
""",
        encoding="utf-8",
    )
    print(f"Training complete. Best val loss {best_val:.6f} at epoch {best_epoch}.")
    print(f"Model directory: {MODEL_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
