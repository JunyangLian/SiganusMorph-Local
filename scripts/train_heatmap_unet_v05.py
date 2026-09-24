"""Train or fine-tune heatmap U-Net v0.5 on the candidate dataset."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from siganusmorph.heatmap_unet import HeatmapDataset, LightweightUNet, set_seed  # noqa: E402
from train_heatmap_unet import run_epoch, save_checkpoint, save_loss_curve, write_yaml  # noqa: E402


DATASET_ROOT = PROJECT_ROOT / "datasets" / "siganusmorph_heatmap_unet_v0.5_candidate"
MODEL_DIR = PROJECT_ROOT / "models" / "siganusmorph_heatmap_unet_v0.5"
PRETRAINED = PROJECT_ROOT / "models" / "siganusmorph_heatmap_unet_v0.1" / "best_model.pt"


CONFIG: dict[str, Any] = {
    "image_size": 512,
    "epochs": 150,
    "batch_size": 4,
    "patience": 35,
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "optimizer": "AdamW",
    "loss": "WeightedMSELoss",
    "positive_heatmap_weight": 200.0,
    "heatmap_sigma": 4.0,
    "seed": 20260515,
    "base_channels": 24,
    "num_keypoints": 16,
    "pretrained_checkpoint": str(PRETRAINED),
    "augmentation": "brightness_contrast, gaussian_noise, rotation_pm5, translation_pm4pct, scale_0.92_1.08, no_horizontal_flip",
}


def _load_pretrained(model: nn.Module, path: Path, device: torch.device) -> tuple[bool, str]:
    if not path.exists():
        return False, f"pretrained checkpoint not found: {path}"
    try:
        checkpoint = torch.load(path, map_location=device)
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state, strict=True)
        return True, f"loaded pretrained checkpoint: {path}"
    except Exception as exc:  # noqa: BLE001
        return False, f"failed to load pretrained checkpoint: {exc}"


def main() -> None:
    set_seed(int(CONFIG["seed"]))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DATASET_ROOT / "keypoint_schema.json", MODEL_DIR / "keypoint_schema.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(CONFIG["batch_size"])
    train_dataset = HeatmapDataset(DATASET_ROOT, "train", int(CONFIG["image_size"]), float(CONFIG["heatmap_sigma"]), augment=True)
    val_dataset = HeatmapDataset(DATASET_ROOT, "val", int(CONFIG["image_size"]), float(CONFIG["heatmap_sigma"]), augment=False)
    test_dataset = HeatmapDataset(DATASET_ROOT, "test", int(CONFIG["image_size"]), float(CONFIG["heatmap_sigma"]), augment=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=device.type == "cuda")

    model = LightweightUNet(out_channels=int(CONFIG["num_keypoints"]), base_channels=int(CONFIG["base_channels"])).to(device)
    pretrained_loaded, pretrained_note = _load_pretrained(model, PRETRAINED, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(CONFIG["learning_rate"]), weight_decay=float(CONFIG["weight_decay"]))
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    run_config = dict(CONFIG)
    run_config.update(
        {
            "dataset_root": str(DATASET_ROOT.relative_to(PROJECT_ROOT)),
            "model_dir": str(MODEL_DIR.relative_to(PROJECT_ROOT)),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
            "num_train_images": len(train_dataset),
            "num_val_images": len(val_dataset),
            "num_test_images": len(test_dataset),
            "pretrained_loaded": pretrained_loaded,
            "pretrained_note": pretrained_note,
        }
    )
    write_yaml(MODEL_DIR / "train_config.yaml", run_config)
    print(pretrained_note)

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
        f"""# SiganusMorph Heatmap U-Net v0.5

Data: `datasets/siganusmorph_heatmap_unet_v0.5_candidate/`

Train/val/test images: {len(train_dataset)} / {len(val_dataset)} / {len(test_dataset)}

Keypoints: 16 manual landmarks. P7V is derived and excluded.

Pretrained checkpoint: `{PRETRAINED.relative_to(PROJECT_ROOT)}`; loaded={pretrained_loaded}

Image size: {CONFIG['image_size']}

Heatmap sigma: {CONFIG['heatmap_sigma']} px

Loss: {CONFIG['loss']}

Optimizer: {CONFIG['optimizer']} lr={CONFIG['learning_rate']}

Augmentation: {CONFIG['augmentation']}

Device: {run_config['device']} {run_config['gpu_name']}

Best epoch by val loss: {best_epoch}

Evaluation results are added by the prediction and comparison scripts.
""",
        encoding="utf-8",
    )
    print(f"Training complete. Best val loss {best_val:.6f} at epoch {best_epoch}.")
    print(f"Model directory: {MODEL_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
