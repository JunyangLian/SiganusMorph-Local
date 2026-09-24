from __future__ import annotations

from pathlib import Path

import cv2
import streamlit
import torch
import ultralytics


ROOT = Path(__file__).resolve().parent
REQUIRED_MODELS = (
    ROOT / "models" / "siganusmorph_heatmap_unet_v0.1" / "preannotation_candidate.pt",
    ROOT / "models" / "siganusmorph_heatmap_unet_v0.5" / "preannotation_candidate.pt",
    ROOT
    / "models"
    / "siganusmorph_yolopose_v0.3_corrected_real_5_15"
    / "preannotation_candidate.pt",
)


def main() -> None:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED_MODELS if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing required model files: {missing}")
    if not hasattr(cv2, "aruco"):
        raise SystemExit("OpenCV ArUco support is unavailable")

    print(f"Streamlit: {streamlit.__version__}")
    print(f"OpenCV: {cv2.__version__}; ArUco: yes")
    print(f"PyTorch: {torch.__version__}; CUDA available: {torch.cuda.is_available()}")
    print(f"Ultralytics: {ultralytics.__version__}")
    print("Required model files: present")


if __name__ == "__main__":
    main()
