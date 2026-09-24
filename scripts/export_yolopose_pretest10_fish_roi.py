"""Export the ready 10-image pretest dataset cropped to the fish-placement ROI."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph.real_dataset import export_yolopose_dataset, relpath  # noqa: E402


def main() -> None:
    result = export_yolopose_dataset(
        PROJECT_ROOT,
        train_count=8,
        val_count=1,
        test_count=1,
        dataset_root=Path("datasets") / "siganusmorph_real_5_15_yolopose_pretest10_fish_roi",
        crop_to_fish_roi=True,
        fish_roi_padding_mm=5.0,
    )
    print(f"Exported fish-ROI YOLO-pose pretest dataset: {relpath(result['dataset_root'], PROJECT_ROOT)}")
    print(f"Images: {result['num_images']}")
    print(f"Specimens: {result['num_specimens']}")


if __name__ == "__main__":
    main()
