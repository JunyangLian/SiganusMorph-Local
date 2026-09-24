"""Export ready real annotations to a YOLO-pose dataset split by specimen_id."""

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
        train_count=24,
        val_count=5,
        test_count=5,
        crop_to_fish_roi=True,
        fish_roi_padding_mm=5.0,
    )
    print(f"Exported YOLO-pose dataset: {relpath(result['dataset_root'], PROJECT_ROOT)}")
    print(f"Images: {result['num_images']}")
    print(f"Specimens: {result['num_specimens']}")
    print("Input mode: fish ROI crop")


if __name__ == "__main__":
    main()
