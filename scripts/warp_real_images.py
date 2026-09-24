"""Warp real imported PNG images using the current ArUco/board-coordinate pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph.real_dataset import relpath, warp_real_images  # noqa: E402


def main() -> None:
    result = warp_real_images(PROJECT_ROOT)
    print(f"Warp success: {result['successes']}")
    print(f"Warp failed: {result['failures']}")
    print(f"Updated manifest: {relpath(result['manifest'], PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
