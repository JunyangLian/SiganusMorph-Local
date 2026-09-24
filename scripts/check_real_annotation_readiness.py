"""Check whether real annotations are ready for YOLO-pose export."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph.real_dataset import check_real_annotation_readiness, real_annotation_dir, relpath  # noqa: E402


def main() -> None:
    report = check_real_annotation_readiness(PROJECT_ROOT)
    ready_count = int(report["ready_for_training"].sum()) if not report.empty else 0
    print(f"Ready for training: {ready_count}/{len(report)}")
    print(f"Report: {relpath(real_annotation_dir(PROJECT_ROOT) / 'readiness_report.csv', PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
