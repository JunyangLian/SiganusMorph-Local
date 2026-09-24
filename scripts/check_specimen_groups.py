"""Check manually assigned specimen_id groups in the real image manifest."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph.real_dataset import check_specimen_groups, relpath  # noqa: E402


def main() -> None:
    result = check_specimen_groups(PROJECT_ROOT)
    print(f"Missing specimen_id images: {result['missing_count']}")
    print(f"Unique specimen_id count: {result['num_specimens']}")
    print(f"Summary: {relpath(result['summary_path'], PROJECT_ROOT)}")
    if result["num_specimens"] and result["num_specimens"] != 35:
        print("Note: expected about 35 fish; please review grouping.")


if __name__ == "__main__":
    main()
