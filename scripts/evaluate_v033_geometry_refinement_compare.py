"""Run v0.3.3 geometry-refinement comparison on the same v0.3.2 sample."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

os.environ["SIGANUS_COMPARE_LABEL"] = "v0.3.3_geometry_refinement"
os.environ["SIGANUS_COMPARE_OUTPUT_DIR"] = str(
    PROJECT_ROOT / "results" / "model_eval" / "v0.3.3_geometry_refinement_compare_30"
)
os.environ["SIGANUS_COMPARE_SAMPLE_MANIFEST"] = str(
    PROJECT_ROOT / "results" / "model_eval" / "v0.3.2_tail_geometry_compare_30" / "sample_manifest.csv"
)

from evaluate_v031_preannotation_compare import main


if __name__ == "__main__":
    main()
