from __future__ import annotations

from pathlib import Path
from typing import Any

from .session_store import PROJECT_ROOT, V2_OUTPUT_ROOT


def project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def model_info(name: str, relative_path: str) -> dict[str, Any]:
    path = PROJECT_ROOT / relative_path
    return {
        "name": name,
        "path": relative_path.replace("\\", "/"),
        "exists": path.exists(),
    }


def get_v2_settings() -> dict[str, Any]:
    """Return user-safe V2 settings with project-relative paths only."""
    admin_output = PROJECT_ROOT / "results" / "v2_admin_outputs"
    return {
        "version": "2.0.0",
        "app": "SiganusMorph Local V2.0",
        "core_engine": "siganusmorph/",
        "backend_output_root": project_relative(V2_OUTPUT_ROOT),
        "user_output_root": "results/v2_user_outputs/",
        "admin_output_root": project_relative(admin_output),
        "model_paths": [
            model_info("heatmap_unet_v0.1_candidate", "models/siganusmorph_heatmap_unet_v0.1/preannotation_candidate.pt"),
            model_info("heatmap_unet_v0.5_candidate", "models/siganusmorph_heatmap_unet_v0.5/preannotation_candidate.pt"),
        ],
        "notes": [
            "V2.0 user outputs are isolated from V1.0 historical results.",
            "The user app reuses the existing siganusmorph Python measurement core.",
            "Model paths are reported for transparency and are not editable from the ordinary user app.",
        ],
    }
