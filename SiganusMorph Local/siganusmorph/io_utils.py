"""Input/output helpers for result artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
from PIL import Image

from .config import RESULT_COLUMNS


def resolve_output_dir(path_text: str | Path, project_root: str | Path | None = None) -> Path:
    """Resolve output directory text to an absolute path."""
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        base = Path(project_root) if project_root is not None else Path.cwd()
        path = base / path
    return path


def ensure_output_dirs(output_dir: str | Path) -> dict[str, Path]:
    """Create result subdirectories and return their paths."""
    root = Path(output_dir)
    paths = {
        "root": root,
        "annotations": root / "annotations",
        "keypoints": root / "keypoints",
        "masks": root / "masks",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def sanitize_filename_stem(text: str) -> str:
    """Make a safe filename stem while preserving readable IDs."""
    cleaned = re.sub(r"[^\w.\-]+", "_", text, flags=re.UNICODE).strip("_")
    return cleaned or "siganus_sample"


def _ordered_dataframe(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    for column in RESULT_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df.loc[:, list(RESULT_COLUMNS)]


def save_results_csv(rows: Iterable[Mapping[str, Any]], output_path: str | Path) -> Path:
    """Save all result rows to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _ordered_dataframe(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_results_excel(rows: Iterable[Mapping[str, Any]], output_path: str | Path) -> Path:
    """Save all result rows to Excel."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _ordered_dataframe(rows).to_excel(path, index=False)
    return path


def upsert_result_row(row: Mapping[str, Any], output_dir: str | Path) -> dict[str, Path]:
    """Upsert one result row into results.csv and results.xlsx."""
    root = Path(output_dir)
    ensure_output_dirs(root)
    csv_path = root / "results.csv"
    xlsx_path = root / "results.xlsx"

    if csv_path.exists():
        existing = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    else:
        existing = pd.DataFrame(columns=list(RESULT_COLUMNS))

    for column in RESULT_COLUMNS:
        if column not in existing.columns:
            existing[column] = ""

    image_name = str(row.get("image_name", ""))
    specimen_id = str(row.get("specimen_id", ""))
    if not existing.empty:
        keep_mask = ~(
            (existing["image_name"].astype(str) == image_name)
            & (existing["specimen_id"].astype(str) == specimen_id)
        )
        existing = existing.loc[keep_mask, list(RESULT_COLUMNS)]

    new_row = _ordered_dataframe([row])
    if existing.empty:
        combined = new_row
    else:
        combined = pd.concat([existing, new_row], ignore_index=True)
    combined = combined.loc[:, list(RESULT_COLUMNS)]
    combined.to_csv(csv_path, index=False, encoding="utf-8-sig")
    combined.to_excel(xlsx_path, index=False)
    return {"csv": csv_path, "xlsx": xlsx_path}


def save_annotation_image(image, output_path: str | Path) -> Path:
    """Save a drawn annotation image."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
    return path
