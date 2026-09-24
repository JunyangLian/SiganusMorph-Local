"""Mark currently retained HEIC files as the real annotation/training set."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph.real_dataset import REAL_SOURCE_DIR, load_manifest, save_manifest  # noqa: E402


KEEP_NOTE = "curated_keep_68_from_folder_2026-05-16"
EXCLUDE_NOTE = "excluded_not_in_curated_68_folder_2026-05-16"


def main() -> None:
    source_dir = REAL_SOURCE_DIR
    current_files = {
        path.name
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".heic", ".heif"}
    }
    if not current_files:
        raise SystemExit(f"No HEIC/HEIF files found in {source_dir}")

    df = load_manifest(PROJECT_ROOT)
    if df.empty:
        raise SystemExit("data/real_images_manifest.csv not found or empty.")

    in_current_folder = df["original_file_name"].astype(str).isin(current_files)
    missing_in_manifest = sorted(current_files - set(df["original_file_name"].astype(str)))
    if missing_in_manifest:
        raise SystemExit(f"These current-folder files are missing from manifest: {missing_in_manifest}")

    df.loc[in_current_folder, "keep_for_annotation"] = "true"
    df.loc[in_current_folder, "keep_for_training"] = "true"
    df.loc[in_current_folder, "exclude_reason"] = ""
    df.loc[in_current_folder, "duplicate_group"] = ""
    df.loc[in_current_folder & df["curation_notes"].astype(str).eq(""), "curation_notes"] = KEEP_NOTE

    excluded = ~in_current_folder
    df.loc[excluded, "keep_for_annotation"] = "false"
    df.loc[excluded, "keep_for_training"] = "false"
    df.loc[excluded & df["exclude_reason"].astype(str).eq(""), "exclude_reason"] = "not_in_curated_folder"
    df.loc[excluded & df["curation_notes"].astype(str).eq(""), "curation_notes"] = EXCLUDE_NOTE

    warp_failed = df["warp_success"].astype(str).str.lower().eq("false")
    df.loc[warp_failed, "keep_for_training"] = "false"
    df.loc[warp_failed & df["exclude_reason"].astype(str).eq(""), "exclude_reason"] = "warp_failed"

    manifest_path = save_manifest(PROJECT_ROOT, df)
    kept = int((df["keep_for_annotation"].astype(str).str.lower() == "true").sum())
    train_kept = int((df["keep_for_training"].astype(str).str.lower() == "true").sum())
    print(f"Current HEIC files: {len(current_files)}")
    print(f"Marked keep_for_annotation=true: {kept}")
    print(f"Marked keep_for_training=true: {train_kept}")
    print(f"Excluded from annotation/training: {len(df) - kept}")
    print(f"Updated manifest: {manifest_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
