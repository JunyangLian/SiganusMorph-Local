"""Import real HEIC/HEIF rabbitfish photos as PNG and create a manifest."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph.real_dataset import (  # noqa: E402
    REAL_SOURCE_DIR,
    discover_heic_files,
    imported_manifest_columns,
    read_existing_manual_fields,
    real_manifest_path,
    real_raw_dir,
    relpath,
    save_manifest,
)


def register_heif() -> None:
    try:
        from pillow_heif import register_heif_opener
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pillow-heif. Install it with `pip install pillow-heif` "
            "or run `pip install -r requirements.txt`."
        ) from exc
    register_heif_opener()


def main() -> None:
    register_heif()
    source_dir = REAL_SOURCE_DIR
    if not source_dir.exists():
        raise SystemExit(f"Source directory not found: {source_dir}")

    files = discover_heic_files(source_dir)
    if not files:
        raise SystemExit(f"No HEIC/HEIF files found in: {source_dir}")

    output_dir = real_raw_dir(PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_manual = read_existing_manual_fields(PROJECT_ROOT)
    rows: list[dict[str, str | int]] = []

    for index, source_path in enumerate(files, start=1):
        image_name = f"real_{index:03d}.png"
        output_path = output_dir / image_name
        with Image.open(source_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.save(output_path, format="PNG")
            width, height = image.size

        manual = existing_manual.get(source_path.name, {})
        rows.append(
            {
                "image_name": image_name,
                "original_file_name": source_path.name,
                "original_path": str(source_path),
                "source_type": "real",
                "specimen_id": manual.get("specimen_id", ""),
                "view_id": manual.get("view_id", ""),
                "imported_path": relpath(output_path, PROJECT_ROOT),
                "width": width,
                "height": height,
                "warped_image_path": "",
                "warp_success": "",
                "needs_review": "false",
                "review_reason": "",
                "notes": manual.get("notes", ""),
                "keep_for_annotation": manual.get("keep_for_annotation", "true"),
                "keep_for_training": manual.get("keep_for_training", "true"),
                "duplicate_group": manual.get("duplicate_group", ""),
                "exclude_reason": manual.get("exclude_reason", ""),
                "quality_score": manual.get("quality_score", ""),
                "curation_notes": manual.get("curation_notes", ""),
            }
        )

    df = pd.DataFrame(rows, columns=imported_manifest_columns())
    manifest = save_manifest(PROJECT_ROOT, df)
    print(f"Imported {len(rows)} HEIC/HEIF images.")
    print(f"Output directory: {relpath(output_dir, PROJECT_ROOT)}/")
    print(f"Manifest: {relpath(manifest, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
