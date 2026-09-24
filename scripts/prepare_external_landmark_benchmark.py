"""Export corrected SiganusMorph labels for external landmark-model benchmarks."""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph.config import KEYPOINT_DEFS


YOLO_DATASET = PROJECT_ROOT / "datasets" / "siganusmorph_real_5_15_yolopose_maskcrop_v0.3_corrected"
CORRECTED_DIR = PROJECT_ROOT / "results" / "real_annotation_5_15_v0.3_corrected" / "keypoints"
DLC_OUT = PROJECT_ROOT / "datasets" / "siganusmorph_dlc_v0.1"
SLEAP_OUT = PROJECT_ROOT / "datasets" / "siganusmorph_sleap_v0.1"
HEATMAP_OUT = PROJECT_ROOT / "datasets" / "siganusmorph_heatmap_unet_v0.1"
SCORER = "SiganusMorph"
HEATMAP_SIZE = 512
HEATMAP_SIGMA = 4.0
KEYPOINT_KEYS = tuple(f"{definition.code}_{definition.name}" for definition in KEYPOINT_DEFS)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def base_from_image_name(image_name: str) -> str:
    stem = Path(image_name).stem
    changed = True
    while changed:
        changed = False
        for suffix in ("_512", "_warped", "_maskcrop"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                changed = True
    return stem


def xy(value: Any) -> tuple[float, float] | None:
    if isinstance(value, Mapping) and "x" in value and "y" in value:
        return float(value["x"]), float(value["y"])
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return None


def load_records() -> list[dict[str, Any]]:
    split_df = pd.read_csv(YOLO_DATASET / "split_manifest.csv")
    crop_df = pd.read_csv(YOLO_DATASET / "crop_metadata.csv")
    crop_by_image = {str(row["image_name"]): row.to_dict() for _, row in crop_df.iterrows()}
    records: list[dict[str, Any]] = []
    for _, row in split_df.iterrows():
        image_name = str(row["image_name"])
        base = base_from_image_name(image_name)
        json_path = CORRECTED_DIR / f"{base}_keypoints.json"
        crop_meta = crop_by_image.get(image_name)
        if not json_path.exists() or crop_meta is None:
            continue
        payload = read_json(json_path)
        keypoints = payload.get("corrected_keypoints") or payload.get("keypoints") or {}
        if any(key not in keypoints for key in KEYPOINT_KEYS):
            continue
        crop_path = PROJECT_ROOT / str(crop_meta["crop_image_path"])
        if not crop_path.exists():
            continue
        bbox = (
            float(crop_meta["bbox_x1"]),
            float(crop_meta["bbox_y1"]),
            float(crop_meta["bbox_x2"]),
            float(crop_meta["bbox_y2"]),
        )
        crop_points: dict[str, list[float]] = {}
        for key in KEYPOINT_KEYS:
            point = xy(keypoints.get(key))
            if point is None:
                continue
            crop_points[key] = [point[0] - bbox[0], point[1] - bbox[1]]
        records.append(
            {
                "image_name": image_name,
                "base": base,
                "specimen_id": str(row.get("specimen_id", "")),
                "split": str(row.get("split", "")),
                "source_type": "real",
                "crop_image_path": crop_path,
                "warped_image_path": PROJECT_ROOT / str(crop_meta["source_warped_image_path"]),
                "bbox": bbox,
                "crop_width": int(crop_meta["crop_width"]),
                "crop_height": int(crop_meta["crop_height"]),
                "warped_keypoints": {key: list(map(float, xy(keypoints[key]))) for key in KEYPOINT_KEYS},
                "crop_keypoints": crop_points,
            }
        )
    return records


def keypoint_schema() -> dict[str, Any]:
    return {
        "schema_version": "v0.3_16kp",
        "coordinate_space": "maskcrop",
        "keypoints": [
            {"index": idx, "name": key, "code": key.split("_", 1)[0]}
            for idx, key in enumerate(KEYPOINT_KEYS)
        ],
        "notes": "P7V is derived and intentionally excluded from external landmark-model training.",
    }


def copy_crop(record: Mapping[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(record["crop_image_path"])
    dst = out_dir / src.name
    shutil.copy2(src, dst)
    return dst


def export_common_manifests(records: list[dict[str, Any]], out_dir: Path) -> None:
    rows = []
    for record in records:
        rows.append(
            {
                "image_name": record["image_name"],
                "specimen_id": record["specimen_id"],
                "split": record["split"],
                "source_type": "real",
                "crop_image_path": str(record["crop_image_path"].relative_to(PROJECT_ROOT)),
                "warped_image_path": str(record["warped_image_path"].relative_to(PROJECT_ROOT)),
                "bbox_x1": record["bbox"][0],
                "bbox_y1": record["bbox"][1],
                "bbox_x2": record["bbox"][2],
                "bbox_y2": record["bbox"][3],
                "crop_width": record["crop_width"],
                "crop_height": record["crop_height"],
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "split_manifest.csv", index=False, encoding="utf-8-sig")
    write_json(out_dir / "keypoint_schema.json", keypoint_schema())


def export_dlc(records: list[dict[str, Any]]) -> None:
    labeled_dir = DLC_OUT / "labeled-data" / "siganusmorph"
    rows: dict[str, list[float]] = {}
    for record in records:
        dst = copy_crop(record, labeled_dir)
        rel = str(Path("labeled-data") / "siganusmorph" / dst.name).replace("\\", "/")
        values: list[float] = []
        for key in KEYPOINT_KEYS:
            point = record["crop_keypoints"][key]
            values.extend([point[0], point[1]])
        rows[rel] = values
    columns = pd.MultiIndex.from_product([[SCORER], KEYPOINT_KEYS, ["x", "y"]], names=["scorer", "bodyparts", "coords"])
    df = pd.DataFrame.from_dict(rows, orient="index", columns=columns)
    DLC_OUT.mkdir(parents=True, exist_ok=True)
    csv_path = DLC_OUT / f"CollectedData_{SCORER}.csv"
    h5_path = DLC_OUT / f"CollectedData_{SCORER}.h5"
    df.to_csv(csv_path)
    try:
        df.to_hdf(h5_path, key="df_with_missing", mode="w")
        h5_note = f"HDF written: {h5_path.name}"
    except Exception as exc:
        h5_note = f"HDF not written because pandas/PyTables failed: {exc}"
    export_common_manifests(records, DLC_OUT)
    (DLC_OUT / "README_DLC_export.md").write_text(
        f"""# SiganusMorph DeepLabCut Export v0.1

Images are maskcrop inputs copied to `labeled-data/siganusmorph/`.

Labels:

- `{csv_path.name}`
- `{h5_path.name}` ({h5_note})

Bodyparts: {len(KEYPOINT_KEYS)} manual landmarks. P7V is derived and excluded.

Suggested workflow:

1. Create a new DeepLabCut project.
2. Copy this `labeled-data/siganusmorph/` folder into the project.
3. Copy `CollectedData_{SCORER}.csv` and, if available, `CollectedData_{SCORER}.h5`.
4. Configure `config.yaml` bodyparts to match `keypoint_schema.json`.
5. Keep horizontal flip disabled because fish head/tail semantics are directional.
6. Use `split_manifest.csv` to reproduce the specimen_id train/val/test split.
""",
        encoding="utf-8",
    )


def export_sleap(records: list[dict[str, Any]]) -> None:
    images_dir = SLEAP_OUT / "images"
    frames = []
    for record in records:
        dst = copy_crop(record, images_dir / record["split"])
        frames.append(
            {
                "image_name": record["image_name"],
                "specimen_id": record["specimen_id"],
                "split": record["split"],
                "image_path": str(dst.relative_to(SLEAP_OUT)).replace("\\", "/"),
                "instances": [
                    {
                        "track": "rabbitfish",
                        "points": {
                            key: {
                                "x": float(record["crop_keypoints"][key][0]),
                                "y": float(record["crop_keypoints"][key][1]),
                                "visible": True,
                            }
                            for key in KEYPOINT_KEYS
                        },
                    }
                ],
            }
        )
    skeleton = {
        "nodes": list(KEYPOINT_KEYS),
        "edges": [
            ["P1_snout_tip", "C1_head_axis_point"],
            ["C1_head_axis_point", "C2_trunk_axis_point"],
            ["C2_trunk_axis_point", "C3_posterior_trunk_axis_point"],
            ["C3_posterior_trunk_axis_point", "P4_peduncle_start_midpoint"],
            ["P4_peduncle_start_midpoint", "C4_peduncle_axis_point"],
            ["C4_peduncle_axis_point", "P5_caudal_base_midpoint"],
            ["P5_caudal_base_midpoint", "P6_caudal_fork_midpoint"],
            ["P6_caudal_fork_midpoint", "P7U_caudal_fin_upper_tip"],
            ["P6_caudal_fork_midpoint", "P7L_caudal_fin_lower_tip"],
        ],
    }
    write_json(SLEAP_OUT / "siganusmorph_sleap_labels.json", {"skeleton": skeleton, "frames": frames})
    write_json(SLEAP_OUT / "skeleton.json", skeleton)
    export_common_manifests(records, SLEAP_OUT)
    (SLEAP_OUT / "README_SLEAP_export.md").write_text(
        """# SiganusMorph SLEAP Export v0.1

This folder contains maskcrop images and a SLEAP-friendly intermediate JSON:

- `siganusmorph_sleap_labels.json`
- `skeleton.json`
- `split_manifest.csv`
- `keypoint_schema.json`

Suggested workflow:

1. Create a SLEAP project.
2. Import images from `images/train`, `images/val`, and `images/test`.
3. Recreate the skeleton from `skeleton.json`.
4. Use `siganusmorph_sleap_labels.json` as the authoritative label interchange file if direct import needs a small converter for your SLEAP version.
5. Keep train/val/test grouped by specimen_id according to `split_manifest.csv`.
""",
        encoding="utf-8",
    )


def gaussian_heatmap(width: int, height: int, x: float, y: float, sigma: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    return np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma * sigma)).astype(np.float32)


def export_heatmap_unet(records: list[dict[str, Any]]) -> None:
    images_dir = HEATMAP_OUT / "images"
    heatmaps_dir = HEATMAP_OUT / "heatmaps"
    labels: dict[str, Any] = {
        "input_size": [HEATMAP_SIZE, HEATMAP_SIZE],
        "sigma_px": HEATMAP_SIGMA,
        "coordinate_space": "resized_maskcrop",
        "samples": [],
    }
    for record in records:
        image = Image.open(record["crop_image_path"]).convert("RGB")
        resized = image.resize((HEATMAP_SIZE, HEATMAP_SIZE), Image.Resampling.BILINEAR)
        out_image_dir = images_dir / record["split"]
        out_heat_dir = heatmaps_dir / record["split"]
        out_image_dir.mkdir(parents=True, exist_ok=True)
        out_heat_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(record["crop_image_path"]).stem
        image_path = out_image_dir / f"{stem}_{HEATMAP_SIZE}.png"
        heatmap_path = out_heat_dir / f"{stem}_heatmaps.npz"
        resized.save(image_path)
        sx = HEATMAP_SIZE / float(record["crop_width"])
        sy = HEATMAP_SIZE / float(record["crop_height"])
        resized_points: dict[str, list[float]] = {}
        heatmaps = []
        for key in KEYPOINT_KEYS:
            x, y = record["crop_keypoints"][key]
            rx, ry = float(x) * sx, float(y) * sy
            resized_points[key] = [rx, ry]
            heatmaps.append(gaussian_heatmap(HEATMAP_SIZE, HEATMAP_SIZE, rx, ry, HEATMAP_SIGMA))
        np.savez_compressed(heatmap_path, heatmaps=np.stack(heatmaps, axis=0), keypoints=np.asarray([resized_points[k] for k in KEYPOINT_KEYS], dtype=np.float32))
        labels["samples"].append(
            {
                "image_name": record["image_name"],
                "specimen_id": record["specimen_id"],
                "split": record["split"],
                "image_path": str(image_path.relative_to(HEATMAP_OUT)).replace("\\", "/"),
                "heatmap_path": str(heatmap_path.relative_to(HEATMAP_OUT)).replace("\\", "/"),
                "keypoints": resized_points,
                "source_crop_width": record["crop_width"],
                "source_crop_height": record["crop_height"],
            }
        )
    write_json(HEATMAP_OUT / "labels.json", labels)
    export_common_manifests(records, HEATMAP_OUT)
    (HEATMAP_OUT / "README_heatmap_unet_export.md").write_text(
        f"""# SiganusMorph Heatmap U-Net Export v0.1

Images are resized maskcrop inputs at `{HEATMAP_SIZE}x{HEATMAP_SIZE}`.

Each `.npz` in `heatmaps/` contains:

- `heatmaps`: `[16, {HEATMAP_SIZE}, {HEATMAP_SIZE}]` Gaussian heatmaps
- `keypoints`: `[16, 2]` resized keypoint coordinates

Gaussian sigma: {HEATMAP_SIGMA} px.

Use `split_manifest.csv` to keep specimen_id splits identical to v0.3/v0.4.
""",
        encoding="utf-8",
    )


def main() -> None:
    records = load_records()
    if not records:
        raise SystemExit("No corrected maskcrop records found.")
    export_dlc(records)
    export_sleap(records)
    export_heatmap_unet(records)
    print(f"Exported {len(records)} corrected real images for external landmark benchmarks.")
    print(f"DLC: {DLC_OUT.relative_to(PROJECT_ROOT)}/")
    print(f"SLEAP: {SLEAP_OUT.relative_to(PROJECT_ROOT)}/")
    print(f"Heatmap U-Net: {HEATMAP_OUT.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
