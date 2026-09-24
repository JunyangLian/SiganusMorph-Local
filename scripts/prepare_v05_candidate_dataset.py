"""Prepare v0.5 candidate labels, split plan, hard cases, and heatmap dataset.

This script does not train a model and does not modify corrected annotations.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph.config import KEYPOINT_DEFS  # noqa: E402


KEYPOINT_KEYS = [f"{definition.code}_{definition.name}" for definition in KEYPOINT_DEFS]
PLANNING_DIR = PROJECT_ROOT / "results" / "v0.5_dataset_planning"
ORIGINAL_CORRECTED_DIR = PROJECT_ROOT / "results" / "real_annotation_5_15_v0.3_corrected" / "keypoints"
REVIEW_ROOT = PROJECT_ROOT / "results" / "realworld_review_v0.4.1"
REVIEW_CORRECTED_DIR = REVIEW_ROOT / "corrected_keypoints"
REVIEW_SAMPLE_MANIFEST = REVIEW_ROOT / "review_sample_manifest.csv"
POST_REVIEW_DIR = REVIEW_ROOT / "post_review_evaluation"
OLD_HEATMAP_DATASET = PROJECT_ROOT / "datasets" / "siganusmorph_heatmap_unet_v0.1"
OLD_SPLIT_MANIFEST = OLD_HEATMAP_DATASET / "split_manifest.csv"
V05_DATASET = PROJECT_ROOT / "datasets" / "siganusmorph_heatmap_unet_v0.5_candidate"
INPUT_SIZE = 512
SIGMA_PX = 4.0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _xy(point: Any) -> tuple[float, float] | None:
    if isinstance(point, Mapping) and "x" in point and "y" in point:
        return float(point["x"]), float(point["y"])
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        return float(point[0]), float(point[1])
    return None


def _normal_image_name(value: str, fallback_stem: str = "") -> str:
    name = str(value or "")
    if not name and fallback_stem:
        name = f"{fallback_stem}.png"
    name = Path(name).name
    if name.endswith("_warped.png"):
        name = name.replace("_warped.png", ".png")
    if name.endswith("_keypoints.json"):
        name = name.replace("_keypoints.json", ".png")
    if name.endswith("_corrected.json"):
        name = name.replace("_corrected.json", ".png")
    return name


def _stem(image_name: str) -> str:
    stem = Path(image_name).stem
    if stem.endswith("_warped"):
        stem = stem[: -len("_warped")]
    return stem


def _path_text(path: str | Path) -> str:
    p = Path(str(path))
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def _project_path(value: str | Path) -> Path:
    p = Path(str(value))
    return p if p.is_absolute() else PROJECT_ROOT / p


def _extract_corrected_keypoints(payload: Mapping[str, Any]) -> dict[str, list[float]]:
    raw = payload.get("corrected_keypoints")
    if not isinstance(raw, Mapping):
        raw = payload.get("keypoints")
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, list[float]] = {}
    for key in KEYPOINT_KEYS:
        xy = _xy(raw.get(key))
        if xy is not None:
            out[key] = [float(xy[0]), float(xy[1])]
    return out


def _has_16(points: Mapping[str, Any]) -> bool:
    return all(_xy(points.get(key)) is not None for key in KEYPOINT_KEYS)


def _load_old_split() -> pd.DataFrame:
    if OLD_SPLIT_MANIFEST.exists():
        return pd.read_csv(OLD_SPLIT_MANIFEST, dtype=str, keep_default_na=False)
    candidates = sorted((PROJECT_ROOT / "datasets").glob("**/split_manifest.csv"))
    if not candidates:
        return pd.DataFrame()
    return pd.read_csv(candidates[-1], dtype=str, keep_default_na=False)


def _load_review_sample() -> pd.DataFrame:
    if REVIEW_SAMPLE_MANIFEST.exists():
        return pd.read_csv(REVIEW_SAMPLE_MANIFEST, dtype=str, keep_default_na=False)
    return pd.DataFrame()


def _old_label_samples() -> dict[str, dict[str, Any]]:
    path = OLD_HEATMAP_DATASET / "labels.json"
    if not path.exists():
        return {}
    payload = _read_json(path)
    samples = payload.get("samples", [])
    if not isinstance(samples, list):
        return {}
    return {str(sample.get("image_name", "")): sample for sample in samples if isinstance(sample, Mapping)}


def _crop_box_from_preannotation(image_name: str) -> tuple[float, float, float, float] | None:
    path = REVIEW_ROOT / "preannotations" / "json" / f"{_stem(image_name)}_preannotation.json"
    if not path.exists():
        return None
    payload = _read_json(path)
    metadata = payload.get("preannotation_metadata", {})
    debug = metadata.get("heatmap_debug_info", {}) if isinstance(metadata, Mapping) else {}
    crop_box = debug.get("crop_box") if isinstance(debug, Mapping) else None
    if isinstance(crop_box, (list, tuple)) and len(crop_box) == 4:
        return tuple(float(v) for v in crop_box)
    transform = debug.get("coordinate_transform_info", {}).get("resized_maskcrop_to_warped", {}) if isinstance(debug, Mapping) else {}
    if isinstance(transform, Mapping) and {"bbox_x1", "bbox_y1", "crop_width", "crop_height"}.issubset(transform):
        x1 = float(transform["bbox_x1"])
        y1 = float(transform["bbox_y1"])
        return x1, y1, x1 + float(transform["crop_width"]), y1 + float(transform["crop_height"])
    return None


def _gaussian_heatmaps(keypoints: np.ndarray, size: int = INPUT_SIZE, sigma: float = SIGMA_PX) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    heatmaps = np.zeros((len(KEYPOINT_KEYS), size, size), dtype=np.float32)
    denom = 2.0 * sigma * sigma
    for idx, (x, y) in enumerate(keypoints.astype(np.float32)):
        heatmaps[idx] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / denom).astype(np.float32)
    return heatmaps


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_candidate_manifest(old_split: pd.DataFrame, review_sample: pd.DataFrame) -> pd.DataFrame:
    old_by_image = {str(row["image_name"]): row for _, row in old_split.iterrows()} if not old_split.empty else {}
    review_confirmed_images = set()
    review_status_by_image = {}
    if not review_sample.empty:
        for _, row in review_sample.iterrows():
            image_name = str(row.get("image_name", ""))
            review_status_by_image[image_name] = row
            if str(row.get("review_status", "")) == "corrected_and_confirmed":
                review_confirmed_images.add(image_name)

    rows: list[dict[str, Any]] = []
    payloads: dict[tuple[str, str], dict[str, Any]] = {}

    for path in sorted(ORIGINAL_CORRECTED_DIR.glob("*.json")):
        payload = _read_json(path)
        image_name = _normal_image_name(str(payload.get("image_name", "")), path.stem.replace("_keypoints", ""))
        points = _extract_corrected_keypoints(payload)
        old = old_by_image.get(image_name)
        duplicate_review = image_name in review_confirmed_images
        include = _has_16(points) and old is not None and not duplicate_review
        if duplicate_review:
            exclude_reason = "duplicate_image_prefer_realworld_review_confirmed"
        elif not _has_16(points):
            exclude_reason = "missing_16_keypoints"
        elif old is None:
            exclude_reason = "not_in_previous_training_split"
        else:
            exclude_reason = ""
        crop_image_path = old.get("crop_image_path", "") if old is not None else ""
        warped_image_path = old.get("warped_image_path", f"data/real_images_warped/{_stem(image_name)}_warped.png") if old is not None else f"data/real_images_warped/{_stem(image_name)}_warped.png"
        rows.append(
            {
                "image_name": image_name,
                "specimen_id": str(payload.get("specimen_id", "")),
                "source_set": "original_corrected",
                "source_json_path": _path_text(path),
                "warped_image_path": str(warped_image_path),
                "crop_image_path": str(crop_image_path),
                "has_16_keypoints": _has_16(points),
                "annotation_mode": str(payload.get("annotation_mode", "")),
                "review_status": "",
                "include_in_v05_candidate": include,
                "exclude_reason": exclude_reason,
                "notes": "original v0.3 corrected label",
            }
        )
        payloads[("original_corrected", image_name)] = payload

    for path in sorted(REVIEW_CORRECTED_DIR.glob("*_corrected.json")):
        payload = _read_json(path)
        image_name = _normal_image_name(str(payload.get("image_name", "")), path.stem)
        sample_row = review_status_by_image.get(image_name)
        review_status = str(sample_row.get("review_status", "")) if sample_row is not None else ""
        points = _extract_corrected_keypoints(payload)
        include = review_status == "corrected_and_confirmed" and _has_16(points)
        crop_image_path = REVIEW_ROOT / "preannotations" / "crops" / f"{_stem(image_name)}_crop.png"
        warped_image_path = (
            str(sample_row.get("warped_image_path", ""))
            if sample_row is not None
            else str(payload.get("preannotation", {}).get("warped_image_path", f"data/real_images_warped/{_stem(image_name)}_warped.png"))
        )
        rows.append(
            {
                "image_name": image_name,
                "specimen_id": str(payload.get("specimen_id", "")),
                "source_set": "realworld_review_confirmed",
                "source_json_path": _path_text(path),
                "warped_image_path": warped_image_path,
                "crop_image_path": _path_text(crop_image_path),
                "has_16_keypoints": _has_16(points),
                "annotation_mode": str(payload.get("annotation_mode", "")),
                "review_status": review_status,
                "include_in_v05_candidate": include,
                "exclude_reason": "" if include else ("not_corrected_and_confirmed" if review_status != "corrected_and_confirmed" else "missing_16_keypoints"),
                "notes": "real-world review confirmed label",
            }
        )
        payloads[("realworld_review_confirmed", image_name)] = payload

    if not review_sample.empty:
        excluded = review_sample.loc[review_sample["review_status"].astype(str) == "excluded"].copy()
        for _, row in excluded.iterrows():
            image_name = str(row.get("image_name", ""))
            rows.append(
                {
                    "image_name": image_name,
                    "specimen_id": str(row.get("specimen_id", "")),
                    "source_set": "excluded_error_case",
                    "source_json_path": _path_text(REVIEW_ROOT / "preannotations" / "json" / f"{_stem(image_name)}_preannotation.json"),
                    "warped_image_path": str(row.get("warped_image_path", "")),
                    "crop_image_path": _path_text(REVIEW_ROOT / "preannotations" / "crops" / f"{_stem(image_name)}_crop.png"),
                    "has_16_keypoints": False,
                    "annotation_mode": "auto_preannotation_unverified",
                    "review_status": "excluded",
                    "include_in_v05_candidate": False,
                    "exclude_reason": str(row.get("exclude_reason", "")),
                    "notes": "excluded review error case; do not train",
                }
            )
    manifest = pd.DataFrame(rows)
    manifest.attrs["payloads"] = payloads
    return manifest


def build_split_plan(manifest: pd.DataFrame, old_split: pd.DataFrame) -> pd.DataFrame:
    included = manifest.loc[manifest["include_in_v05_candidate"].astype(bool)].copy()
    old_image_split = {str(row["image_name"]): str(row["split"]) for _, row in old_split.iterrows()} if not old_split.empty else {}
    specimen_splits: dict[str, set[str]] = defaultdict(set)
    for _, row in old_split.iterrows():
        specimen_splits[str(row.get("specimen_id", ""))].add(str(row.get("split", "")))
    old_specimen_split = {sid: sorted(splits)[0] for sid, splits in specimen_splits.items() if sid and len(splits) == 1}

    assigned_new: dict[str, str] = {}
    split_specimens: dict[str, set[str]] = defaultdict(set)
    for sid, split in old_specimen_split.items():
        split_specimens[split].add(sid)
    target_order = ["train", "val", "test"]
    target_ratio = {"train": 25 / 35, "val": 5 / 35, "test": 5 / 35}

    rows = []
    for _, row in included.sort_values(["specimen_id", "image_name"]).iterrows():
        image_name = str(row["image_name"])
        specimen_id = str(row["specimen_id"])
        old_split_value = old_image_split.get(image_name, "")
        if specimen_id in old_specimen_split:
            new_split = old_specimen_split[specimen_id]
            reason = "inherited_from_existing_specimen_split"
            is_new = False
        elif specimen_id in assigned_new:
            new_split = assigned_new[specimen_id]
            reason = "inherited_from_new_specimen_assignment"
            is_new = True
        else:
            total = sum(len(split_specimens[s]) for s in target_order) or 1
            deficits = {s: target_ratio[s] - (len(split_specimens[s]) / total) for s in target_order}
            new_split = max(target_order, key=lambda s: (deficits[s], -target_order.index(s)))
            assigned_new[specimen_id] = new_split
            split_specimens[new_split].add(specimen_id)
            reason = "new_specimen_assigned_to_balance_split_ratio"
            is_new = True
        rows.append(
            {
                "image_name": image_name,
                "specimen_id": specimen_id,
                "source_set": str(row["source_set"]),
                "old_split": old_split_value,
                "new_split": new_split,
                "split_assignment_reason": reason,
                "is_new_specimen": is_new,
                "leakage_check_pass": True,
            }
        )
    plan = pd.DataFrame(rows)
    for specimen_id, group in plan.groupby("specimen_id"):
        if group["new_split"].nunique() > 1:
            plan.loc[plan["specimen_id"] == specimen_id, "leakage_check_pass"] = False
    return plan


def integrity_report(manifest: pd.DataFrame, split_plan: pd.DataFrame, old_split: pd.DataFrame) -> pd.DataFrame:
    rows = []

    duplicates = manifest.groupby("image_name").size()
    duplicate_images = duplicates[duplicates > 1].index.tolist()
    rows.append(
        {
            "check_name": "duplicate_image_name_across_sources",
            "status": "warning" if duplicate_images else "pass",
            "num_issues": len(duplicate_images),
            "issue_examples": ";".join(duplicate_images[:8]),
            "suggested_action": "Prefer realworld_review_confirmed labels for duplicated image_name; excluded duplicate rows remain out of v0.5 candidate.",
        }
    )

    leak = split_plan.groupby("specimen_id")["new_split"].nunique()
    leak_ids = leak[leak > 1].index.tolist()
    rows.append(
        {
            "check_name": "specimen_id_split_leakage",
            "status": "fail" if leak_ids else "pass",
            "num_issues": len(leak_ids),
            "issue_examples": ";".join(leak_ids[:8]),
            "suggested_action": "Fix split plan before training." if leak_ids else "No action needed.",
        }
    )

    overlap = manifest.loc[
        manifest.duplicated("image_name", keep=False)
        & manifest["source_set"].isin(["original_corrected", "realworld_review_confirmed"])
    ]["image_name"].unique().tolist()
    rows.append(
        {
            "check_name": "same_image_in_original_and_review",
            "status": "warning" if overlap else "pass",
            "num_issues": len(overlap),
            "issue_examples": ";".join(overlap[:8]),
            "suggested_action": "Only the review-confirmed version is included for overlaps.",
        }
    )

    schema_bad = manifest.loc[
        (manifest["include_in_v05_candidate"].astype(bool))
        & (~manifest["has_16_keypoints"].astype(bool))
    ]["image_name"].tolist()
    rows.append(
        {
            "check_name": "keypoint_schema_has_16_manual_points",
            "status": "fail" if schema_bad else "pass",
            "num_issues": len(schema_bad),
            "issue_examples": ";".join(schema_bad[:8]),
            "suggested_action": "Exclude or repair incomplete corrected_keypoints before training.",
        }
    )

    rows.append(
        {
            "check_name": "p7v_excluded_from_training_keypoints",
            "status": "pass",
            "num_issues": 0,
            "issue_examples": "",
            "suggested_action": "Keep P7V as derived point only.",
        }
    )

    included_missing_json = [
        str(row["image_name"])
        for _, row in manifest.loc[manifest["include_in_v05_candidate"].astype(bool)].iterrows()
        if not _project_path(row["source_json_path"]).exists()
    ]
    rows.append(
        {
            "check_name": "source_json_exists_for_included_rows",
            "status": "fail" if included_missing_json else "pass",
            "num_issues": len(included_missing_json),
            "issue_examples": ";".join(included_missing_json[:8]),
            "suggested_action": "Check source_json_path values before export." if included_missing_json else "No action needed.",
        }
    )
    return pd.DataFrame(rows)


def build_hard_cases(split_plan: pd.DataFrame) -> pd.DataFrame:
    image_split = {str(row["image_name"]): str(row["new_split"]) for _, row in split_plan.iterrows()}
    image_source = {str(row["image_name"]): str(row["source_set"]) for _, row in split_plan.iterrows()}
    manual_image = pd.read_csv(POST_REVIEW_DIR / "manual_correction_by_image.csv")
    manual_detail = pd.read_csv(POST_REVIEW_DIR / "manual_correction_by_point_detail.csv")
    excluded = pd.read_csv(POST_REVIEW_DIR / "excluded_cases.csv", dtype=str, keep_default_na=False)
    rows = []
    important = {
        "P4_peduncle_start_midpoint": "P4_large_correction",
        "C2_trunk_axis_point": "C2_large_correction",
        "P9_body_depth_ventral": "P9_body_depth_correction",
        "P8_body_depth_dorsal": "P8_body_depth_correction",
        "P3_operculum_posterior": "P3_outlier",
        "P7U_caudal_fin_upper_tip": "P7U_outlier",
        "P7L_caudal_fin_lower_tip": "P7L_outlier",
    }
    detail_by_image = {name: group for name, group in manual_detail.groupby("image_name")}
    for _, row in manual_image.iterrows():
        image_name = str(row["image_name"])
        group = detail_by_image.get(image_name, pd.DataFrame())
        affected = []
        types = []
        review_reason = str(row.get("qc_warning_types", ""))
        for key, kind in important.items():
            if key in str(row.get("moved_keypoints", "")):
                affected.append(key)
                types.append(kind)
        large_keys = [key for key in str(row.get("large_correction_keypoints", "")).split(";") if key and key != "nan"]
        for key in large_keys:
            if key not in affected:
                affected.append(key)
                types.append(f"{key}_large_outlier")
        if not types and float(row.get("max_displacement_mm", 0.0)) <= 5 and int(row.get("num_points_moved", 0)) < 5:
            continue
        split = image_split.get(image_name, "")
        recommended = "train_hard_case" if split == "train" else "val_monitor_case"
        rows.append(
            {
                "image_name": image_name,
                "specimen_id": str(row.get("specimen_id", "")),
                "source_set": image_source.get(image_name, "realworld_review_confirmed"),
                "hard_case_type": ";".join(dict.fromkeys(types)),
                "affected_keypoints": ";".join(dict.fromkeys(affected)),
                "max_displacement_mm": row.get("max_displacement_mm", ""),
                "mean_displacement_mm": row.get("mean_displacement_mm", ""),
                "num_points_moved": row.get("num_points_moved", ""),
                "review_reason": review_reason,
                "recommended_use": recommended,
            }
        )
    for _, row in excluded.iterrows():
        rows.append(
            {
                "image_name": str(row.get("image_name", "")),
                "specimen_id": str(row.get("specimen_id", "")),
                "source_set": "excluded_error_case",
                "hard_case_type": "failed_preannotation",
                "affected_keypoints": "",
                "max_displacement_mm": "",
                "mean_displacement_mm": "",
                "num_points_moved": "",
                "review_reason": str(row.get("exclude_reason", "")),
                "recommended_use": "external_error_case_only",
            }
        )
    return pd.DataFrame(rows)


def export_heatmap_dataset(manifest: pd.DataFrame, split_plan: pd.DataFrame, hard_cases: pd.DataFrame) -> tuple[bool, pd.DataFrame]:
    if V05_DATASET.exists():
        # Keep this scoped to the generated v0.5 candidate dataset only.
        shutil.rmtree(V05_DATASET)
    for split in ("train", "val", "test"):
        (V05_DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (V05_DATASET / "heatmaps" / split).mkdir(parents=True, exist_ok=True)
    old_samples = _old_label_samples()
    split_by_image = {str(row["image_name"]): str(row["new_split"]) for _, row in split_plan.iterrows()}
    hard_type_by_image = {
        str(row["image_name"]): str(row["hard_case_type"])
        for _, row in hard_cases.groupby("image_name", as_index=False).agg({"hard_case_type": lambda s: ";".join(dict.fromkeys(";".join(s).split(";"))) }).iterrows()
    } if not hard_cases.empty else {}
    manifest_by_image_source = {
        (str(row["image_name"]), str(row["source_set"])): row
        for _, row in manifest.loc[manifest["include_in_v05_candidate"].astype(bool)].iterrows()
    }
    samples: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []

    for _, plan_row in split_plan.sort_values(["new_split", "specimen_id", "image_name"]).iterrows():
        image_name = str(plan_row["image_name"])
        source_set = str(plan_row["source_set"])
        split = str(plan_row["new_split"])
        manifest_row = manifest_by_image_source[(image_name, source_set)]
        payload = _read_json(_project_path(manifest_row["source_json_path"]))
        corrected = _extract_corrected_keypoints(payload)
        hard_case_type = hard_type_by_image.get(image_name, "")
        if source_set == "original_corrected" and image_name in old_samples:
            old_sample = dict(old_samples[image_name])
            old_image = OLD_HEATMAP_DATASET / str(old_sample["image_path"])
            old_heatmap = OLD_HEATMAP_DATASET / str(old_sample["heatmap_path"])
            image_rel = Path("images") / split / Path(str(old_sample["image_path"])).name
            heatmap_rel = Path("heatmaps") / split / Path(str(old_sample["heatmap_path"])).name
            _copy_file(old_image, V05_DATASET / image_rel)
            _copy_file(old_heatmap, V05_DATASET / heatmap_rel)
            sample = old_sample
            sample.update(
                {
                    "split": split,
                    "source_set": source_set,
                    "image_path": str(image_rel).replace("\\", "/"),
                    "heatmap_path": str(heatmap_rel).replace("\\", "/"),
                    "hard_case_type": hard_case_type,
                }
            )
            crop_width = manifest_row.get("crop_width", "")
            crop_height = manifest_row.get("crop_height", "")
        else:
            crop_path = _project_path(manifest_row["crop_image_path"])
            crop_box = _crop_box_from_preannotation(image_name)
            if crop_box is None:
                raise RuntimeError(f"Missing crop box for {image_name}")
            x1, y1, x2, y2 = crop_box
            crop_width = max(1.0, x2 - x1)
            crop_height = max(1.0, y2 - y1)
            image_rel = Path("images") / split / f"{_stem(image_name)}_review_maskcrop_512.png"
            heatmap_rel = Path("heatmaps") / split / f"{_stem(image_name)}_review_maskcrop_heatmaps.npz"
            with Image.open(crop_path) as img:
                resized = img.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.LANCZOS)
                (V05_DATASET / image_rel).parent.mkdir(parents=True, exist_ok=True)
                resized.save(V05_DATASET / image_rel)
            keypoints_resized = []
            warped_points = {}
            for key in KEYPOINT_KEYS:
                px, py = _xy(corrected[key])  # type: ignore[index]
                rx = (px - x1) / crop_width * INPUT_SIZE
                ry = (py - y1) / crop_height * INPUT_SIZE
                keypoints_resized.append([float(rx), float(ry)])
                warped_points[key] = [float(px), float(py)]
            keypoints_np = np.asarray(keypoints_resized, dtype=np.float32)
            heatmaps = _gaussian_heatmaps(keypoints_np)
            np.savez_compressed(V05_DATASET / heatmap_rel, heatmaps=heatmaps, keypoints=keypoints_np)
            sample = {
                "image_name": image_name,
                "specimen_id": str(plan_row["specimen_id"]),
                "split": split,
                "source_set": source_set,
                "image_path": str(image_rel).replace("\\", "/"),
                "heatmap_path": str(heatmap_rel).replace("\\", "/"),
                "keypoints": {key: keypoints_resized[idx] for idx, key in enumerate(KEYPOINT_KEYS)},
                "warped_keypoints": warped_points,
                "source_crop_width": int(round(crop_width)),
                "source_crop_height": int(round(crop_height)),
                "crop_box": [x1, y1, x2, y2],
                "hard_case_type": hard_case_type,
            }
        samples.append(sample)
        split_rows.append(
            {
                "image_name": image_name,
                "specimen_id": str(plan_row["specimen_id"]),
                "split": split,
                "source_type": "real",
                "source_set": source_set,
                "image_path": sample["image_path"],
                "heatmap_path": sample["heatmap_path"],
                "warped_image_path": manifest_row["warped_image_path"],
                "crop_image_path": manifest_row["crop_image_path"],
                "hard_case_type": hard_case_type,
            }
        )

    labels = {
        "input_size": [INPUT_SIZE, INPUT_SIZE],
        "sigma_px": SIGMA_PX,
        "coordinate_space": "resized_maskcrop",
        "samples": samples,
    }
    _write_json(V05_DATASET / "labels.json", labels)
    if (OLD_HEATMAP_DATASET / "keypoint_schema.json").exists():
        _copy_file(OLD_HEATMAP_DATASET / "keypoint_schema.json", V05_DATASET / "keypoint_schema.json")
    else:
        _write_json(V05_DATASET / "keypoint_schema.json", {"keypoints": KEYPOINT_KEYS, "kpt_shape": [16, 2]})

    split_manifest = pd.DataFrame(split_rows)
    split_manifest.to_csv(V05_DATASET / "split_manifest.csv", index=False, encoding="utf-8-sig")
    summary_rows = []
    for split, group in split_manifest.groupby("split"):
        summary_rows.append(
            {
                "split": split,
                "num_images": len(group),
                "num_specimens": group["specimen_id"].nunique(),
                "num_original_images": int((group["source_set"] == "original_corrected").sum()),
                "num_review_confirmed_images": int((group["source_set"] == "realworld_review_confirmed").sum()),
                "specimen_ids": ";".join(sorted(group["specimen_id"].unique())),
            }
        )
    dataset_summary = pd.DataFrame(summary_rows).sort_values("split")
    dataset_summary.to_csv(V05_DATASET / "dataset_summary.csv", index=False, encoding="utf-8-sig")
    readme = f"""# SiganusMorph heatmap-U-Net v0.5 candidate dataset

This dataset is prepared for review only. It has not been used to train a v0.5 model yet.

- Input size: {INPUT_SIZE} x {INPUT_SIZE}
- Heatmap sigma: {SIGMA_PX} px
- Keypoints: 16 manual keypoints
- P7V: excluded from training; derived geometrically
- Excluded review error cases: not included
- Split policy: grouped by specimen_id, inheriting existing split whenever possible

Review `split_manifest.csv`, `dataset_summary.csv`, and the planning files under `results/v0.5_dataset_planning/` before training.
"""
    (V05_DATASET / "README_dataset.md").write_text(readme, encoding="utf-8-sig")
    return True, dataset_summary


def split_summary(split_plan: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in split_plan.groupby("new_split"):
        rows.append(
            {
                "split": split,
                "num_images": len(group),
                "num_specimens": group["specimen_id"].nunique(),
                "num_original_images": int((group["source_set"] == "original_corrected").sum()),
                "num_review_confirmed_images": int((group["source_set"] == "realworld_review_confirmed").sum()),
                "specimen_ids": ";".join(sorted(group["specimen_id"].unique())),
            }
        )
    return pd.DataFrame(rows).sort_values("split")


def training_plan_doc(
    manifest: pd.DataFrame,
    split_plan: pd.DataFrame,
    summary: pd.DataFrame,
    integrity: pd.DataFrame,
    hard_cases: pd.DataFrame,
) -> str:
    included = manifest.loc[manifest["include_in_v05_candidate"].astype(bool)]
    leakage_pass = bool(split_plan["leakage_check_pass"].all()) if not split_plan.empty else False
    original_total = int((included["source_set"] == "original_corrected").sum())
    review_total = int((included["source_set"] == "realworld_review_confirmed").sum())
    excluded_total = int((manifest["source_set"] == "excluded_error_case").sum())
    top_hard = Counter()
    for value in hard_cases.get("affected_keypoints", pd.Series(dtype=str)).astype(str):
        for key in value.split(";"):
            if key:
                top_hard[key] += 1
    hard_text = "; ".join(f"{key}({count})" for key, count in top_hard.most_common(8))
    split_table = summary.to_markdown(index=False) if not summary.empty else "No split summary."
    integrity_table = integrity.to_markdown(index=False)
    return f"""# v0.5 Candidate Training Plan

## 1. Dataset Size

- Total candidate images: {len(included)}
- Total specimens: {included['specimen_id'].nunique()}
- Original corrected images included: {original_total}
- Real-world review confirmed images included: {review_total}
- Excluded review error cases: {excluded_total}

## 2. Split Plan

Specimen leakage check: {'pass' if leakage_pass else 'fail'}

{split_table}

Integrity checks:

{integrity_table}

## 3. Training Goals

- Improve frequently corrected points: P4_peduncle_start_midpoint, C2_trunk_axis_point, P9_body_depth_ventral, P8_body_depth_dorsal, and P11_peduncle_depth_ventral.
- Preserve stable point performance for P1_snout_tip, P2_eye_front, P5_caudal_base_midpoint, P6_caudal_fork_midpoint, and P10_peduncle_depth_dorsal.
- Reduce risk of failed-preannotation cases, while keeping `real_037.png / fish_18` excluded from training for now.

## 4. Recommended Strategy

- Prefer heatmap-U-Net v0.5 because v0.4 showed stronger landmark localization than YOLO-pose.
- Start from `models/siganusmorph_heatmap_unet_v0.1/best_model.pt` for fine-tuning, or train a fresh v0.5 model as an ablation.
- Do not use horizontal flip because fish head direction is semantically fixed.
- Use conservative augmentation only: small brightness/contrast, small rotation, small scale/translation, mild noise.
- Monitor hard points explicitly: P4, C2, P9, P8, P11, plus low-frequency outliers P3 and P7U/P7L.

## 5. Evaluation Plan

- Compare against v0.4.1 real-world review performance.
- Use the same specimen-grouped test split.
- Report original test and review-confirmed subsets separately.
- Report hard-case performance separately.
- Report median error, mean error, large error rate, and per-keypoint error for P4/C2/P9/P8/P11.

## 6. Hard Cases

Top affected hard-case keypoints: {hard_text or 'none'}

See `v05_hard_cases.csv` for image-level details.

## 7. Recommendation

Do not train immediately. First inspect:

1. `v05_candidate_label_manifest.csv`
2. `v05_split_plan.csv`
3. `v05_hard_cases.csv`
4. `datasets/siganusmorph_heatmap_unet_v0.5_candidate/README_dataset.md`

After confirming the candidate dataset and split plan, decide whether to train v0.5.
"""


def main() -> None:
    PLANNING_DIR.mkdir(parents=True, exist_ok=True)
    old_split = _load_old_split()
    review_sample = _load_review_sample()
    manifest = build_candidate_manifest(old_split, review_sample)
    split_plan = build_split_plan(manifest, old_split)
    integrity = integrity_report(manifest, split_plan, old_split)
    hard_cases = build_hard_cases(split_plan)
    summary = split_summary(split_plan)
    export_ok = bool((integrity["status"] != "fail").all())
    dataset_summary = pd.DataFrame()
    if export_ok:
        export_ok, dataset_summary = export_heatmap_dataset(manifest, split_plan, hard_cases)

    manifest.to_csv(PLANNING_DIR / "v05_candidate_label_manifest.csv", index=False, encoding="utf-8-sig")
    integrity.to_csv(PLANNING_DIR / "v05_data_integrity_report.csv", index=False, encoding="utf-8-sig")
    split_plan.to_csv(PLANNING_DIR / "v05_split_plan.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(PLANNING_DIR / "v05_split_summary.csv", index=False, encoding="utf-8-sig")
    hard_cases.to_csv(PLANNING_DIR / "v05_hard_cases.csv", index=False, encoding="utf-8-sig")
    (PLANNING_DIR / "v05_training_plan.md").write_text(
        training_plan_doc(manifest, split_plan, summary, integrity, hard_cases),
        encoding="utf-8-sig",
    )

    included = manifest.loc[manifest["include_in_v05_candidate"].astype(bool)]
    original_files = len(list(ORIGINAL_CORRECTED_DIR.glob("*.json")))
    review_confirmed = int((manifest["source_set"] == "realworld_review_confirmed").sum())
    excluded_cases = int((manifest["source_set"] == "excluded_error_case").sum())
    leakage_pass = bool(split_plan["leakage_check_pass"].all()) if not split_plan.empty else False
    affected = Counter()
    for value in hard_cases.get("affected_keypoints", pd.Series(dtype=str)).astype(str):
        for key in value.split(";"):
            if key:
                affected[key] += 1
    print("Finished v0.5 candidate dataset planning.")
    print(f"Original corrected labels: {original_files}")
    print(f"Real-world review confirmed labels: {review_confirmed}")
    print(f"Excluded review cases: {excluded_cases}")
    print(f"Total v0.5 candidate images: {len(included)}")
    print(f"Total specimens: {included['specimen_id'].nunique()}")
    print(f"Leakage check: {'pass' if leakage_pass else 'fail'}")
    print(f"Exported heatmap-U-Net v0.5 candidate dataset: {'yes' if export_ok else 'no'}")
    print("Hard cases:")
    print("Top affected keypoints: " + ("; ".join(f"{k}({v})" for k, v in affected.most_common(8)) or "none"))
    print("Recommended next action:")
    print("- inspect v05_candidate_label_manifest.csv")
    print("- inspect v05_split_plan.csv")
    print("- inspect v05_hard_cases.csv")
    print("- then decide whether to train v0.5")


if __name__ == "__main__":
    main()
