"""Sanity check the v0.5 heatmap-U-Net candidate dataset before training."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph.config import KEYPOINT_DEFS  # noqa: E402


DATASET_ROOT = PROJECT_ROOT / "datasets" / "siganusmorph_heatmap_unet_v0.5_candidate"
OUTPUT_DIR = PROJECT_ROOT / "results" / "v0.5_dataset_planning" / "sanity_check"
KEYPOINT_KEYS = [f"{definition.code}_{definition.name}" for definition in KEYPOINT_DEFS]
EXPECTED_SPLIT_COUNTS = {"train": 69, "val": 12, "test": 15}
RANDOM_SEED = 20260515


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _font(size: int = 16) -> ImageFont.ImageFont:
    for candidate in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _argmax_points(heatmaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    channels, height, width = heatmaps.shape
    flat = heatmaps.reshape(channels, -1)
    idx = np.argmax(flat, axis=1)
    peaks = np.max(flat, axis=1)
    ys = idx // width
    xs = idx % width
    return np.stack([xs, ys], axis=1).astype(np.float32), peaks.astype(np.float32)


def _draw_sample(sample: Mapping[str, Any], peak_points: np.ndarray, output_path: Path) -> None:
    image = Image.open(DATASET_ROOT / str(sample["image_path"])).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _font(15)
    keypoints = np.asarray([sample["keypoints"][key] for key in KEYPOINT_KEYS], dtype=np.float32)
    for idx, key in enumerate(KEYPOINT_KEYS):
        x, y = keypoints[idx]
        px, py = peak_points[idx]
        radius = 4
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(60, 220, 90), outline=(0, 80, 0), width=1)
        draw.ellipse((px - radius - 1, py - radius - 1, px + radius + 1, py + radius + 1), outline=(255, 120, 0), width=2)
        draw.line((x, y, px, py), fill=(255, 255, 255), width=1)
        if idx in (0, 1, 2, 3, 8, 9, 13):
            draw.text((x + 5, y + 5), key.split("_")[0], fill=(255, 255, 255), font=font)
    label = f"{sample['split']} {sample['image_name']} {sample['specimen_id']}"
    draw.rectangle((5, 5, min(505, 20 + len(label) * 8), 32), fill=(0, 0, 0))
    draw.text((10, 9), label, fill=(255, 255, 255), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, optimize=True)


def _report_row(check_name: str, status: str, num_issues: int, examples: list[str], action: str) -> dict[str, Any]:
    return {
        "check_name": check_name,
        "status": status,
        "num_issues": int(num_issues),
        "issue_examples": ";".join(examples[:8]),
        "suggested_action": action,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    visual_dir = OUTPUT_DIR / "visual_label_heatmap_checks"
    visual_dir.mkdir(parents=True, exist_ok=True)

    split_manifest = pd.read_csv(DATASET_ROOT / "split_manifest.csv", dtype=str, keep_default_na=False)
    labels = _read_json(DATASET_ROOT / "labels.json")
    schema = _read_json(DATASET_ROOT / "keypoint_schema.json")
    samples = labels.get("samples", [])
    sample_by_name = {str(sample.get("image_name", "")): sample for sample in samples if isinstance(sample, Mapping)}
    rows: list[dict[str, Any]] = []

    # Split leakage and counts.
    leakage = split_manifest.groupby("specimen_id")["split"].nunique()
    leak_ids = leakage[leakage > 1].index.astype(str).tolist()
    rows.append(_report_row("specimen_id_split_leakage", "fail" if leak_ids else "pass", len(leak_ids), leak_ids, "Fix split plan before training." if leak_ids else "No action needed."))
    split_counts = split_manifest.groupby("split").size().to_dict()
    bad_counts = [f"{split}: expected {expected}, found {split_counts.get(split, 0)}" for split, expected in EXPECTED_SPLIT_COUNTS.items() if int(split_counts.get(split, 0)) != expected]
    rows.append(_report_row("expected_split_counts_69_12_15", "fail" if bad_counts else "pass", len(bad_counts), bad_counts, "Regenerate split_manifest if counts are unexpected."))

    # Schema checks.
    schema_names = [str(item.get("name", "")) for item in schema.get("keypoints", []) if isinstance(item, Mapping)]
    schema_text = json.dumps(schema, ensure_ascii=False)
    p7v_present = any("P7V" in name or "caudal_fin_posterior_endpoint" in name for name in schema_names)
    rows.append(_report_row("p7v_not_in_keypoint_schema", "fail" if p7v_present else "pass", int(p7v_present), ["P7V present"] if p7v_present else [], "Remove P7V from training schema." if p7v_present else "No action needed."))
    schema_contains_all = all(key in schema_text for key in KEYPOINT_KEYS)
    rows.append(_report_row("schema_contains_16_manual_keypoints", "pass" if schema_contains_all else "fail", 0 if schema_contains_all else 1, [] if schema_contains_all else ["schema missing at least one key"], "Check keypoint_schema.json."))

    # Per-sample checks.
    alignment_rows: list[dict[str, Any]] = []
    severe_examples: list[str] = []
    missing_examples: list[str] = []
    channel_examples: list[str] = []
    p7v_label_examples: list[str] = []
    hard_case_missing = []
    for sample in samples:
        image_name = str(sample.get("image_name", ""))
        keypoints = sample.get("keypoints", {})
        if not isinstance(keypoints, Mapping) or not all(key in keypoints for key in KEYPOINT_KEYS):
            missing_examples.append(image_name)
            continue
        if any("P7V" in key or "caudal_fin_posterior_endpoint" in key for key in keypoints):
            p7v_label_examples.append(image_name)
        heatmap_path = DATASET_ROOT / str(sample.get("heatmap_path", ""))
        image_path = DATASET_ROOT / str(sample.get("image_path", ""))
        if not heatmap_path.exists() or not image_path.exists():
            missing_examples.append(image_name)
            continue
        loaded = np.load(heatmap_path)
        heatmaps = loaded["heatmaps"].astype(np.float32)
        stored_keypoints = loaded["keypoints"].astype(np.float32) if "keypoints" in loaded.files else None
        if heatmaps.shape != (len(KEYPOINT_KEYS), 512, 512):
            channel_examples.append(f"{image_name}: {heatmaps.shape}")
            continue
        label_points = np.asarray([keypoints[key] for key in KEYPOINT_KEYS], dtype=np.float32)
        peak_points, peak_values = _argmax_points(heatmaps)
        label_peak_dist = np.sqrt(np.sum((label_points - peak_points) ** 2, axis=1))
        stored_label_dist = np.sqrt(np.sum((label_points - stored_keypoints) ** 2, axis=1)) if stored_keypoints is not None else np.full(len(KEYPOINT_KEYS), np.nan)
        if float(np.max(label_peak_dist)) > 2.0 or (np.isfinite(stored_label_dist).all() and float(np.max(stored_label_dist)) > 0.05):
            severe_examples.append(f"{image_name}: max_peak_dist={float(np.max(label_peak_dist)):.2f}")
        if not str(sample.get("hard_case_type", "")) and image_name in set(split_manifest.loc[split_manifest.get("hard_case_type", "").astype(str) != "", "image_name"]):
            hard_case_missing.append(image_name)
        for idx, key in enumerate(KEYPOINT_KEYS):
            alignment_rows.append(
                {
                    "image_name": image_name,
                    "split": str(sample.get("split", "")),
                    "specimen_id": str(sample.get("specimen_id", "")),
                    "keypoint_name": key,
                    "label_x": float(label_points[idx, 0]),
                    "label_y": float(label_points[idx, 1]),
                    "heatmap_peak_x": float(peak_points[idx, 0]),
                    "heatmap_peak_y": float(peak_points[idx, 1]),
                    "peak_value": float(peak_values[idx]),
                    "label_to_peak_px": float(label_peak_dist[idx]),
                    "stored_label_to_json_label_px": float(stored_label_dist[idx]) if np.isfinite(stored_label_dist[idx]) else np.nan,
                }
            )

    rows.append(_report_row("all_samples_have_16_manual_keypoints", "fail" if missing_examples else "pass", len(missing_examples), missing_examples, "Repair or exclude incomplete samples." if missing_examples else "No action needed."))
    rows.append(_report_row("p7v_not_in_training_labels", "fail" if p7v_label_examples else "pass", len(p7v_label_examples), p7v_label_examples, "Remove P7V from labels.json." if p7v_label_examples else "No action needed."))
    rows.append(_report_row("heatmap_channel_shape_16x512x512", "fail" if channel_examples else "pass", len(channel_examples), channel_examples, "Regenerate heatmaps with 16 channels." if channel_examples else "No action needed."))
    rows.append(_report_row("label_heatmap_peak_alignment", "fail" if severe_examples else "pass", len(severe_examples), severe_examples, "Regenerate heatmaps; peaks should be within 2 px of labels." if severe_examples else "No action needed."))
    real037_in = split_manifest["image_name"].astype(str).eq("real_037.png").any() or any(str(sample.get("image_name", "")) == "real_037.png" for sample in samples)
    rows.append(_report_row("excluded_real_037_not_in_training", "fail" if real037_in else "pass", int(real037_in), ["real_037.png"] if real037_in else [], "Remove excluded case from split/labels." if real037_in else "No action needed."))
    dup_conflict = split_manifest["image_name"].duplicated().sum()
    rows.append(_report_row("no_duplicate_image_conflict", "fail" if dup_conflict else "pass", int(dup_conflict), split_manifest.loc[split_manifest["image_name"].duplicated(), "image_name"].astype(str).tolist(), "Remove duplicate image rows." if dup_conflict else "No action needed."))
    rows.append(_report_row("hard_cases_marked_in_split_manifest", "warning" if hard_case_missing else "pass", len(hard_case_missing), hard_case_missing, "Check hard_case_type propagation." if hard_case_missing else "No action needed."))

    # Visual samples: train 6, val 3, test 3.
    rng = random.Random(RANDOM_SEED)
    visual_counts = {"train": 6, "val": 3, "test": 3}
    for split, count in visual_counts.items():
        split_samples = [sample for sample in samples if str(sample.get("split", "")) == split]
        for sample in rng.sample(split_samples, min(count, len(split_samples))):
            heatmaps = np.load(DATASET_ROOT / str(sample["heatmap_path"]))["heatmaps"].astype(np.float32)
            peaks, _ = _argmax_points(heatmaps)
            _draw_sample(sample, peaks, visual_dir / f"{split}_{sample['image_name'].replace('.png', '')}_labels_heatmaps.png")

    report = pd.DataFrame(rows)
    align = pd.DataFrame(alignment_rows)
    serious_fail = bool((report["status"] == "fail").any())
    summary = {
        "dataset_root": str(DATASET_ROOT.relative_to(PROJECT_ROOT)),
        "total_images": int(len(samples)),
        "split_counts": {split: int(count) for split, count in split_counts.items()},
        "num_specimens": int(split_manifest["specimen_id"].nunique()),
        "specimen_leakage": bool(leak_ids),
        "serious_fail": serious_fail,
        "num_failed_checks": int((report["status"] == "fail").sum()),
        "num_warning_checks": int((report["status"] == "warning").sum()),
        "max_label_to_peak_px": float(align["label_to_peak_px"].max()) if not align.empty else None,
        "mean_label_to_peak_px": float(align["label_to_peak_px"].mean()) if not align.empty else None,
        "visual_check_count": len(list(visual_dir.glob("*.png"))),
    }
    report.to_csv(OUTPUT_DIR / "dataset_sanity_report.csv", index=False, encoding="utf-8-sig")
    align.to_csv(OUTPUT_DIR / "label_heatmap_alignment_report.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "dataset_sanity_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Finished v0.5 candidate dataset sanity check.")
    print(f"Serious fail: {serious_fail}")
    print(f"Split counts: {summary['split_counts']}")
    print(f"Max label-to-heatmap peak distance: {summary['max_label_to_peak_px']:.3f} px")
    print(f"Visual checks: {summary['visual_check_count']}")
    print(f"Results saved to: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/")
    if serious_fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
