"""Evaluate v0.6.8 peduncle-depth geometry suggestions.

This script is read-only with respect to labels and historical batch results.
It compares current/preannotation P10/P11 with geometric P10/P11 where ground
truth corrected labels are available, and records 5.24 preannotation QC.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph.image_utils import ensure_rgb, load_image_file  # noqa: E402
from siganusmorph.local_normal_measurement import estimate_peduncle_depth_by_axis_normals  # noqa: E402
from siganusmorph.realworld_review import xy_from_payload  # noqa: E402
from siganusmorph.segmentation import segment_fish_from_blue_board  # noqa: E402


OUT_DIR = PROJECT_ROOT / "results" / "model_eval" / "v0.6.8_review_overlay_and_peduncle_depth_geometry"
VIS_DIR = OUT_DIR / "visual_comparisons"
MM_PER_PIXEL = 0.1


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve(path_text: str) -> Path:
    path = Path(str(path_text))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _corrected_points(payload: Mapping[str, Any]) -> dict[str, Any]:
    points = payload.get("corrected_keypoints", {}) if isinstance(payload.get("corrected_keypoints", {}), Mapping) else {}
    return dict(points)


def _pre_points(payload: Mapping[str, Any]) -> dict[str, Any]:
    for container in (
        payload.get("preannotation", {}) if isinstance(payload.get("preannotation", {}), Mapping) else {},
        payload,
        payload.get("preannotation_metadata", {}) if isinstance(payload.get("preannotation_metadata", {}), Mapping) else {},
    ):
        if not isinstance(container, Mapping):
            continue
        for key in ("v06_keypoints", "hybrid_keypoints", "corrected_keypoints"):
            value = container.get(key)
            if isinstance(value, Mapping) and value:
                return dict(value)
    return {}


def _dist_mm(a: Any, b: Any) -> float | None:
    pa = xy_from_payload(a)
    pb = xy_from_payload(b)
    if pa is None or pb is None:
        return None
    return math.hypot(pa[0] - pb[0], pa[1] - pb[1]) * MM_PER_PIXEL


def _depth_mm(points: Mapping[str, Any], p10_key: str = "P10_peduncle_depth_dorsal", p11_key: str = "P11_peduncle_depth_ventral") -> float | None:
    return _dist_mm(points.get(p10_key), points.get(p11_key))


def _draw_visual(
    image: Any,
    out_path: Path,
    *,
    title: str,
    gt: Mapping[str, Any] | None,
    current: Mapping[str, Any] | None,
    geometry: Mapping[str, Any],
) -> None:
    canvas = Image.fromarray(ensure_rgb(image)).convert("RGB")
    max_side = 1400
    scale = 1.0
    if max(canvas.size) > max_side:
        scale = max_side / float(max(canvas.size))
        resized = (max(1, round(canvas.width * scale)), max(1, round(canvas.height * scale)))
        canvas = canvas.resize(resized, Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(canvas)
    radius = max(5, round(min(canvas.size) / 260))
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 18)
    except OSError:
        font = ImageFont.load_default()

    def point(payload: Any, label: str, fill: tuple[int, int, int]) -> tuple[float, float] | None:
        xy = xy_from_payload(payload)
        if xy is None:
            return None
        x, y = xy[0] * scale, xy[1] * scale
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=(255, 255, 255), width=2)
        draw.text((x + radius + 3, y - radius), label, font=font, fill=fill)
        return xy

    gt10 = point((gt or {}).get("P10_peduncle_depth_dorsal"), "P10_gt", (60, 230, 90))
    gt11 = point((gt or {}).get("P11_peduncle_depth_ventral"), "P11_gt", (60, 230, 90))
    cur10 = point((current or {}).get("P10_peduncle_depth_dorsal"), "P10_cur", (240, 55, 65))
    cur11 = point((current or {}).get("P11_peduncle_depth_ventral"), "P11_cur", (240, 55, 65))
    geo10 = point(geometry.get("P10_geometric"), "P10_geo", (0, 220, 180))
    geo11 = point(geometry.get("P11_geometric"), "P11_geo", (0, 220, 180))
    if gt10 and gt11:
        draw.line((gt10[0] * scale, gt10[1] * scale, gt11[0] * scale, gt11[1] * scale), fill=(60, 230, 90), width=3)
    if cur10 and cur11:
        draw.line((cur10[0] * scale, cur10[1] * scale, cur11[0] * scale, cur11[1] * scale), fill=(240, 55, 65), width=3)
    if geo10 and geo11:
        draw.line((geo10[0] * scale, geo10[1] * scale, geo11[0] * scale, geo11[1] * scale), fill=(0, 245, 190), width=5)
    draw.rectangle((8, 8, min(canvas.width - 8, 24 + len(title) * 10), 42), fill=(0, 0, 0))
    draw.text((16, 12), title, font=font, fill=(255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, optimize=True)


def _records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    batch_manifest = PROJECT_ROOT / "results" / "batch_measurement_v0.6.4_stable" / "batch_measurement_manifest.csv"
    if batch_manifest.exists():
        df = pd.read_csv(batch_manifest, dtype=str, keep_default_na=False)
        df = df[df["include_in_batch_measurement"].astype(str).str.lower().isin(["true", "1"])]
        for _, row in df.iterrows():
            records.append(
                {
                    "subset": "final_analysis_96",
                    "image_name": row.get("image_name", ""),
                    "specimen_id": row.get("specimen_id", ""),
                    "corrected_json_path": row.get("corrected_json_path", ""),
                    "warped_image_path": row.get("warped_image_path", ""),
                }
            )
    review_manifest = PROJECT_ROOT / "results" / "realworld_review_5_24_v0.6.5" / "review_sample_manifest.csv"
    if review_manifest.exists():
        df = pd.read_csv(review_manifest, dtype=str, keep_default_na=False)
        df = df[df["preannotation_success"].astype(str).str.lower().isin(["true", "1"])]
        for _, row in df.iterrows():
            records.append(
                {
                    "subset": "5_24_preannotation",
                    "image_name": row.get("image_name", ""),
                    "specimen_id": row.get("specimen_id", ""),
                    "corrected_json_path": "",
                    "preannotation_json_path": row.get("preannotation_json_path", ""),
                    "warped_image_path": row.get("warped_image_path", ""),
                }
            )
    return records


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for record in _records():
        image_path = _resolve(str(record.get("warped_image_path", "")))
        if not image_path.exists():
            continue
        image = load_image_file(image_path)
        payload = _read_json(_resolve(str(record.get("corrected_json_path", "")))) if record.get("corrected_json_path") else {}
        pre_payload = _read_json(_resolve(str(record.get("preannotation_json_path", "")))) if record.get("preannotation_json_path") else {}
        gt = _corrected_points(payload)
        current = _pre_points(payload) or _pre_points(pre_payload) or gt
        try:
            fish_mask, _, _ = segment_fish_from_blue_board(image)
            geometry = estimate_peduncle_depth_by_axis_normals(fish_mask, current or gt, image.shape, config={"mm_per_pixel": MM_PER_PIXEL})
        except Exception as exc:  # noqa: BLE001
            geometry = {
                "P10_geometric": None,
                "P11_geometric": None,
                "peduncle_depth_geometric_mm": None,
                "peduncle_depth_geometric_qc_pass": False,
                "peduncle_depth_geometric_review_reason": f"failed:{exc}",
            }
        current_p10_error = _dist_mm(current.get("P10_peduncle_depth_dorsal"), gt.get("P10_peduncle_depth_dorsal")) if gt else None
        current_p11_error = _dist_mm(current.get("P11_peduncle_depth_ventral"), gt.get("P11_peduncle_depth_ventral")) if gt else None
        geo_p10_error = _dist_mm(geometry.get("P10_geometric"), gt.get("P10_peduncle_depth_dorsal")) if gt else None
        geo_p11_error = _dist_mm(geometry.get("P11_geometric"), gt.get("P11_peduncle_depth_ventral")) if gt else None
        gt_depth = _depth_mm(gt) if gt else None
        current_depth = _depth_mm(current) if current else None
        geo_depth = geometry.get("peduncle_depth_geometric_mm")
        try:
            geo_depth = float(geo_depth) if geo_depth not in ("", None) else None
        except (TypeError, ValueError):
            geo_depth = None
        current_depth_error = abs(current_depth - gt_depth) if current_depth is not None and gt_depth is not None else None
        geo_depth_error = abs(geo_depth - gt_depth) if geo_depth is not None and gt_depth is not None else None
        rows.append(
            {
                **record,
                "current_P10_error_mm": current_p10_error,
                "geometric_P10_error_mm": geo_p10_error,
                "current_P11_error_mm": current_p11_error,
                "geometric_P11_error_mm": geo_p11_error,
                "current_peduncle_depth_error_mm": current_depth_error,
                "geometric_peduncle_depth_error_mm": geo_depth_error,
                "peduncle_depth_geometric_mm": geo_depth,
                "peduncle_depth_geometric_qc_pass": geometry.get("peduncle_depth_geometric_qc_pass", ""),
                "peduncle_depth_geometric_review_reason": geometry.get("peduncle_depth_geometric_review_reason", ""),
                "has_ground_truth": bool(gt),
            }
        )
        if len(rows) <= 140:
            _draw_visual(
                image,
                VIS_DIR / f"{Path(str(record.get('image_name','image'))).stem}_peduncle_geometry.png",
                title=f"{record.get('image_name')} {record.get('subset')}",
                gt=gt,
                current=current,
                geometry=geometry,
            )
    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "peduncle_depth_error_by_image.csv", index=False, encoding="utf-8-sig")
    gt_df = result[result["has_ground_truth"].astype(bool)].copy() if not result.empty else pd.DataFrame()
    summary_rows = []
    for keypoint, current_col, geo_col in (
        ("P10_peduncle_depth_dorsal", "current_P10_error_mm", "geometric_P10_error_mm"),
        ("P11_peduncle_depth_ventral", "current_P11_error_mm", "geometric_P11_error_mm"),
        ("peduncle_depth_mm", "current_peduncle_depth_error_mm", "geometric_peduncle_depth_error_mm"),
    ):
        current = pd.to_numeric(gt_df.get(current_col, pd.Series(dtype=float)), errors="coerce").dropna()
        geo = pd.to_numeric(gt_df.get(geo_col, pd.Series(dtype=float)), errors="coerce").dropna()
        paired = gt_df[[current_col, geo_col]].apply(pd.to_numeric, errors="coerce").dropna() if not gt_df.empty else pd.DataFrame()
        summary_rows.append(
            {
                "keypoint_name": keypoint,
                "num_images": int(len(paired)),
                "v06_median_error_mm": float(current.median()) if len(current) else "",
                "geometric_median_error_mm": float(geo.median()) if len(geo) else "",
                "v06_mean_error_mm": float(current.mean()) if len(current) else "",
                "geometric_mean_error_mm": float(geo.mean()) if len(geo) else "",
                "geometric_better_count": int((paired[geo_col] < paired[current_col]).sum()) if not paired.empty else 0,
                "v06_better_count": int((paired[current_col] <= paired[geo_col]).sum()) if not paired.empty else 0,
                "recommend_use_geometric": bool(len(paired) and paired[geo_col].median() < paired[current_col].median()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    qc = result.groupby(["subset", "peduncle_depth_geometric_qc_pass"], dropna=False).size().reset_index(name="count") if not result.empty else pd.DataFrame()
    summary.to_csv(OUT_DIR / "peduncle_depth_source_comparison.csv", index=False, encoding="utf-8-sig")
    qc.to_csv(OUT_DIR / "peduncle_depth_geometric_qc_summary.csv", index=False, encoding="utf-8-sig")
    readme = ["# v0.6.8 Review Overlay and Peduncle-Depth Geometry", ""]
    readme.append("This evaluation does not modify corrected keypoints or historical measurement files.")
    readme.append("")
    readme.append(summary.to_markdown(index=False) if not summary.empty else "No ground-truth comparison rows available.")
    readme.append("")
    readme.append("Recommendation: keep geometric P10/P11 as Review Queue QC/reference unless its median error is consistently lower than v06/current points.")
    (OUT_DIR / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print("Finished v0.6.8 review overlay simplification and peduncle-depth geometry.")
    print("Review Queue simplified overlay: yes")
    print("Only geometric axis shown by default: yes")
    print("Only geometric body-depth line shown by default: yes")
    for _, row in summary.iterrows():
        print(f"{row['keypoint_name']}: v06/geometric median {row['v06_median_error_mm']} / {row['geometric_median_error_mm']}")
    print("Corrected keypoints protected: yes")
    print("Historical results unchanged: yes")


if __name__ == "__main__":
    main()
