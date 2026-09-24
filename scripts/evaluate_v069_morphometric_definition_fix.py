"""Evaluate v0.6.9 morphometric definition fixes.

Read-only evaluation for:
- trunk-boundary geometric P8/P9 body-depth sections
- P4-P5 normal geometric P10/P11 peduncle-depth sections
- P3 operculum posterior QC/suggestion
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

from siganusmorph.body_contour_midline import estimate_body_contour_midline_points  # noqa: E402
from siganusmorph.image_utils import ensure_rgb, load_image_file  # noqa: E402
from siganusmorph.local_normal_measurement import (  # noqa: E402
    estimate_body_depth_by_body_midline_normals,
    estimate_peduncle_depth_by_axis_normals,
)
from siganusmorph.operculum_geometry import validate_p3_operculum_position  # noqa: E402
from siganusmorph.realworld_review import xy_from_payload  # noqa: E402
from siganusmorph.segmentation import segment_fish_from_blue_board  # noqa: E402


OUT_DIR = PROJECT_ROOT / "results" / "model_eval" / "v0.6.9_morphometric_definition_fix"
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


def _points(payload: Mapping[str, Any], key: str = "corrected_keypoints") -> dict[str, Any]:
    value = payload.get(key, {})
    return dict(value) if isinstance(value, Mapping) else {}


def _pre_points(payload: Mapping[str, Any]) -> dict[str, Any]:
    for container in (
        payload.get("preannotation_metadata", {}) if isinstance(payload.get("preannotation_metadata", {}), Mapping) else {},
        payload,
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


def _depth_mm(points: Mapping[str, Any], a: str, b: str) -> float | None:
    return _dist_mm(points.get(a), points.get(b))


def _geometry_payload_point(geometry: Mapping[str, Any], key: str) -> list[float] | None:
    value = geometry.get(key)
    xy = xy_from_payload(value)
    return [float(xy[0]), float(xy[1])] if xy else None


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
                    "preannotation_json_path": "",
                }
            )
    review_manifest = PROJECT_ROOT / "results" / "realworld_review_5_24_v0.6.5" / "review_sample_manifest.csv"
    if review_manifest.exists():
        df = pd.read_csv(review_manifest, dtype=str, keep_default_na=False)
        if "preannotation_success" in df.columns:
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


def _draw_visual(image: Any, out_path: Path, *, title: str, current: Mapping[str, Any], gt: Mapping[str, Any], body: Mapping[str, Any], ped: Mapping[str, Any], p3qc: Mapping[str, Any]) -> None:
    canvas = Image.fromarray(ensure_rgb(image)).convert("RGB")
    max_side = 1400
    scale = 1.0
    if max(canvas.size) > max_side:
        scale = max_side / float(max(canvas.size))
        canvas = canvas.resize((max(1, round(canvas.width * scale)), max(1, round(canvas.height * scale))), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(canvas)
    radius = max(4, round(min(canvas.size) / 260))
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 16)
    except OSError:
        font = ImageFont.load_default()

    def xy(point: Any) -> tuple[float, float] | None:
        parsed = xy_from_payload(point)
        return (parsed[0] * scale, parsed[1] * scale) if parsed else None

    def dot(point: Any, label: str, color: tuple[int, int, int]) -> tuple[float, float] | None:
        p = xy(point)
        if p is None:
            return None
        draw.ellipse((p[0] - radius, p[1] - radius, p[0] + radius, p[1] + radius), fill=color, outline=(255, 255, 255), width=2)
        draw.text((p[0] + radius + 2, p[1] - radius), label, font=font, fill=color)
        return p

    for key, label in (
        ("P8_body_depth_dorsal", "P8_cur"),
        ("P9_body_depth_ventral", "P9_cur"),
        ("P10_peduncle_depth_dorsal", "P10_cur"),
        ("P11_peduncle_depth_ventral", "P11_cur"),
        ("P3_operculum_posterior", "P3_cur"),
    ):
        dot(current.get(key), label, (240, 55, 65))
        if gt:
            dot(gt.get(key), label.replace("_cur", "_gt"), (60, 220, 90))

    p8g = dot(body.get("P8_geometric"), "P8_geo", (0, 135, 255))
    p9g = dot(body.get("P9_geometric"), "P9_geo", (0, 135, 255))
    if p8g and p9g:
        draw.line((p8g[0], p8g[1], p9g[0], p9g[1]), fill=(0, 230, 255), width=4)
    p10g = dot(ped.get("P10_geometric"), "P10_geo", (0, 220, 180))
    p11g = dot(ped.get("P11_geometric"), "P11_geo", (0, 220, 180))
    if p10g and p11g:
        draw.line((p10g[0], p10g[1], p11g[0], p11g[1]), fill=(0, 245, 190), width=4)
    p3s = dot(p3qc.get("P3_operculum_edge_suggestion"), "P3_edge", (255, 80, 210))
    p3c = xy(current.get("P3_operculum_posterior"))
    if p3s and p3c:
        draw.line((p3s[0], p3s[1], p3c[0], p3c[1]), fill=(255, 150, 230), width=2)
    draw.rectangle((8, 8, min(canvas.width - 8, 24 + len(title) * 8), 36), fill=(0, 0, 0))
    draw.text((14, 11), title, font=font, fill=(255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, optimize=True)


def _summarize_pair(df: pd.DataFrame, label: str, current_col: str, geo_col: str) -> dict[str, Any]:
    paired = df[[current_col, geo_col]].apply(pd.to_numeric, errors="coerce").dropna() if not df.empty else pd.DataFrame()
    current = pd.to_numeric(df.get(current_col, pd.Series(dtype=float)), errors="coerce").dropna()
    geo = pd.to_numeric(df.get(geo_col, pd.Series(dtype=float)), errors="coerce").dropna()
    return {
        "metric": label,
        "num_images": int(len(paired)),
        "current_median_error_mm": float(current.median()) if len(current) else "",
        "geometric_median_error_mm": float(geo.median()) if len(geo) else "",
        "current_mean_error_mm": float(current.mean()) if len(current) else "",
        "geometric_mean_error_mm": float(geo.mean()) if len(geo) else "",
        "geometric_better_count": int((paired[geo_col] < paired[current_col]).sum()) if not paired.empty else 0,
        "current_better_count": int((paired[current_col] <= paired[geo_col]).sum()) if not paired.empty else 0,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for record in _records():
        image_path = _resolve(str(record.get("warped_image_path", "")))
        if not image_path.exists():
            continue
        image = load_image_file(image_path)
        corrected_payload = _read_json(_resolve(str(record.get("corrected_json_path", "")))) if record.get("corrected_json_path") else {}
        pre_payload = _read_json(_resolve(str(record.get("preannotation_json_path", "")))) if record.get("preannotation_json_path") else {}
        gt = _points(corrected_payload)
        current = _pre_points(pre_payload) or _pre_points(corrected_payload) or gt
        try:
            fish_mask, _, _ = segment_fish_from_blue_board(image)
            body_midline = estimate_body_contour_midline_points(fish_mask, current, image.shape, config={"mm_per_pixel": MM_PER_PIXEL})
            body_geometry = estimate_body_depth_by_body_midline_normals(fish_mask, body_midline.get("body_core_mask"), body_midline, current, image.shape, config={"mm_per_pixel": MM_PER_PIXEL})
            ped_geometry = estimate_peduncle_depth_by_axis_normals(fish_mask, current, image.shape, config={"mm_per_pixel": MM_PER_PIXEL})
            p3_qc = validate_p3_operculum_position(
                current.get("P1_snout_tip"),
                current.get("P2_eye_front"),
                current.get("P3_operculum_posterior"),
                current.get("C1_head_axis_point"),
                current.get("P4_peduncle_start_midpoint"),
                body_midline,
                fish_mask,
                image,
                config={"mm_per_pixel": MM_PER_PIXEL},
            )
        except Exception as exc:  # noqa: BLE001
            body_geometry = {"body_depth_geometric_qc_pass": False, "body_depth_geometric_review_reason": f"failed:{exc}"}
            ped_geometry = {"peduncle_depth_geometric_qc_pass": False, "peduncle_depth_geometric_review_reason": f"failed:{exc}"}
            p3_qc = {"P3_qc_pass": False, "P3_needs_review": True, "P3_review_reason": f"failed:{exc}"}
        gt_body_depth = _depth_mm(gt, "P8_body_depth_dorsal", "P9_body_depth_ventral") if gt else None
        cur_body_depth = _depth_mm(current, "P8_body_depth_dorsal", "P9_body_depth_ventral")
        geo_body_depth = body_geometry.get("body_depth_geometric_mm")
        gt_ped_depth = _depth_mm(gt, "P10_peduncle_depth_dorsal", "P11_peduncle_depth_ventral") if gt else None
        cur_ped_depth = _depth_mm(current, "P10_peduncle_depth_dorsal", "P11_peduncle_depth_ventral")
        geo_ped_depth = ped_geometry.get("peduncle_depth_geometric_mm")
        row = {
            **record,
            "has_ground_truth": bool(gt),
            "current_P8_error_mm": _dist_mm(current.get("P8_body_depth_dorsal"), gt.get("P8_body_depth_dorsal")) if gt else None,
            "geometric_P8_error_mm": _dist_mm(_geometry_payload_point(body_geometry, "P8_geometric"), gt.get("P8_body_depth_dorsal")) if gt else None,
            "current_P9_error_mm": _dist_mm(current.get("P9_body_depth_ventral"), gt.get("P9_body_depth_ventral")) if gt else None,
            "geometric_P9_error_mm": _dist_mm(_geometry_payload_point(body_geometry, "P9_geometric"), gt.get("P9_body_depth_ventral")) if gt else None,
            "current_body_depth_error_mm": abs(cur_body_depth - gt_body_depth) if cur_body_depth is not None and gt_body_depth is not None else None,
            "geometric_body_depth_error_mm": abs(float(geo_body_depth) - gt_body_depth) if geo_body_depth not in ("", None) and gt_body_depth is not None else None,
            "body_depth_geometric_qc_pass": body_geometry.get("body_depth_geometric_qc_pass", ""),
            "body_depth_review_suggested": body_geometry.get("body_depth_review_suggested", ""),
            "body_depth_geometric_review_reason": body_geometry.get("body_depth_geometric_review_reason", ""),
            "fin_suppression_count": len(body_geometry.get("fin_suppression_regions", []) or []),
            "current_P10_error_mm": _dist_mm(current.get("P10_peduncle_depth_dorsal"), gt.get("P10_peduncle_depth_dorsal")) if gt else None,
            "geometric_P10_error_mm": _dist_mm(_geometry_payload_point(ped_geometry, "P10_geometric"), gt.get("P10_peduncle_depth_dorsal")) if gt else None,
            "current_P11_error_mm": _dist_mm(current.get("P11_peduncle_depth_ventral"), gt.get("P11_peduncle_depth_ventral")) if gt else None,
            "geometric_P11_error_mm": _dist_mm(_geometry_payload_point(ped_geometry, "P11_geometric"), gt.get("P11_peduncle_depth_ventral")) if gt else None,
            "current_peduncle_depth_error_mm": abs(cur_ped_depth - gt_ped_depth) if cur_ped_depth is not None and gt_ped_depth is not None else None,
            "geometric_peduncle_depth_error_mm": abs(float(geo_ped_depth) - gt_ped_depth) if geo_ped_depth not in ("", None) and gt_ped_depth is not None else None,
            "peduncle_depth_geometric_qc_pass": ped_geometry.get("peduncle_depth_geometric_qc_pass", ""),
            "peduncle_depth_review_suggested": ped_geometry.get("peduncle_depth_review_suggested", ""),
            "peduncle_depth_geometric_review_reason": ped_geometry.get("peduncle_depth_geometric_review_reason", ""),
            "P3_qc_pass": p3_qc.get("P3_qc_pass", ""),
            "P3_needs_review": p3_qc.get("P3_needs_review", ""),
            "P3_review_reason": p3_qc.get("P3_review_reason", ""),
            "P3_current_to_suggestion_distance_mm": p3_qc.get("P3_current_to_suggestion_distance_mm", ""),
        }
        rows.append(row)
        if len(rows) <= 120:
            _draw_visual(image, VIS_DIR / f"{Path(str(record.get('image_name','image'))).stem}_v069.png", title=f"{record.get('image_name')} {record.get('subset')}", current=current, gt=gt, body=body_geometry, ped=ped_geometry, p3qc=p3_qc)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "morphometric_qc_by_image.csv", index=False, encoding="utf-8-sig")
    gt_df = df[df["has_ground_truth"].astype(bool)].copy() if not df.empty else pd.DataFrame()
    body_summary = pd.DataFrame(
        [
            _summarize_pair(gt_df, "P8_body_depth_dorsal", "current_P8_error_mm", "geometric_P8_error_mm"),
            _summarize_pair(gt_df, "P9_body_depth_ventral", "current_P9_error_mm", "geometric_P9_error_mm"),
            _summarize_pair(gt_df, "body_depth_mm", "current_body_depth_error_mm", "geometric_body_depth_error_mm"),
        ]
    )
    ped_summary = pd.DataFrame(
        [
            _summarize_pair(gt_df, "P10_peduncle_depth_dorsal", "current_P10_error_mm", "geometric_P10_error_mm"),
            _summarize_pair(gt_df, "P11_peduncle_depth_ventral", "current_P11_error_mm", "geometric_P11_error_mm"),
            _summarize_pair(gt_df, "peduncle_depth_mm", "current_peduncle_depth_error_mm", "geometric_peduncle_depth_error_mm"),
        ]
    )
    p3_summary = df.groupby(["subset", "P3_qc_pass", "P3_needs_review"], dropna=False).size().reset_index(name="count") if not df.empty else pd.DataFrame()
    body_summary.to_csv(OUT_DIR / "body_depth_trunk_comparison.csv", index=False, encoding="utf-8-sig")
    ped_summary.to_csv(OUT_DIR / "peduncle_depth_geometry_comparison.csv", index=False, encoding="utf-8-sig")
    p3_summary.to_csv(OUT_DIR / "P3_operculum_qc_summary.csv", index=False, encoding="utf-8-sig")
    readme = [
        "# v0.6.9 Morphometric Definition Fix",
        "",
        "This evaluation is read-only: no corrected keypoints or historical measurements are modified.",
        "",
        "## Body Depth",
        body_summary.to_markdown(index=False),
        "",
        "## Peduncle Depth",
        ped_summary.to_markdown(index=False),
        "",
        "## P3 Operculum QC",
        p3_summary.to_markdown(index=False) if not p3_summary.empty else "No P3 QC rows.",
        "",
        "Recommendation: use trunk-based P8/P9, geometric P10/P11, and P3 operculum checks as review/QC references first; do not overwrite confirmed labels.",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print("Finished v0.6.9 morphometric definition fix.")
    print("Body depth:")
    print(f"fin attachment warnings: {int(pd.Series(df.get('body_depth_review_suggested', [])).astype(str).str.lower().isin(['true', '1']).sum()) if not df.empty else 0}")
    for _, row in body_summary.iterrows():
        print(f"{row['metric']}: current/geometric median {row['current_median_error_mm']} / {row['geometric_median_error_mm']}")
    print("Peduncle depth:")
    print(f"non-perpendicular warnings: {int(pd.Series(df.get('peduncle_depth_review_suggested', [])).astype(str).str.lower().isin(['true', '1']).sum()) if not df.empty else 0}")
    for _, row in ped_summary.iterrows():
        print(f"{row['metric']}: current/geometric median {row['current_median_error_mm']} / {row['geometric_median_error_mm']}")
    print("P3:")
    print(f"operculum drift warnings: {int(pd.Series(df.get('P3_needs_review', [])).astype(str).str.lower().isin(['true', '1']).sum()) if not df.empty else 0}")
    print(f"P3 suggestion available: {int(pd.to_numeric(df.get('P3_current_to_suggestion_distance_mm', pd.Series(dtype=float)), errors='coerce').notna().sum()) if not df.empty else 0}")
    print("Corrected keypoints protected: yes")
    print("Historical results unchanged: yes")


if __name__ == "__main__":
    main()
