from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph.body_contour_midline import estimate_body_contour_midline_points
from siganusmorph.hybrid_point_selector import derive_hybrid_p7v
from siganusmorph.image_utils import load_image_file
from siganusmorph.keypointwise_hybrid_selector import KEYPOINT_NAMES
from siganusmorph.local_normal_measurement import estimate_body_depth_by_body_midline_normals
from siganusmorph.segmentation import segment_fish_from_blue_board
from siganusmorph.tail_geometry import apply_tail_geometry_rules, validate_p6_geometry


MM_PER_PIXEL = 0.1
OUT = PROJECT_ROOT / "results" / "model_eval" / "v0.6.5_tail_and_body_depth_geometry_fix"
VIS_OUT = OUT / "visual_comparisons"
SAME_SET = PROJECT_ROOT / "results" / "model_eval" / "v0.6_keypointwise_hybrid_selector" / "same_set_comparison"
STRICT_MANIFEST = SAME_SET / "strict_union_manifest.csv"
V06_PREDICTIONS = SAME_SET / "predictions" / "v06_predictions.csv"
BATCH_MANIFEST = PROJECT_ROOT / "results" / "batch_measurement_v0.6.4_stable" / "batch_measurement_manifest.csv"

P5 = "P5_caudal_base_midpoint"
P6 = "P6_caudal_fork_midpoint"
P7U = "P7U_caudal_fin_upper_tip"
P7L = "P7L_caudal_fin_lower_tip"
P8 = "P8_body_depth_dorsal"
P9 = "P9_body_depth_ventral"


def resolve_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def xy_from(value: Any) -> tuple[float, float] | None:
    if isinstance(value, Mapping):
        if "x" in value and "y" in value:
            return float(value["x"]), float(value["y"])
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return None


def corrected_keypoints(path: Path) -> dict[str, list[float]]:
    payload = load_json(path)
    points = payload.get("corrected_keypoints") or payload.get("keypoints") or {}
    out: dict[str, list[float]] = {}
    for key in KEYPOINT_NAMES:
        xy = xy_from(points.get(key))
        if xy is not None:
            out[key] = [xy[0], xy[1]]
    return out


def prediction_to_image_map(path: Path) -> dict[str, dict[str, list[float]]]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    out: dict[str, dict[str, list[float]]] = {}
    for row in frame.to_dict("records"):
        out.setdefault(str(row["image_name"]), {})[str(row["keypoint_name"])] = [float(row["x"]), float(row["y"])]
    return out


def dist_mm(a: Any, b: Any) -> float:
    aa = xy_from(a)
    bb = xy_from(b)
    if aa is None or bb is None:
        return float("nan")
    return math.hypot(aa[0] - bb[0], aa[1] - bb[1]) * MM_PER_PIXEL


def tail_axis_from_lobes(points: Mapping[str, Any]) -> tuple[float, float] | None:
    p5 = xy_from(points.get(P5))
    p7u = xy_from(points.get(P7U))
    p7l = xy_from(points.get(P7L))
    if p5 is None or p7u is None or p7l is None:
        return None
    mid = ((p7u[0] + p7l[0]) / 2.0, (p7u[1] + p7l[1]) / 2.0)
    dx = mid[0] - p5[0]
    dy = mid[1] - p5[1]
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        return None
    return dx / length, dy / length


def build_eval_manifest() -> pd.DataFrame:
    rows: dict[str, dict[str, Any]] = {}
    if STRICT_MANIFEST.exists():
        strict = pd.read_csv(STRICT_MANIFEST)
        for record in strict.to_dict("records"):
            image_name = str(record["image_name"])
            rows[image_name] = {
                "image_name": image_name,
                "specimen_id": record.get("specimen_id", ""),
                "subset_source": "same_set_41",
                "in_same_set_41": True,
                "in_final_analysis_96": False,
                "corrected_json_path": record.get("corrected_json_path", ""),
                "warped_image_path": record.get("warped_image_path", ""),
                "crop_image_path": record.get("crop_image_path", ""),
            }
    if BATCH_MANIFEST.exists():
        batch = pd.read_csv(BATCH_MANIFEST)
        batch = batch[batch["include_in_batch_measurement"].astype(str).str.lower().isin(["true", "1"])]
        for record in batch.to_dict("records"):
            image_name = str(record["image_name"])
            target = rows.setdefault(
                image_name,
                {
                    "image_name": image_name,
                    "specimen_id": record.get("specimen_id", ""),
                    "subset_source": "final_analysis_96",
                    "in_same_set_41": False,
                    "in_final_analysis_96": True,
                    "corrected_json_path": record.get("corrected_json_path", ""),
                    "warped_image_path": record.get("warped_image_path", ""),
                    "crop_image_path": record.get("crop_image_path", ""),
                },
            )
            target["in_final_analysis_96"] = True
            if target.get("subset_source") == "same_set_41":
                target["subset_source"] = "same_set_41+final_analysis_96"
            # Keep same-set review corrected JSON when present; otherwise use batch.
            if not target.get("corrected_json_path"):
                target["corrected_json_path"] = record.get("corrected_json_path", "")
            if not target.get("warped_image_path"):
                target["warped_image_path"] = record.get("warped_image_path", "")
            if not target.get("crop_image_path"):
                target["crop_image_path"] = record.get("crop_image_path", "")
    return pd.DataFrame(rows.values()).sort_values("image_name").reset_index(drop=True)


def body_depth_row(
    record: Mapping[str, Any],
    gt: Mapping[str, Any],
    current: Mapping[str, Any],
    body_depth: Mapping[str, Any],
) -> dict[str, Any]:
    p8_geo = body_depth.get("P8_geometric")
    p9_geo = body_depth.get("P9_geometric")
    return {
        "image_name": record.get("image_name"),
        "specimen_id": record.get("specimen_id"),
        "subset_source": record.get("subset_source"),
        "in_same_set_41": record.get("in_same_set_41"),
        "in_final_analysis_96": record.get("in_final_analysis_96"),
        "P8_v06_error_mm": dist_mm(current.get(P8), gt.get(P8)) if record.get("in_same_set_41") else np.nan,
        "P8_geometric_error_mm": dist_mm(p8_geo, gt.get(P8)),
        "P9_v06_error_mm": dist_mm(current.get(P9), gt.get(P9)) if record.get("in_same_set_41") else np.nan,
        "P9_geometric_error_mm": dist_mm(p9_geo, gt.get(P9)),
        "body_depth_v06_mm": dist_mm(current.get(P8), current.get(P9)) if record.get("in_same_set_41") else np.nan,
        "body_depth_geometric_mm": body_depth.get("body_depth_geometric_mm"),
        "body_depth_manual_mm": dist_mm(gt.get(P8), gt.get(P9)),
        "body_depth_geometric_qc_pass": body_depth.get("body_depth_geometric_qc_pass", False),
        "body_depth_geometric_review_reason": body_depth.get("body_depth_geometric_review_reason", ""),
        "body_depth_source": body_depth.get("body_depth_source", "body_midline_normal_max_width"),
        "body_depth_selected_section": json.dumps(body_depth.get("body_depth_selected_section", {}), ensure_ascii=False),
    }


def draw_visual(
    record: Mapping[str, Any],
    gt: Mapping[str, Any],
    current: Mapping[str, Any],
    body_result: Mapping[str, Any],
    body_depth: Mapping[str, Any],
    p6_qc: Mapping[str, Any],
    p6_fallback: Any,
) -> None:
    warped_path = resolve_path(record.get("warped_image_path"))
    if warped_path is None or not warped_path.exists():
        return
    image = Image.open(warped_path).convert("RGB")
    scale = min(1.0, 1800 / image.width)
    display = image.resize((int(image.width * scale), int(image.height * scale))) if scale < 1 else image.copy()
    draw = ImageDraw.Draw(display, "RGBA")
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font = ImageFont.load_default()

    def sxy(point: Any) -> tuple[float, float] | None:
        xy = xy_from(point)
        if xy is None:
            return None
        return xy[0] * scale, xy[1] * scale

    def dot(point: Any, label: str, fill: tuple[int, int, int, int], r: int = 5, outline: tuple[int, int, int, int] = (0, 0, 0, 240)) -> None:
        xy = sxy(point)
        if xy is None:
            return
        x, y = xy
        draw.ellipse([x - r, y - r, x + r, y + r], fill=fill, outline=outline, width=2)
        draw.text((x + r + 2, y - r), label, fill=fill, font=font)

    def line(points: list[Any], color: tuple[int, int, int, int], width: int = 3) -> None:
        pts = [sxy(p) for p in points]
        pts = [p for p in pts if p is not None]
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=width)

    body_midline = body_result.get("body_midline_points", [])
    if isinstance(body_midline, list) and len(body_midline) >= 2:
        line(body_midline[::3], (0, 100, 255, 190), 2)
    line(body_result.get("dorsal_body_contour", []), (80, 210, 255, 150), 2)
    line(body_result.get("ventral_body_contour", []), (80, 210, 255, 150), 2)

    p8_geo = body_depth.get("P8_geometric")
    p9_geo = body_depth.get("P9_geometric")
    dot(gt.get(P8), "P8 gt", (0, 220, 60, 240), 5)
    dot(gt.get(P9), "P9 gt", (0, 220, 60, 240), 5)
    dot(current.get(P8), "P8 v06", (230, 30, 40, 220), 5)
    dot(current.get(P9), "P9 v06", (230, 30, 40, 220), 5)
    dot(p8_geo, "P8 geom", (0, 130, 255, 240), 7)
    dot(p9_geo, "P9 geom", (0, 130, 255, 240), 7)
    line([p8_geo, p9_geo], (0, 220, 255, 240), 4)

    for key, label in [(P5, "P5"), (P6, "P6"), (P7U, "P7U"), (P7L, "P7L")]:
        dot(current.get(key), label, (255, 255, 255, 230), 4)
    dot(p6_fallback, "P6 fb", (0, 220, 255, 230), 6)
    if not bool(p6_qc.get("P6_geometry_qc_pass", False)):
        xy = sxy(current.get(P6))
        if xy is not None:
            x, y = xy
            draw.ellipse([x - 18, y - 18, x + 18, y + 18], outline=(255, 235, 0, 255), width=5)

    lines = [
        f"{record.get('image_name')} v0.6.5 geometry QC",
        f"P6 QC={p6_qc.get('P6_geometry_qc_pass')} fallback={p6_qc.get('P6_fallback_used')} {p6_qc.get('P6_review_reason','')}",
        f"body depth QC={body_depth.get('body_depth_geometric_qc_pass')} {body_depth.get('body_depth_geometric_review_reason','')}",
    ]
    draw.rectangle([0, 0, 1180, 74], fill=(0, 0, 0, 220))
    for i, text in enumerate(lines):
        draw.text((8, 6 + 22 * i), text, fill=(255, 255, 255, 255) if i == 0 else (220, 235, 255, 255), font=font)
    VIS_OUT.mkdir(parents=True, exist_ok=True)
    display.save(VIS_OUT / f"{Path(str(record['image_name'])).stem}_v065_geometry_compare.png")


def summarize_body_depth(by_image: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, v06_col, geo_col in [
        (P8, "P8_v06_error_mm", "P8_geometric_error_mm"),
        (P9, "P9_v06_error_mm", "P9_geometric_error_mm"),
    ]:
        subset = by_image[by_image["in_same_set_41"] == True]
        v06 = pd.to_numeric(subset[v06_col], errors="coerce")
        geo = pd.to_numeric(subset[geo_col], errors="coerce")
        rows.append(
            {
                "keypoint_name": key,
                "num_images": int(subset[geo_col].notna().sum()),
                "v06_median_error_mm": float(v06.median()),
                "geometric_median_error_mm": float(geo.median()),
                "v06_mean_error_mm": float(v06.mean()),
                "geometric_mean_error_mm": float(geo.mean()),
                "geometric_better_count": int((geo < v06).sum()),
                "v06_better_count": int((v06 < geo).sum()),
                "recommend_use_geometric": bool(geo.median() < v06.median() and int((geo < v06).sum()) >= int((v06 < geo).sum())),
                "notes": "Compared on strict same-set images with v06 predictions; manual P8/P9 can be subjective.",
            }
        )
    v06_depth = pd.to_numeric(by_image.loc[by_image["in_same_set_41"] == True, "body_depth_v06_mm"], errors="coerce")
    geo_depth = pd.to_numeric(by_image.loc[by_image["in_same_set_41"] == True, "body_depth_geometric_mm"], errors="coerce")
    manual_depth = pd.to_numeric(by_image.loc[by_image["in_same_set_41"] == True, "body_depth_manual_mm"], errors="coerce")
    rows.append(
        {
            "keypoint_name": "body_depth_mm",
            "num_images": int(manual_depth.notna().sum()),
            "v06_median_error_mm": float((v06_depth - manual_depth).abs().median()),
            "geometric_median_error_mm": float((geo_depth - manual_depth).abs().median()),
            "v06_mean_error_mm": float((v06_depth - manual_depth).abs().mean()),
            "geometric_mean_error_mm": float((geo_depth - manual_depth).abs().mean()),
            "geometric_better_count": int(((geo_depth - manual_depth).abs() < (v06_depth - manual_depth).abs()).sum()),
            "v06_better_count": int(((v06_depth - manual_depth).abs() < (geo_depth - manual_depth).abs()).sum()),
            "recommend_use_geometric": bool((geo_depth - manual_depth).abs().median() < (v06_depth - manual_depth).abs().median()),
            "notes": "Width-level comparison against manual P8-P9 width.",
        }
    )
    return pd.DataFrame(rows)


def write_readme(
    p6_summary: pd.DataFrame,
    p6_errors: pd.DataFrame,
    body_comparison: pd.DataFrame,
    body_by_image: pd.DataFrame,
    num_images: int,
) -> None:
    def fmt(value: Any) -> str:
        try:
            return f"{float(value):.3f}"
        except Exception:
            return ""

    failed = int((p6_summary["P6_geometry_qc_pass"] == False).sum())
    fallback = int((p6_summary["P6_fallback_used"] == True).sum())
    body_fail = int((body_by_image["body_depth_geometric_qc_pass"] == False).sum())
    lines = ["# v0.6.5 Tail and Body-Depth Geometry Fix Evaluation", ""]
    lines.append("No model was trained. Corrected keypoints and historical batch outputs were not modified.")
    lines.append("")
    lines.append(f"Images evaluated: `{num_images}`")
    lines.append("")
    lines.append("## P6 Geometry QC")
    lines.append("")
    lines.append(f"- P6 geometry failed cases: `{failed}`")
    lines.append(f"- P6 fallback candidates available/used: `{fallback}`")
    lines.append(f"- P6 error-case rows: `{len(p6_errors)}`")
    lines.append("")
    lines.append("The P6 constraints check whether P6 lies between P5 and the caudal tips along the tail axis, is closer to both tail lobes than P5, and is not excessively biased toward one lobe.")
    lines.append("")
    lines.append("## Body Depth Geometry")
    lines.append("")
    lines.append("| item | v06 median error | geometric median error | v06 mean error | geometric mean error | geometric better | v06 better | recommend geometric |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for row in body_comparison.to_dict("records"):
        lines.append(
            f"| {row['keypoint_name']} | {fmt(row['v06_median_error_mm'])} | {fmt(row['geometric_median_error_mm'])} | "
            f"{fmt(row['v06_mean_error_mm'])} | {fmt(row['geometric_mean_error_mm'])} | "
            f"{row['geometric_better_count']} | {row['v06_better_count']} | {row['recommend_use_geometric']} |"
        )
    lines.append("")
    lines.append(f"Body-depth geometric QC failed/warning cases: `{body_fail}`")
    lines.append("")
    lines.append("## Answers")
    lines.append("")
    lines.append(f"1. P6 distance/projection constraints detect suspicious P6 cases: {'yes' if failed else 'no issues found in this set'}.")
    lines.append(f"2. P6 fallback to mask_fork_rule is available in `{fallback}` cases; keep it QC-gated rather than overwriting confirmed points.")
    p8 = body_comparison[body_comparison["keypoint_name"] == P8]
    p9 = body_comparison[body_comparison["keypoint_name"] == P9]
    geom_better = bool((not p8.empty and bool(p8["recommend_use_geometric"].iloc[0])) or (not p9.empty and bool(p9["recommend_use_geometric"].iloc[0])))
    lines.append(f"3. Body-midline normal P8/P9 is better than v06 for at least one point: {'yes' if geom_better else 'no'}")
    lines.append("4. Fin interference is handled by body_core_mask and smoothed width profiles; failed QC rows should be reviewed in visual_comparisons/.")
    lines.append("5. Recommendation: keep P6 geometry QC enabled; use P6 fallback only for unconfirmed preannotations.")
    lines.append(
        f"6. Geometric P8/P9 default: {'consider after manual review' if geom_better else 'no'}; as QC suggestion: yes."
    )
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    for name in [
        "P6_geometry_qc_summary.csv",
        "P6_error_cases.csv",
        "body_depth_source_comparison.csv",
        "body_depth_error_by_image.csv",
        "body_depth_geometric_qc_summary.csv",
        "visual_comparisons/",
    ]:
        lines.append(f"- `{name}`")
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    VIS_OUT.mkdir(parents=True, exist_ok=True)
    manifest = build_eval_manifest()
    manifest.to_csv(OUT / "v065_evaluation_manifest.csv", index=False, encoding="utf-8-sig")
    v06_map = prediction_to_image_map(V06_PREDICTIONS)

    p6_rows = []
    p6_error_rows = []
    body_rows = []
    body_qc_rows = []

    for idx, record in enumerate(manifest.to_dict("records"), start=1):
        image_name = str(record["image_name"])
        gt_path = resolve_path(record.get("corrected_json_path"))
        warped_path = resolve_path(record.get("warped_image_path"))
        if gt_path is None or not gt_path.exists() or warped_path is None or not warped_path.exists():
            continue
        gt = corrected_keypoints(gt_path)
        current = v06_map.get(image_name, gt)
        image = load_image_file(warped_path)
        fish_mask, fish_bbox, segmentation_quality = segment_fish_from_blue_board(image)

        tail_rule = apply_tail_geometry_rules(
            warped_image=image,
            fish_mask=fish_mask,
            fish_bbox=fish_bbox,
            corrected_keypoints=current,
            model_keypoints_raw=current,
            confidences={},
            segmentation_success=True,
        )
        tail_geo = tail_rule.get("tail_geometry_qc", {}) if isinstance(tail_rule.get("tail_geometry_qc", {}), Mapping) else {}
        p6_fallback = (tail_rule.get("geometric_suggestions", {}) or {}).get(P6) or tail_geo.get("P6_gap") or tail_geo.get("P6_concavity")
        p7v_payload = derive_hybrid_p7v(current)
        p7v_point = p7v_payload.get("point")
        p6_qc = validate_p6_geometry(
            xy_from(current.get(P5)),
            xy_from(current.get(P6)),
            xy_from(current.get(P7U)),
            xy_from(current.get(P7L)),
            xy_from(p7v_point),
            tail_axis_from_lobes(current),
            fish_mask=fish_mask,
            tail_mask=tail_rule.get("tail_detail_mask"),
            config={"P6_mask_fork_rule": p6_fallback},
        )
        p6_row = {
            "image_name": image_name,
            "specimen_id": record.get("specimen_id", ""),
            "subset_source": record.get("subset_source", ""),
            "P6_geometry_qc_pass": p6_qc.get("P6_geometry_qc_pass", False),
            "P6_projection_ok": p6_qc.get("P6_projection_ok", False),
            "P6_projection_between_P5_and_tailtips": p6_qc.get("P6_projection_between_P5_and_tailtips", False),
            "P6_distance_to_P7U_ok": p6_qc.get("P6_distance_to_P7U_ok", False),
            "P6_distance_to_P7L_ok": p6_qc.get("P6_distance_to_P7L_ok", False),
            "P6_tail_leaf_balance_ok": p6_qc.get("P6_tail_leaf_balance_ok", False),
            "P6_fallback_used": p6_qc.get("P6_fallback_used", False),
            "P6_review_reason": p6_qc.get("P6_review_reason", ""),
            "P6_current_error_mm": dist_mm(current.get(P6), gt.get(P6)) if record.get("in_same_set_41") else np.nan,
            "P6_fallback_error_mm": dist_mm(p6_fallback, gt.get(P6)) if record.get("in_same_set_41") else np.nan,
            "segmentation_quality": segmentation_quality,
        }
        p6_rows.append(p6_row)
        if (not bool(p6_row["P6_geometry_qc_pass"])) or (isinstance(p6_row["P6_current_error_mm"], float) and p6_row["P6_current_error_mm"] > 10):
            p6_error_rows.append(p6_row)

        body_result = estimate_body_contour_midline_points(
            fish_mask,
            current,
            image.shape,
            config={"mm_per_pixel": MM_PER_PIXEL},
        )
        body_depth = estimate_body_depth_by_body_midline_normals(
            fish_mask,
            body_result.get("body_core_mask"),
            body_result,
            current,
            image.shape,
            config={"mm_per_pixel": MM_PER_PIXEL},
        )
        body_rows.append(body_depth_row(record, gt, current, body_depth))
        body_qc_rows.append(
            {
                "image_name": image_name,
                "specimen_id": record.get("specimen_id", ""),
                "subset_source": record.get("subset_source", ""),
                "body_depth_geometric_qc_pass": body_depth.get("body_depth_geometric_qc_pass", False),
                "body_depth_geometric_review_reason": body_depth.get("body_depth_geometric_review_reason", ""),
                "body_depth_source": body_depth.get("body_depth_source", ""),
                "mask_source": body_depth.get("mask_source", ""),
                "body_midline_qc_pass": (body_result.get("body_midline_qc") or {}).get("body_midline_qc_pass", ""),
                "body_midline_qc_reason": (body_result.get("body_midline_qc") or {}).get("body_midline_qc_reason", ""),
                "segmentation_quality": segmentation_quality,
            }
        )
        draw_visual(record, gt, current, body_result, body_depth, p6_qc, p6_fallback)
        print(f"[{idx}/{len(manifest)}] v0.6.5 geometry: {image_name}")

    p6_summary = pd.DataFrame(p6_rows)
    p6_errors = pd.DataFrame(p6_error_rows)
    body_by_image = pd.DataFrame(body_rows)
    body_qc = pd.DataFrame(body_qc_rows)
    body_comparison = summarize_body_depth(body_by_image)

    p6_summary.to_csv(OUT / "P6_geometry_qc_summary.csv", index=False, encoding="utf-8-sig")
    p6_errors.to_csv(OUT / "P6_error_cases.csv", index=False, encoding="utf-8-sig")
    body_comparison.to_csv(OUT / "body_depth_source_comparison.csv", index=False, encoding="utf-8-sig")
    body_by_image.to_csv(OUT / "body_depth_error_by_image.csv", index=False, encoding="utf-8-sig")
    body_qc.to_csv(OUT / "body_depth_geometric_qc_summary.csv", index=False, encoding="utf-8-sig")
    write_readme(p6_summary, p6_errors, body_comparison, body_by_image, len(manifest))

    p8_row = body_comparison[body_comparison["keypoint_name"] == P8].iloc[0]
    p9_row = body_comparison[body_comparison["keypoint_name"] == P9].iloc[0]
    print("Finished v0.6.5 tail and body-depth geometry fix evaluation.")
    print("P6 QC:")
    print(f"failed cases: {int((p6_summary['P6_geometry_qc_pass'] == False).sum())}")
    print(f"fallback cases: {int((p6_summary['P6_fallback_used'] == True).sum())}")
    print("recommended action: enable P6 geometry QC; keep fallback QC-gated")
    print("Body depth:")
    print(f"P8 v06/geometric median: {float(p8_row['v06_median_error_mm']):.3f} / {float(p8_row['geometric_median_error_mm']):.3f}")
    print(f"P9 v06/geometric median: {float(p9_row['v06_median_error_mm']):.3f} / {float(p9_row['geometric_median_error_mm']):.3f}")
    print("body_depth v06/geometric comparison: see body_depth_source_comparison.csv")
    print("Recommendation:")
    print("Use P6 geometry QC: yes")
    print("Use P6 fallback: yes, only for unconfirmed preannotation or review suggestion")
    print(f"Use geometric P8/P9 as default: {'yes' if bool(body_comparison['recommend_use_geometric'].all()) else 'no'}")
    print("Use geometric P8/P9 as QC suggestion: yes")


if __name__ == "__main__":
    main()
