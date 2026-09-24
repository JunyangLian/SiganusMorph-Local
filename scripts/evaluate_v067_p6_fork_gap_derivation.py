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

from siganusmorph.hybrid_point_selector import derive_hybrid_p7v
from siganusmorph.image_utils import load_image_file
from siganusmorph.keypointwise_hybrid_selector import KEYPOINT_NAMES
from siganusmorph.segmentation import segment_fish_from_blue_board
from siganusmorph.tail_geometry import estimate_p6_from_tail_fork_gap, select_tail_axis_without_p6, validate_p6_geometry


MM_PER_PIXEL = 0.1
OUT = PROJECT_ROOT / "results" / "model_eval" / "v0.6.7_p6_fork_gap_derivation"
VIS_OUT = OUT / "visual_comparisons"
SAME_SET = PROJECT_ROOT / "results" / "model_eval" / "v0.6_keypointwise_hybrid_selector" / "same_set_comparison"
STRICT_MANIFEST = SAME_SET / "strict_union_manifest.csv"
V06_PREDICTIONS = SAME_SET / "predictions" / "v06_predictions.csv"
BATCH_MANIFEST = PROJECT_ROOT / "results" / "batch_measurement_v0.6.4_stable" / "batch_measurement_manifest.csv"

P5 = "P5_caudal_base_midpoint"
P6 = "P6_caudal_fork_midpoint"
P7U = "P7U_caudal_fin_upper_tip"
P7L = "P7L_caudal_fin_lower_tip"


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
    if isinstance(value, (list, tuple)) and len(value) >= 2 and value[0] is not None and value[1] is not None:
        return float(value[0]), float(value[1])
    return None


def serial(point: Any) -> list[float] | None:
    xy = xy_from(point)
    if xy is None:
        return None
    return [float(xy[0]), float(xy[1])]


def dist_mm(a: Any, b: Any) -> float:
    aa = xy_from(a)
    bb = xy_from(b)
    if aa is None or bb is None:
        return float("nan")
    return math.hypot(aa[0] - bb[0], aa[1] - bb[1]) * MM_PER_PIXEL


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
        if "include_in_batch_measurement" in batch.columns:
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
            if not target.get("corrected_json_path"):
                target["corrected_json_path"] = record.get("corrected_json_path", "")
            if not target.get("warped_image_path"):
                target["warped_image_path"] = record.get("warped_image_path", "")
            if not target.get("crop_image_path"):
                target["crop_image_path"] = record.get("crop_image_path", "")
    return pd.DataFrame(rows.values()).sort_values("image_name").reset_index(drop=True)


def p7v_from_axis(points: Mapping[str, Any], axis: tuple[float, float] | None) -> dict[str, Any]:
    p5 = xy_from(points.get(P5))
    p7u = xy_from(points.get(P7U))
    p7l = xy_from(points.get(P7L))
    if p5 is None or p7u is None or p7l is None or axis is None:
        return {"point": None, "p7v_valid": False, "extension_px": np.nan}
    proj_u = (p7u[0] - p5[0]) * axis[0] + (p7u[1] - p5[1]) * axis[1]
    proj_l = (p7l[0] - p5[0]) * axis[0] + (p7l[1] - p5[1]) * axis[1]
    extension = max(0.0, float(max(proj_u, proj_l)))
    point = [p5[0] + extension * axis[0], p5[1] + extension * axis[1]]
    return {"point": point, "p7v_valid": extension > 0, "extension_px": extension}


def axis_from_p6(points: Mapping[str, Any]) -> tuple[float, float] | None:
    p5 = xy_from(points.get(P5))
    p6 = xy_from(points.get(P6))
    if p5 is None or p6 is None:
        return None
    dx = p6[0] - p5[0]
    dy = p6[1] - p5[1]
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        return None
    return dx / length, dy / length


def lobe_bisector_axis(points: Mapping[str, Any]) -> tuple[float, float] | None:
    p5 = xy_from(points.get(P5))
    p7u = xy_from(points.get(P7U))
    p7l = xy_from(points.get(P7L))
    if p5 is None or p7u is None or p7l is None:
        return None
    out = []
    for point in (p7u, p7l):
        dx = point[0] - p5[0]
        dy = point[1] - p5[1]
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            return None
        out.append((dx / length, dy / length))
    bx = out[0][0] + out[1][0]
    by = out[0][1] + out[1][1]
    length = math.hypot(bx, by)
    if length <= 1e-9:
        return None
    return bx / length, by / length


def midpoint_axis(points: Mapping[str, Any]) -> tuple[float, float] | None:
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


def draw_visual(record: Mapping[str, Any], image: Any, gt: Mapping[str, Any], current: Mapping[str, Any], p6_geo: Any, p6_qc: Mapping[str, Any]) -> None:
    canvas = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    def draw_point(point: Any, color: tuple[int, int, int], label: str, r: int = 8) -> None:
        xy = xy_from(point)
        if xy is None:
            return
        x, y = xy
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=(255, 255, 255), width=2)
        draw.text((x + r + 2, y - r), label, fill=color, font=font)

    draw_point(gt.get(P6), (0, 180, 80), "P6_gt", 9)
    draw_point(current.get(P6), (230, 35, 45), "P6_v06", 7)
    draw_point(p6_geo, (0, 230, 210), "P6_gap", 8)
    cur_xy = xy_from(current.get(P6))
    geo_xy = xy_from(p6_geo)
    if cur_xy and geo_xy:
        draw.line((cur_xy[0], cur_xy[1], geo_xy[0], geo_xy[1]), fill=(0, 230, 210), width=3)
    if p6_qc.get("P6_gap_qc_pass") is False and geo_xy:
        x, y = geo_xy
        draw.ellipse((x - 18, y - 18, x + 18, y + 18), outline=(255, 230, 0), width=5)
    text = f"{record.get('image_name')} | P6 geo QC={p6_qc.get('P6_gap_qc_pass')} | {p6_qc.get('P6_review_reason','')}"
    draw.rectangle((8, 8, min(canvas.width - 8, 8 + len(text) * 7), 32), fill=(0, 0, 0))
    draw.text((12, 12), text, fill=(255, 255, 255), font=font)
    canvas.thumbnail((1400, 900))
    VIS_OUT.mkdir(parents=True, exist_ok=True)
    canvas.save(VIS_OUT / f"{Path(str(record.get('image_name'))).stem}_p6_gap_compare.png")


def summarize_errors(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    records = []
    for source, col in (("v06_current", "P6_current_error_mm"), ("p6_geometric", "P6_geometric_error_mm")):
        vals = pd.to_numeric(frame[col], errors="coerce").dropna()
        records.append(
            {
                "source": source,
                "num_images": int(vals.count()),
                "mean_error_mm": float(vals.mean()) if len(vals) else np.nan,
                "median_error_mm": float(vals.median()) if len(vals) else np.nan,
                "max_error_mm": float(vals.max()) if len(vals) else np.nan,
                "large_error_rate": float((vals > 10).mean()) if len(vals) else np.nan,
            }
        )
    return pd.DataFrame(records)


def write_readme(source_summary: pd.DataFrame, p6_rows: list[dict[str, Any]], axis_rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(p6_rows)
    axis = pd.DataFrame(axis_rows)
    current = source_summary[source_summary["source"] == "v06_current"].iloc[0].to_dict()
    geo = source_summary[source_summary["source"] == "p6_geometric"].iloc[0].to_dict()
    fail_current_geo_pass = frame[(frame["P6_current_geometry_qc_pass"] == False) & (frame["P6_gap_qc_pass"] == True)]  # noqa: E712
    improved = frame[pd.to_numeric(frame["P6_geometric_error_mm"], errors="coerce") < pd.to_numeric(frame["P6_current_error_mm"], errors="coerce")]
    lines = [
        "# v0.6.7 P6 Fork Gap Derivation",
        "",
        "This evaluation derives P6 from the V-shaped caudal fork gap and compares it with the current v0.6 P6. No labels or historical measurement outputs were modified.",
        "",
        "## P6 error summary",
        "",
        f"- Current v06 P6 median / mean: {current.get('median_error_mm'):.3f} / {current.get('mean_error_mm'):.3f} mm",
        f"- Gap-derived P6 median / mean: {geo.get('median_error_mm'):.3f} / {geo.get('mean_error_mm'):.3f} mm",
        f"- Current large-error rate: {current.get('large_error_rate'):.3f}",
        f"- Geometric large-error rate: {geo.get('large_error_rate'):.3f}",
        f"- Current fail but geometric pass cases: {len(fail_current_geo_pass)}",
        f"- Geometric improves cases: {len(improved)} / {len(frame)}",
        "",
        "## Tail axis / P7V",
        "",
    ]
    if len(axis):
        valid_rates = axis.groupby("tail_axis_source")["p7v_valid"].mean().to_dict()
        lines.extend([f"- {key}: P7V valid rate {value:.3f}" for key, value in valid_rates.items()])
    lines.extend(
        [
            "",
            "## Answers",
            "",
            "1. P6 gap geometry is useful as a QC/fallback candidate, but should only become the default if the median/large-error results beat current v06 P6 on the same set.",
            "2. P6 should not drive tail axis by default. P5 -> midpoint(P7U, P7L) and lobe-bisector axes are evaluated separately.",
            "3. P7V should continue to prefer P5 -> midpoint(P7U, P7L) when P7U/P7L are reliable.",
            "4. Confirmed P6 labels should not be overwritten; use the gap point as suggestion/QC.",
        ]
    )
    OUT.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    VIS_OUT.mkdir(parents=True, exist_ok=True)
    manifest = build_eval_manifest()
    manifest.to_csv(OUT / "v067_evaluation_manifest.csv", index=False, encoding="utf-8-sig")
    v06_predictions = prediction_to_image_map(V06_PREDICTIONS)
    rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    fallback_rows: list[dict[str, Any]] = []
    axis_rows: list[dict[str, Any]] = []

    for record in manifest.to_dict("records"):
        corrected_path = resolve_path(record.get("corrected_json_path"))
        image_path = resolve_path(record.get("warped_image_path")) or resolve_path(record.get("crop_image_path"))
        if corrected_path is None or not corrected_path.exists() or image_path is None or not image_path.exists():
            continue
        gt = corrected_keypoints(corrected_path)
        if P6 not in gt:
            continue
        image = load_image_file(image_path)
        current = v06_predictions.get(str(record["image_name"]), {}) or gt
        try:
            fish_mask, _bbox, _quality = segment_fish_from_blue_board(image)
        except Exception:
            fish_mask = None
        p6_gap = estimate_p6_from_tail_fork_gap(
            tail_mask=None,
            fish_mask=fish_mask,
            keypoints=current,
            image_shape=getattr(image, "shape", None) or [],
            config={"mm_per_pixel": MM_PER_PIXEL, "warped_image": image},
        )
        axis_payload = select_tail_axis_without_p6(current)
        axis = xy_from(axis_payload.get("tail_axis"))
        current_qc = validate_p6_geometry(
            xy_from(current.get(P5)),
            xy_from(current.get(P6)),
            xy_from(current.get(P7U)),
            xy_from(current.get(P7L)),
            None,
            axis,
            fish_mask=fish_mask,
            config={"P6_geometric": p6_gap.get("P6_geometric")},
        )
        row = {
            "image_name": record.get("image_name"),
            "specimen_id": record.get("specimen_id"),
            "subset_source": record.get("subset_source"),
            "in_same_set_41": record.get("in_same_set_41"),
            "in_final_analysis_96": record.get("in_final_analysis_96"),
            "P6_current_x": (xy_from(current.get(P6)) or (np.nan, np.nan))[0],
            "P6_current_y": (xy_from(current.get(P6)) or (np.nan, np.nan))[1],
            "P6_geometric_x": (xy_from(p6_gap.get("P6_geometric")) or (np.nan, np.nan))[0],
            "P6_geometric_y": (xy_from(p6_gap.get("P6_geometric")) or (np.nan, np.nan))[1],
            "P6_manual_x": (xy_from(gt.get(P6)) or (np.nan, np.nan))[0],
            "P6_manual_y": (xy_from(gt.get(P6)) or (np.nan, np.nan))[1],
            "P6_current_error_mm": dist_mm(current.get(P6), gt.get(P6)) if record.get("in_same_set_41") else np.nan,
            "P6_geometric_error_mm": dist_mm(p6_gap.get("P6_geometric"), gt.get(P6)),
            "P6_gap_qc_pass": bool(p6_gap.get("P6_gap_qc_pass", False)),
            "P6_current_geometry_qc_pass": bool(current_qc.get("P6_geometry_qc_pass", False)),
            "P6_fallback_recommended": bool(p6_gap.get("P6_fallback_recommended", False)),
            "P6_current_to_geometric_distance_mm": p6_gap.get("P6_current_to_geometric_distance_mm", ""),
            "P6_gap_review_reason": p6_gap.get("P6_gap_review_reason", ""),
            "P6_review_reason": p6_gap.get("P6_review_reason", ""),
            "tail_axis_source": p6_gap.get("tail_axis_source", ""),
            "tail_axis_uses_P6": p6_gap.get("tail_axis_uses_P6", ""),
        }
        rows.append(row)
        qc_rows.append({**row, **{k: v for k, v in p6_gap.items() if k not in {"tail_gap_centerline"}}})
        if row["P6_fallback_recommended"]:
            fallback_rows.append(row)
        for source, axis_value in (
            ("P6_based_axis", axis_from_p6(current)),
            ("P5_midpoint_axis", midpoint_axis(current)),
            ("lobe_bisector_axis", lobe_bisector_axis(current)),
        ):
            p7v = p7v_from_axis(current, axis_value)
            current_p7v = derive_hybrid_p7v(current)
            axis_rows.append(
                {
                    "image_name": record.get("image_name"),
                    "specimen_id": record.get("specimen_id"),
                    "tail_axis_source": source,
                    "P7V_x": (xy_from(p7v.get("point")) or (np.nan, np.nan))[0],
                    "P7V_y": (xy_from(p7v.get("point")) or (np.nan, np.nan))[1],
                    "TL_final_mm": dist_mm(current.get("P1_snout_tip"), p7v.get("point")),
                    "caudal_extension_axis_mm": float(p7v.get("extension_px", np.nan)) * MM_PER_PIXEL,
                    "p7v_valid": bool(p7v.get("p7v_valid", False)),
                    "difference_from_current_mm": dist_mm(p7v.get("point"), current_p7v.get("point")),
                    "notes": "",
                }
            )
        if len(rows) <= 120:
            draw_visual(record, image, gt, current, p6_gap.get("P6_geometric"), p6_gap)

    pd.DataFrame(rows).to_csv(OUT / "P6_error_by_image.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(qc_rows).to_csv(OUT / "P6_gap_qc_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fallback_rows).to_csv(OUT / "P6_fallback_cases.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(axis_rows).to_csv(OUT / "P7V_tail_axis_comparison.csv", index=False, encoding="utf-8-sig")
    source_summary = summarize_errors(rows)
    source_summary.to_csv(OUT / "P6_source_comparison.csv", index=False, encoding="utf-8-sig")
    write_readme(source_summary, rows, axis_rows)

    current = source_summary[source_summary["source"] == "v06_current"].iloc[0].to_dict()
    geo = source_summary[source_summary["source"] == "p6_geometric"].iloc[0].to_dict()
    frame = pd.DataFrame(rows)
    fail_current_geo_pass = frame[(frame["P6_current_geometry_qc_pass"] == False) & (frame["P6_gap_qc_pass"] == True)]  # noqa: E712
    axis_frame = pd.DataFrame(axis_rows)
    valid = axis_frame.groupby("tail_axis_source")["p7v_valid"].mean().to_dict() if len(axis_frame) else {}
    print("Finished v0.6.7 P6 fork gap derivation evaluation.")
    print(f"P6 current median/mean: {current.get('median_error_mm'):.3f} / {current.get('mean_error_mm'):.3f}")
    print(f"P6 geometric median/mean: {geo.get('median_error_mm'):.3f} / {geo.get('mean_error_mm'):.3f}")
    print(f"P6 large error current/geometric: {current.get('large_error_rate'):.3f} / {geo.get('large_error_rate'):.3f}")
    print(f"Current fail but geometric pass cases: {len(fail_current_geo_pass)}")
    print(f"Geometric fallback recommended cases: {len(fallback_rows)}")
    print("Tail axis comparison:")
    print(f"P6-based axis valid: {valid.get('P6_based_axis', 0):.3f}")
    print(f"P5-midpoint axis valid: {valid.get('P5_midpoint_axis', 0):.3f}")
    print(f"lobe-bisector axis valid: {valid.get('lobe_bisector_axis', 0):.3f}")
    use_default = bool(geo.get("median_error_mm", np.inf) < current.get("median_error_mm", np.inf))
    print("Recommendation:")
    print(f"Use P6 gap geometry as default preannotation P6: {'yes' if use_default else 'no'}")
    print("Use P6 only as QC/landmark, not tail axis: yes")
    print("Update P7V tail axis source: no")


if __name__ == "__main__":
    main()
