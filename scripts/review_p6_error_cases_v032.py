"""Create a diagnostic review for v0.3.2 P6 large-error cases.

This is a read-only analysis over evaluation outputs. It does not alter
corrected labels, model weights, or tail-geometry rules.
"""

from __future__ import annotations

import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "results" / "model_eval" / "v0.3.2_tail_geometry_compare_30"
SUMMARY_CSV = EVAL_DIR / "preannotation_vs_manual_summary.csv"
DEBUG_DIR = EVAL_DIR / "debug_json"
VIS_DIR = EVAL_DIR / "visual_comparisons"
P6_CASE_DIR = EVAL_DIR / "P6_error_cases"
REVIEW_CSV = EVAL_DIR / "P6_error_case_review.csv"
TYPE_SUMMARY_CSV = EVAL_DIR / "P6_error_type_summary.csv"

P6 = "P6_caudal_fork_midpoint"
P5 = "P5_caudal_base_midpoint"
P7U = "P7U_caudal_fin_upper_tip"
P7L = "P7L_caudal_fin_lower_tip"


def read_debug(image_name: str) -> dict[str, Any]:
    path = DEBUG_DIR / f"{Path(image_name).stem}_debug.json"
    return json.loads(path.read_text(encoding="utf-8"))


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def xy(value: Any) -> tuple[float, float] | None:
    if isinstance(value, Mapping):
        if "x" in value and "y" in value:
            return float(value["x"]), float(value["y"])
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return None


def dist(a: Any, b: Any) -> float | None:
    pa = xy(a)
    pb = xy(b)
    if pa is None or pb is None:
        return None
    return math.hypot(pa[0] - pb[0], pa[1] - pb[1])


def row_for_key(summary: pd.DataFrame, image_name: str, key: str) -> pd.Series | None:
    rows = summary[(summary["image_name"] == image_name) & (summary["keypoint_name"] == key)]
    if rows.empty:
        return None
    return rows.iloc[0]


def classify_case(
    *,
    debug: Mapping[str, Any],
    p6_error_mm: float,
    p7u_error_mm: float,
    p7l_error_mm: float,
) -> tuple[str, str, str]:
    meta = debug.get("preannotation_metadata", {}) if isinstance(debug.get("preannotation_metadata"), Mapping) else {}
    tail_qc = meta.get("tail_geometry_qc", {}) if isinstance(meta.get("tail_geometry_qc"), Mapping) else {}
    manual = debug.get("manual_corrected_keypoints", {}) if isinstance(debug.get("manual_corrected_keypoints"), Mapping) else {}
    auto = debug.get("auto_preannotation_keypoints", {}) if isinstance(debug.get("auto_preannotation_keypoints"), Mapping) else {}
    p6_auto = xy(auto.get(P6))
    p6_manual = xy(manual.get(P6))
    p7u_auto = xy(auto.get(P7U))
    p7l_auto = xy(auto.get(P7L))
    p5_auto = xy(auto.get(P5))
    types: list[str] = []
    fixes: list[str] = []
    notes: list[str] = []

    p6_quality = str(tail_qc.get("P6_mask_quality", ""))
    p6_method = str(tail_qc.get("P6_candidate_method", ""))
    p6_review = str(tail_qc.get("P6_mask_review_reason", "") or tail_qc.get("P6_review_reason", ""))
    tail_axis_source = str(tail_qc.get("tail_axis_source", ""))
    p5_qc_pass = as_bool(tail_qc.get("P5_qc_pass", True))
    p7u_quality = str(tail_qc.get("P7U_mask_quality", ""))
    candidate_count = int(float(tail_qc.get("P6_candidate_count", 0) or 0))

    if not p5_qc_pass:
        types.append("P5_shifted")
        fixes.append("Improve P5 transition suggestion or require manual P5 confirmation before P6 rule.")
    if "fallback" in tail_axis_source:
        types.append("tail_axis_shifted")
        fixes.append("When using fallback tail axis, evaluate fork candidates in both C4->P5 and local lobe direction frames.")
    if p7u_error_mm > 10 or p7l_error_mm > 10 or p7u_quality == "failed":
        types.append("P7U_or_P7L_shifted")
        fixes.append("Stabilize P7U/P7L before fork notch selection; reject fork candidates if lobe tips disagree.")

    if p6_auto and p6_manual and p7u_auto and p7l_auto:
        y_min = min(p7u_auto[1], p7l_auto[1])
        y_max = max(p7u_auto[1], p7l_auto[1])
        gap_height = max(1.0, y_max - y_min)
        if p6_auto[1] < y_min + 0.20 * gap_height or p6_auto[1] > y_max - 0.20 * gap_height:
            types.append("selected_outer_edge_instead_of_notch")
            fixes.append("Add a stronger normal-axis midpoint penalty; reject candidates too close to upper/lower lobe outer margins.")
        if p6_auto[1] < p6_manual[1] - 60:
            types.append("selected_outer_edge_instead_of_notch")
            fixes.append("Prefer the deepest interior gap center over upper contour points when vertical gap is visible.")
        if abs(p6_auto[0] - p6_manual[0]) < 35 and abs(p6_auto[1] - p6_manual[1]) > 70:
            types.append("tail_mask_gap_filled")
            fixes.append("Inspect whether mask close/fill merges the fork gap; reduce close kernel or use raw mask for tail fork detection.")
        if p5_auto and p6_auto[0] - p5_auto[0] < 0:
            types.append("wrong_contour_arc_selected")
            fixes.append("Constrain fork projection to be right of P5 and below lobe-tip projection.")

    if candidate_count < 3 or "failed" in p6_quality or "failed" in p6_review:
        types.append("fork_notch_unclear")
        fixes.append("Use vertical gap fallback and mark low-confidence fork suggestions for manual review.")
    if "gap" in p6_method and p6_error_mm > 10:
        types.append("tail_mask_gap_filled")
        fixes.append("Use unclosed fish mask or local blue-background subtraction in the fork region.")

    if not types:
        if p6_error_mm > 18:
            types.append("manual_label_uncertain")
            fixes.append("Open comparison image and verify whether manual P6 is at the anatomical fork or deeper notch.")
        else:
            types.append("compressed_tail_fin")
            fixes.append("Keep mask fork rule but flag compressed/closed tails for manual confirmation.")

    notes.append(f"P6_quality={p6_quality}; method={p6_method}; candidates={candidate_count}; review={p6_review}")
    notes.append(f"tail_axis_source={tail_axis_source}; P7U_error={p7u_error_mm:.2f}; P7L_error={p7l_error_mm:.2f}")
    return ";".join(dict.fromkeys(types)), " | ".join(dict.fromkeys(fixes)), " | ".join(notes)


def suggested_rule_change_for_type(failure_type: str) -> str:
    mapping = {
        "fork_notch_unclear": "Keep P6 as needs-review; add a confidence score from candidate count and fork gap visibility.",
        "tail_mask_gap_filled": "Run P6 fork detection on a less-closed/raw mask or reduce morphological close near the tail fork.",
        "wrong_contour_arc_selected": "Score both contour arcs using gap depth and normal midpoint, not only mean tail-axis projection.",
        "P5_shifted": "Refine P5 transition detection before using P5 as fork reference.",
        "tail_axis_shifted": "Try both P5->P6 candidate and C4->P5 axes, then choose the candidate closest to the lobe gap center.",
        "P7U_or_P7L_shifted": "Stabilize lobe tip candidates before estimating fork point.",
        "selected_outer_edge_instead_of_notch": "Increase midpoint/gap penalty to reject outer-lobe contour points.",
        "compressed_tail_fin": "Flag compressed tails for manual confirmation; geometry may be underdetermined.",
        "reflection_or_mask_noise": "Clean local mask with smaller kernels and ignore isolated bright/noisy components.",
        "manual_label_uncertain": "Review manual P6 definition consistency before changing the rule.",
    }
    return mapping.get(failure_type, "Review visual comparisons before changing the rule.")


def main() -> None:
    summary = pd.read_csv(SUMMARY_CSV)
    p6_rows = summary[
        (summary["keypoint_name"] == P6)
        & ((summary["is_large_error"].astype(str).str.lower() == "true") | (summary["error_mm"].astype(float) > 10.0))
    ].copy()
    if p6_rows.empty:
        print("No P6 large-error cases found.")
        pd.DataFrame().to_csv(REVIEW_CSV, index=False, encoding="utf-8-sig")
        pd.DataFrame().to_csv(TYPE_SUMMARY_CSV, index=False, encoding="utf-8-sig")
        return

    P6_CASE_DIR.mkdir(parents=True, exist_ok=True)
    review_rows: list[dict[str, Any]] = []
    type_to_images: dict[str, list[str]] = defaultdict(list)
    type_counts: Counter[str] = Counter()

    for _, p6 in p6_rows.iterrows():
        image_name = str(p6["image_name"])
        debug = read_debug(image_name)
        meta = debug.get("preannotation_metadata", {}) if isinstance(debug.get("preannotation_metadata"), Mapping) else {}
        tail_qc = meta.get("tail_geometry_qc", {}) if isinstance(meta.get("tail_geometry_qc"), Mapping) else {}
        p7u = row_for_key(summary, image_name, P7U)
        p7l = row_for_key(summary, image_name, P7L)
        p7u_error = float(p7u["error_mm"]) if p7u is not None else float("nan")
        p7l_error = float(p7l["error_mm"]) if p7l is not None else float("nan")
        failure_type, suggested_fix, notes = classify_case(
            debug=debug,
            p6_error_mm=float(p6["error_mm"]),
            p7u_error_mm=p7u_error,
            p7l_error_mm=p7l_error,
        )
        for failure in failure_type.split(";"):
            if failure:
                type_counts[failure] += 1
                type_to_images[failure].append(image_name)

        src = VIS_DIR / f"{Path(image_name).stem}_compare.png"
        if src.exists():
            shutil.copy2(src, P6_CASE_DIR / src.name)

        review_rows.append(
            {
                "image_name": image_name,
                "specimen_id": p6.get("specimen_id", ""),
                "split": p6.get("split", ""),
                "P6_error_mm": float(p6["error_mm"]),
                "P6_auto_x": p6.get("auto_x", ""),
                "P6_auto_y": p6.get("auto_y", ""),
                "P6_manual_x": p6.get("manual_x", ""),
                "P6_manual_y": p6.get("manual_y", ""),
                "P6_source": p6.get("auto_point_source", tail_qc.get("P6_source", "")),
                "P6_mask_quality": tail_qc.get("P6_mask_quality", ""),
                "P5_qc_pass": tail_qc.get("P5_qc_pass", ""),
                "P7U_error_mm": p7u_error,
                "P7L_error_mm": p7l_error,
                "tail_axis_source": tail_qc.get("tail_axis_source", ""),
                "p7v_valid": tail_qc.get("p7v_valid", meta.get("p7v_valid", "")),
                "possible_failure_type": failure_type,
                "suggested_fix": suggested_fix,
                "notes": notes,
            }
        )

    review_df = pd.DataFrame(review_rows).sort_values("P6_error_mm", ascending=False)
    review_df.to_csv(REVIEW_CSV, index=False, encoding="utf-8-sig")

    type_rows = []
    for failure, count in type_counts.most_common():
        examples = type_to_images[failure][:8]
        type_rows.append(
            {
                "failure_type": failure,
                "count": count,
                "example_images": ";".join(examples),
                "suggested_rule_change": suggested_rule_change_for_type(failure),
            }
        )
    pd.DataFrame(type_rows).to_csv(TYPE_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    print(f"Reviewed {len(review_rows)} P6 large-error cases.")
    print(f"Review CSV: {REVIEW_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Type summary: {TYPE_SUMMARY_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Copied visual comparisons to: {P6_CASE_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
