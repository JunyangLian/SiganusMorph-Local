# Measurement Axis Modes

SiganusMorph keeps keypoint annotation and curve-based measurement as two separate layers.

## Modes

`model_axis` is the current default and compatibility mode. It uses the confirmed or current C1-C4 axis keypoints together with P1, P4, P5, and the derived P7V point.

`body_midline_axis` is an experimental smoother measurement axis. C1-C3 are derived from a de-finned body-contour midline, C4 is derived as the midpoint of P4-P5, and the tail still uses the P4-P5-P7V geometry. These geometric C points are stored only in `measurement_axes`; they do not overwrite `corrected_keypoints`.

`auto_qc_gated` selects `body_midline_axis` only when its QC passes and the two axes agree. If the body-midline axis fails QC, or if TL/SL/curvature differences exceed review thresholds, the software falls back to `model_axis`. High-curvature or invalid-P7V cases are marked for manual review.

## Output Fields

Both axes are exported:

- `TL_curve_model_axis_mm`, `SL_curve_model_axis_mm`, `curvature_index_model_axis`
- `TL_curve_body_midline_axis_mm`, `SL_curve_body_midline_axis_mm`, `curvature_index_body_midline_axis`
- `TL_curve_selected_mm`, `SL_curve_selected_mm`, `curvature_index_selected`
- `selected_measurement_axis`
- `dual_axis_disagreement`
- `measurement_axis_review_reason`

Legacy `TL_curve_mm`, `SL_curve_mm`, and `curvature_index` are retained for compatibility. For new saves they follow the selected measurement axis when an explicit selected axis is used.

## Review Rules

Manual review is required when:

- P7V is invalid.
- Curvature is high.
- `TL_curve_axis_diff_mm` or `SL_curve_axis_diff_mm` exceeds 5 mm.
- `curvature_index_axis_diff` exceeds 0.03.
- Body-midline QC fails and the user selected `auto_qc_gated`.

## Why C1-C4 Are Not Replaced

C1-C4 are still saved as human-reviewable auxiliary keypoints because they are useful for inspection and backward compatibility. The body-contour midline is a measurement-axis candidate, not a replacement for the annotation schema or training labels.

Current default remains `model_axis`. `body_midline_axis` is experimental / QC-gated.
