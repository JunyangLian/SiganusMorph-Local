# V2.0 Measurement Coordinate System

Date: 2026-06-21

## Coordinate Flow

V2.0 user measurements follow one coordinate path:

```text
raw image
-> calibration / homography
-> canonical warped image
-> formal measurement points in warped image coordinates
-> SVG display coordinates
-> returned warped image coordinates
-> backend measurements in mm
```

The backend measurement values are computed from warped image coordinates and `mm_per_pixel` from successful calibration. Frontend display coordinates are not used for measurement.

## Calibration Gate

Measurement is allowed only when calibration succeeds.

The backend rejects:

- missing calibration;
- failed calibration;
- missing warped image;
- missing `mm_per_pixel`;
- fallback/default `mm_per_pixel`.

## Canonical Board Warp

The current V2.0 calibration path uses canonical board warp:

- automatic ArUco/multi-marker calibration maps the board to a fixed output size from board physical dimensions and `output_px_per_mm`;
- manual four-corner calibration also maps to the same board-coordinate frame;
- successful `real_001` and `real_002` checks produced `4200 x 2970 px` warped images with `0.1 mm/px`.

The API exposes:

- `calibration_status`
- `detected_marker_count`
- `homography_raw_to_warped`
- `warped_width_px`
- `warped_height_px`
- `mm_per_pixel`
- `board_physical_width_mm`
- `board_physical_height_mm`
- `canonical_warp_used`
- `measurement_roi_bounds_warped`
- `fish_bbox_warped`
- `fish_margin_to_roi_px`
- `coordinate_system_version`

Current coordinate system version:

```text
v2.0_warped_image_canonical_board_v1
```

## Frontend Display Mapping

The React measurement canvas uses an SVG `viewBox` matching the full warped image. It does not use `object-fit: cover`.

The editor now has explicit mapping helpers:

- `imageToDisplay()`
- `displayToImage()`

They use SVG CTM transforms so browser zoom, CSS size, and window size do not change measurement coordinates. Points returned to Python are warped image coordinates.

## Landmark Visibility Hardening

The formal measurement canvas now uses larger visible points:

- normal visible radius: 7 px;
- depth endpoint visible radius: 8 px;
- hover/selected visible radius: 11 px;
- invisible drag hit radius: 24 px.

Point rendering also includes:

- white halo;
- dark outline;
- high-contrast fill colors;
- hover/selected labels;
- optional all-label display;
- crosshair during drag;
- magnifier;
- P7V as a non-draggable derived diamond/outline marker.

## Coordinate Invariance Check

Script:

```powershell
python siganusmorph_v2/measurement_coordinate_invariance_check.py
```

Output:

```text
results/v2_user_outputs/measurement_coordinate_invariance_report.json
```

Summary:

- same-image display resize invariance: passed;
- canonical warped image fields: stable at `4200 x 2970 px`, `0.1 mm/px`;
- padded border variant: passed, max difference `0.000 mm`;
- scaled 0.75 variant: passed, max difference `0.194 mm`;
- scaled 1.25 variant: passed, max difference `0.716 mm`;
- edge crop 10 px variant: borderline failed, max difference `1.002 mm` on `FL_mm`.

Interpretation:

The crop failure is a small preannotation/point-estimation sensitivity around P6/FL, not evidence of frontend display coordinates being mixed into backend measurement. The canonical warped coordinate frame and display mapping checks passed.

## Real Fish QA

Two JPG real-fish checks were run through upload, calibration and measurement:

| sample | weight_g | status | canonical_warp_used | warped_size | mm_per_pixel |
| --- | ---: | --- | --- | --- | ---: |
| real_001_65_44g | 65.44 | measured | yes | 4200 x 2970 | 0.1 |
| real_002_69_92g | 69.92 | measured | yes | 4200 x 2970 | 0.1 |

Both checks avoided `Failed to fetch`; calibration failure would block measurement instead of using a fallback scale.
