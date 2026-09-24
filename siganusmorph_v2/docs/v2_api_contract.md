# V2.0 API Contract

Base URL in development:

```text
http://127.0.0.1:8000
```

All measurement coordinates returned by the API are in `warped_image` coordinate space unless explicitly stated otherwise.

Session ids are canonical UUID strings. Invalid session ids are rejected before any filesystem lookup.

## GET /api/health

Returns backend status.

Response:

```json
{
  "status": "ok",
  "version": "2.0.0",
  "app": "SiganusMorph Local V2.0"
}
```

## GET /api/settings

Returns user-safe V2.0 settings, version information, output directories and model path availability. All paths are project-relative.

Response:

```json
{
  "version": "2.0.0",
  "app": "SiganusMorph Local V2.0",
  "core_engine": "siganusmorph/",
  "backend_output_root": "results/v2_user_outputs",
  "user_output_root": "results/v2_user_outputs/",
  "admin_output_root": "results/v2_admin_outputs",
  "model_paths": [
    {
      "name": "heatmap_unet_v0.1_candidate",
      "path": "models/siganusmorph_heatmap_unet_v0.1/preannotation_candidate.pt",
      "exists": true
    }
  ],
  "notes": [
    "V2.0 user outputs are isolated from V1.0 historical results."
  ]
}
```

## POST /api/upload

Uploads an image and creates a session.

Uploaded filenames are sanitized before they are written to disk. Caller-provided path fragments are ignored, and stored files remain inside the V2 session directory.

Request:

```text
multipart/form-data
file
specimen_id
weight_g
```

Response:

```json
{
  "version": "2.0.0",
  "session_id": "uuid",
  "image_name": "sample.png",
  "raw_image_url": "/api/assets/{session_id}/raw.png",
  "status": "uploaded"
}
```

## GET /api/session/{session_id}

Returns a public, user-safe session payload for restoring an existing V2 session in the React app.

This endpoint does not expose internal filesystem paths such as original upload paths or warped-image paths. Image references are returned as API asset URLs, and exported files are returned as project-relative paths.

Response:

```json
{
  "version": "2.0.0",
  "session_id": "uuid",
  "image_name": "sample.png",
  "specimen_id": "fish_001",
  "weight_g": 155.33,
  "status": "measured",
  "coordinate_space": "warped_image",
  "raw_image_url": "/api/assets/{session_id}/raw.png",
  "raw_image_width": 1600,
  "raw_image_height": 1000,
  "calibration": {
    "status": "success",
    "warped_image_url": "/api/assets/{session_id}/warped.png",
    "mm_per_pixel": 0.084,
    "mm_per_pixel_source": "charuco_or_aruco_board"
  },
  "measurement": {
    "coordinate_space": "warped_image",
    "formal_points": {},
    "derived_points": {},
    "measurements": {},
    "qc": {}
  },
  "exports": {
    "csv": "results/v2_user_outputs/exports/{session_id}/measurement.csv"
  }
}
```

## POST /api/calibrate

Runs automatic calibration. For manual calibration or smoke testing, `manual_corners` can be supplied as four raw-image points in this order: top-left, top-right, bottom-right, bottom-left.

Request:

```json
{
  "session_id": "uuid",
  "manual_corners": [[100.0, 100.0], [1500.0, 100.0], [1500.0, 900.0], [100.0, 900.0]]
}
```

Response:

```json
{
  "version": "2.0.0",
  "session_id": "uuid",
  "status": "success",
  "calibration_mode": "automatic_marker_detection",
  "warped_image_url": "/api/assets/{session_id}/warped.png",
  "raw_marker_overlay_url": "/api/assets/{session_id}/raw_marker_overlay.png",
  "warped_image_width": 1600,
  "warped_image_height": 1000,
  "mm_per_pixel": 0.084,
  "mm_per_pixel_source": "charuco_or_aruco_board",
  "marker_count": 12,
  "message": "calibration success"
}
```

If calibration fails, `status` is `failed`, `warped_image_url` is null, and measurement must remain blocked.

If `manual_corners` is supplied, the backend uses the existing V1.0 formal manual-corner calibration routine and returns `calibration_mode = manual_corners` and `mm_per_pixel_source = manual_board_corners` on success. This path is intended for manual correction and controlled API verification; it must not be treated as automatic marker detection.

## POST /api/measure

Runs the recommended measurement workflow on the warped image.

Request:

```json
{
  "session_id": "uuid",
  "preannotation_mode": "recommended",
  "measurement_axis_mode": "auto_qc_gated"
}
```

Response includes:

- `formal_points`
- `derived_points`
- `measurements`
- `qc`
- `coordinate_space = warped_image`

## POST /api/update-points

Applies user-edited formal points and recalculates measurements.

Request:

```json
{
  "session_id": "uuid",
  "formal_points": {
    "P1_snout_tip": [120.0, 300.0]
  },
  "measurement_overrides": {
    "source": "react_measurement_canvas"
  }
}
```

Response returns recalculated formal points, derived points, measurements and QC.

## POST /api/export

Exports the current session.

Export is allowed only after successful calibration and measurement. A calibrated-but-unmeasured session returns an error instead of creating empty result files.

Request:

```json
{
  "session_id": "uuid",
  "formats": ["csv", "xlsx", "json", "preview_png"],
  "save_confirmed_session": true
}
```

Response:

```json
{
  "version": "2.0.0",
  "session_id": "uuid",
  "files": {
    "csv": "results/v2_user_outputs/exports/{session_id}/measurement.csv",
    "xlsx": "results/v2_user_outputs/exports/{session_id}/measurement.xlsx",
    "json": "results/v2_user_outputs/exports/{session_id}/formal_points.json",
    "preview_png": "results/v2_user_outputs/exports/{session_id}/preview.png"
  },
  "saved_to": "results/v2_user_outputs/exports/{session_id}"
}
```

## GET /api/exports

Lists V2 user export folders under `results/v2_user_outputs/exports/`. This endpoint is read-only and returns project-relative paths.

Response:

```json
{
  "version": "2.0.0",
  "exports": [
    {
      "version": "2.0.0",
      "session_id": "uuid",
      "image_name": "sample.png",
      "specimen_id": "fish_001",
      "weight_g": 155.33,
      "saved_to": "results/v2_user_outputs/exports/{session_id}",
      "files": {
        "csv": "results/v2_user_outputs/exports/{session_id}/measurement.csv",
        "xlsx": "results/v2_user_outputs/exports/{session_id}/measurement.xlsx",
        "json": "results/v2_user_outputs/exports/{session_id}/formal_points.json",
        "preview_png": "results/v2_user_outputs/exports/{session_id}/preview.png"
      },
      "exported_at": 1781852653.81758
    }
  ]
}
```
