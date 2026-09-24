# V2.0 User Manual Draft

## Home

The Home page introduces the system and provides entry points for single-fish measurement, batch measurement and export.

Current desktop status:

- Tauri desktop shell can open the user app.
- Tauri release executable build is verified.
- The window title is `蓝子鱼形态测量系统 V2.0`.
- The Home page and left navigation render correctly.
- The visual style follows the Apple-inspired V2.0 design tokens.

## Single Fish Measurement

Current status: the backend HTTP loop is verified, and the browser-level React/FastAPI visual workflow is verified. Tauri-window screenshots may still be collected for product signoff.

1. Upload one lateral fish image.
2. Enter `specimen_id`.
3. Optionally enter `weight_g`.
4. Click automatic measurement.
5. The backend uploads, calibrates and measures the image.
6. If calibration fails, measurement is blocked.
7. Review the warped image in the workbench.
8. Drag points if needed.
9. Click Apply Changes.
10. Save and export CSV, Excel, JSON and preview PNG.

Additional product-signoff validation still recommended:

- open the Single Fish page inside Tauri for screenshots;
- upload a real calibration-board fish image;
- confirm real-image calibration and warped image alignment;
- confirm export writes only under `results/v2_user_outputs/`.

### Restore an Existing V2 Session

The Single Fish page can restore a previous V2 session by `session_id`.

1. Paste the V2 `session_id` into the optional restore field.
2. Click Restore Session.
3. The page reloads the public session payload from `/api/session/{session_id}`.
4. If calibration and measurement already exist, the warped image, formal points and measurement table are restored.

Session restore reads only V2 session data under `results/v2_user_outputs/`. It does not read or modify V1.0 historical outputs or `corrected_keypoints`.

## Batch Measurement

1. Select multiple image files.
2. Click batch measurement.
3. The frontend processes the queue one image at a time through upload, calibration, measurement and export.
4. Each image receives a status badge:
   - waiting
   - uploading
   - calibrating
   - measuring
   - exported
   - calibration_failed
   - needs_review
5. Calibration failure blocks measurement for that image and marks it for review.
6. Exported sessions are added to the local export history.

## Export

Exports are written to `results/v2_user_outputs/`. The Export page reads `/api/exports` to show V2 user export folders from the backend. If the backend is unavailable, it falls back to browser-local export history from Single Fish and Batch workflows. It does not read or modify V1.0 historical outputs.

## Settings

The Settings page reads backend configuration from `/api/settings` and displays:

- backend version;
- Python core engine location;
- V2 user and admin output directories;
- candidate model path availability.

Only project-relative paths are shown. Ordinary users cannot edit model paths from this page.

## Data Protection

V2.0 user exports do not modify V1.0 outputs, historical batch measurements, final analysis files or `corrected_keypoints`.
