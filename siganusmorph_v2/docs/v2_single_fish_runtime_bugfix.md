# V2.0 SingleFishPage Runtime Bugfix

Date: 2026-06-21

## Issue

The FastAPI backend was running at:

```text
http://127.0.0.1:8000
```

Manual browser console health checks against `/api/health` passed on port 8000, but clicking "开始测量" in the React/Vite user app attempted:

```text
POST http://127.0.0.1:8765/api/upload
```

and failed with `ERR_CONNECTION_REFUSED`.

## Root Cause

The frontend API client still used the old default API base URL:

```ts
http://127.0.0.1:8765
```

Therefore health checks could pass on 8000 while upload/calibrate/measure calls from the user interface still went to 8765.

## Fix

The V2.0 user-facing API base URL is now centralized in:

```text
siganusmorph_v2/frontend/src/api/client.ts
```

Current value:

```ts
export const API_BASE_URL = "http://127.0.0.1:8000";
export const API_BASE = API_BASE_URL;
```

All user-facing API calls use this base URL:

- `/api/health`
- `/api/settings`
- `/api/upload`
- `/api/calibrate`
- `/api/measure`
- `/api/update-points`
- `/api/export`
- `/api/exports`
- `/api/session/{session_id}`

The dev launchers, runtime checks, Tauri notes and build/run documentation were also updated from port 8765 to port 8000.

## Expected Behavior After Fix

When the backend is running at `http://127.0.0.1:8000` and the frontend is opened at `http://127.0.0.1:5173`:

1. JPG upload should reach `POST /api/upload`.
2. Calibration should reach `POST /api/calibrate`.
3. If calibration succeeds, measurement should reach `POST /api/measure`.
4. If calibration fails, the UI should show calibration failure or review-needed messaging, not `Failed to fetch`.

## Runtime Verification

Browser verification was rerun with a temporary JPG converted from `real_001.png` and written under `results/v2_user_outputs/bugfix_runtime/`.

Result:

- `GET /api/health` on `http://127.0.0.1:8000`: passed.
- React/Vite user app on `http://127.0.0.1:5173`: passed.
- JPG upload from SingleFishPage reached `POST http://127.0.0.1:8000/api/upload`: passed.
- Calibration reached `POST http://127.0.0.1:8000/api/calibrate`: passed.
- Measurement reached `POST http://127.0.0.1:8000/api/measure`: passed.
- `Failed to fetch` / `ERR_CONNECTION_REFUSED`: not observed.

Evidence:

```text
results/v2_user_outputs/bugfix_runtime/v2_port_8000_bugfix_runtime_report.json
results/v2_user_outputs/bugfix_runtime/browser_bugfix_single_fish_after_measure.png
```

Tauri rerun note:

- A controlled Tauri relaunch was attempted after the port fix.
- The relaunch was blocked by an existing `siganusmorph_v2.exe` / Cargo process holding the shared Cargo target output (`os error 32`).
- This Tauri rerun blocker is an environment/process-lock issue, not an API port issue.
- Tauri uses the same React API client, so the browser runtime verification confirms the corrected API base URL for the shared user client.

Evidence:

```text
results/v2_user_outputs/bugfix_runtime/v2_tauri_port_8000_bugfix_runtime_report.json
results/v2_user_outputs/bugfix_runtime/tauri_bugfix_stderr_retry.log
```

## Protection Boundary

This bugfix only changes V2.0 frontend/runtime configuration and V2.0 documentation. It does not modify V1.0 release files, `corrected_keypoints`, historical batch/final analysis outputs, or model weights.

## 2026-06-21 Measurement Coordinate Hardening Follow-up

The SingleFishPage measurement canvas was hardened after real-use feedback:

- visible landmark radius was increased while keeping a larger invisible drag hit radius;
- points now use high-contrast fills, white halos and dark outlines;
- P7V is rendered as a non-draggable derived marker;
- hover/selected labels, crosshair and magnifier remain available;
- image/display mapping is explicit via `imageToDisplay()` and `displayToImage()`;
- measurements continue to use warped image coordinates, not display coordinates.

Coordinate system notes and invariance results are documented in:

```text
siganusmorph_v2/docs/v2_measurement_coordinate_system.md
results/v2_user_outputs/measurement_coordinate_invariance_report.json
```
