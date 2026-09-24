# SiganusMorph Local V2.0 Build Report

Generated: 2026-06-19

## 2026-06-21 Runtime Bugfix: Frontend API Port

Root cause found during real frontend testing: the FastAPI backend was running on `http://127.0.0.1:8000`, but the React user client still defaulted to `http://127.0.0.1:8765` for `/api/upload`.

Fix:

- `siganusmorph_v2/frontend/src/api/client.ts` now centralizes the user-facing API base URL as `http://127.0.0.1:8000`.
- V2 launchers, runtime checks and run documentation were aligned to port `8000`.
- Browser and Tauri user flows should use the same backend origin for health, upload, calibrate, measure, update-points and export.

Safety boundary: no V1.0 release files, historical batch/final analysis outputs, corrected keypoints, model weights or measurement algorithms were changed.

Verification:

- Frontend production build was regenerated successfully after the API base URL change.
- Runtime search found no active frontend source or built `dist` reference to `127.0.0.1:8765`.
- Browser SingleFishPage JPG workflow passed against `http://127.0.0.1:8000`.
- Observed browser requests: `POST /api/upload`, `POST /api/calibrate`, and `POST /api/measure`, all on port `8000`.
- `Failed to fetch` / `ERR_CONNECTION_REFUSED` was not observed during the browser JPG workflow.
- Tauri dev shell relaunch for this bugfix was attempted, but the rerun was blocked by an existing `siganusmorph_v2.exe` / Cargo process holding `D:/1_postgraduate/tauri_target_siganusmorph_v2` (`os error 32`). This is a local target-directory file lock, not an API port issue. The Tauri shell uses the same React API client verified in the browser runtime.

Evidence:

```text
results/v2_user_outputs/bugfix_runtime/v2_port_8000_bugfix_runtime_report.json
results/v2_user_outputs/bugfix_runtime/v2_tauri_port_8000_bugfix_runtime_report.json
siganusmorph_v2/docs/v2_single_fish_runtime_bugfix.md
```

## 2026-06-21 Measurement Coordinate Hardening

Real-use feedback found that formal measurement points were too small and that image edge/crop differences needed coordinate-system verification.

Changes:

- `MeasurementCanvas` visible radii increased to 7-8 px, hover/selected radius to 11 px, with a 24 px invisible hit radius.
- Point rendering now uses high-contrast colors, white halo, dark outline and selected/hover labels.
- P7V is rendered as a non-draggable derived marker.
- SVG mapping is explicit through `imageToDisplay()` and `displayToImage()` so browser display size is separate from image coordinates.
- Calibration/session API responses now expose coordinate audit metadata: `canonical_warp_used`, `homography_raw_to_warped`, `warped_width_px`, `warped_height_px`, `board_physical_width_mm`, `board_physical_height_mm`, ROI/bbox fields and `coordinate_system_version`.

Verification:

- Frontend build: passed.
- Browser visual workflow: 15/15 passed.
- Display resize coordinate roundtrip: passed.
- Canonical warped frame: confirmed at `4200 x 2970 px`, `0.1 mm/px` for successful real samples.
- Invariance variants:
  - padded border 80 px: max difference `0.000 mm`;
  - scaled 0.75: max difference `0.194 mm`;
  - scaled 1.25: max difference `0.716 mm`;
  - edge crop 10 px: borderline failed on `FL_mm` with `1.002 mm` difference.
- Real fish QA: `65.44 g` and `69.92 g` JPG checks both measured successfully.

Evidence:

```text
results/v2_user_outputs/measurement_coordinate_invariance_report.json
results/v2_user_outputs/v2_frontend_visual_workflow_check.json
siganusmorph_v2/docs/v2_measurement_coordinate_system.md
```

## 1. Directory Structure

```text
siganusmorph_v2/
+-- backend/
|   +-- app.py
|   +-- run_backend.py
|   +-- routers/
|   +-- schemas/
|   +-- services/
|   +-- smoke_test_api.py
|   +-- live_health_check.py
|   +-- live_api_flow_check.py
+-- frontend/
|   +-- index.html
|   +-- package.json
|   +-- tsconfig.json
|   +-- vite.config.ts
|   +-- src/
+-- desktop/
|   +-- tauri/
+-- admin_streamlit/
|   +-- app.py
+-- docs/
```

## 2. Completed Modules

### FastAPI Backend

Implemented endpoints:

- `GET /api/health`
- `GET /api/settings`
- `GET /api/session/{session_id}`
- `GET /api/exports`
- `POST /api/upload`
- `POST /api/calibrate`
- `POST /api/measure`
- `POST /api/update-points`
- `POST /api/export`

Backend behavior:

- Upload creates an isolated V2 session.
- Uploaded filenames are sanitized before writing to the session directory.
- Session ids are validated as canonical UUID strings before filesystem lookup.
- Calibration calls the existing `siganusmorph/` core.
- `/api/calibrate` supports automatic marker detection and optional `manual_corners`.
- Measurement is blocked unless calibration status is `success`.
- Missing `mm_per_pixel` is not accepted.
- `mm_per_pixel_source = fallback_default` is blocked.
- Measurement runs on `warped_image` only.
- Export is blocked after calibration but before measurement.
- Export files include `version = 2.0.0`.
- Export API responses return project-relative paths.
- Settings endpoint reports version, output directories and model availability using project-relative paths only.
- Exports listing endpoint reads V2 user export folders and returns project-relative paths only.
- Public session lookup restores a V2 session without exposing internal filesystem paths.

Backend output directory:

```text
results/v2_user_outputs/
```

### React/Vite/TypeScript Frontend

Implemented scaffold:

- `HomePage`
- `SingleFishPage`
- `BatchPage`
- `ExportPage`
- `SettingsPage`
- `MeasurementCanvas`
- API client for all required backend endpoints.
- SingleFishPage can restore an existing V2 session by `session_id`.
- BatchPage loops through selected files and calls upload, calibrate, measure and export for each image.
- ExportPage reads backend V2 exports and falls back to browser-local export history if needed.
- SettingsPage reads `/api/settings` for model availability, output directories and version information.
- API client normalizes backend errors so ordinary users do not see Python tracebacks.
- Vite environment type declaration is included for `import.meta.env` support.

Apple-inspired design tokens are implemented:

- background `#f5f5f7`
- card `#ffffff`
- text `#1d1d1f`
- secondary `#707070`
- primary CTA `#0071e3`
- link `#0066cc`
- input `#e8e8ed`
- card radius `28px`
- button radius `36px`

`MeasurementCanvas` currently supports:

- warped image display;
- draggable formal points;
- P7V as a derived display point;
- small visible points with larger hit radius;
- hover labels;
- crosshair during drag;
- local magnifier;
- keyboard arrow nudging;
- local temporary measurement preview;
- Apply Changes callback to `/api/update-points`;
- separate display coordinate and image coordinate mapping.

Current frontend status:

- TypeScript build passes.
- React/Vite runtime HTTP check passes.
- Tauri dev shell loads the React/Vite HomePage.
- Browser-level SingleFishPage and MeasurementCanvas visual workflow passes.

### Tauri Desktop Skeleton

Created:

- `desktop/tauri/package.json`
- `desktop/tauri/src-tauri/Cargo.toml`
- `desktop/tauri/src-tauri/tauri.conf.json`
- `desktop/tauri/src-tauri/src/main.rs`
- `desktop/tauri/README.md`

Current mode:

- Development shell verified.
- Release executable build verified.
- FastAPI backend is started manually.
- Tauri loads the Vite dev server.
- Installer/package generation is not enabled yet because `bundle.active = false`.

Tauri environment:

- Rust/Cargo home: `D:/1_postgraduate/.cargo`
- Rustup home: `D:/1_postgraduate/.rustup`
- Rust host triple: `x86_64-pc-windows-gnu`
- MinGW/GCC path: `D:/1Bio_Soft/mingw64/bin`
- recommended Cargo target directory: `D:/1_postgraduate/tauri_target_siganusmorph_v2`
- Tauri icons were generated with `npx tauri icon`

Diagnostic:

```powershell
python siganusmorph_v2/tauri_runtime_diagnostic.py
```

Current result:

```text
total_checks: 14
passed_checks: 14
failed_checks: 0
tauri_runtime_ready: true
```

Release build:

```powershell
cd siganusmorph_v2/desktop/tauri
npm run tauri:build
```

Current result:

```text
release executable build: passed
executable: D:/1_postgraduate/tauri_target_siganusmorph_v2/release/siganusmorph_v2.exe
installer package: not generated because bundle.active is false
```

Build note:

```text
Tauri's production `frontendDist` must resolve from `src-tauri`, so it is set to `../../../frontend/dist`.
Using a Cargo target directory under the workspace path with a space caused a MinGW windres path parsing failure.
The user-specified target directory `D:/1_postgraduate/tauri_target_siganusmorph_v2` avoids that issue.
```

### V2 Launcher Scripts

Created:

- `scripts/check_runtime_prereqs.ps1`
- `scripts/run_backend.ps1`
- `scripts/run_frontend_dev.ps1`
- `scripts/run_tauri_dev.ps1`
- `scripts/run_admin_console.ps1`
- `scripts/run_v2_readiness.ps1`
- `scripts/README.md`

Launcher audit:

```powershell
python siganusmorph_v2/launcher_audit.py
```

Current result:

```text
total_checks: 19
passed_checks: 19
failed_checks: 0
```

The launchers are convenience scripts only. They do not modify V1.0 release files, `corrected_keypoints`, or historical measurement results.

### NPM Dependency Manifest Audit

Command:

```powershell
python siganusmorph_v2/npm_dependency_audit.py
```

Current result:

```text
total_checks: 6
passed_checks: 6
failed_checks: 0
warning_count: 6
```

The audit confirms:

- V2 frontend and Tauri `package.json` files exist;
- local project npm lifecycle scripts such as `preinstall`, `install`, `postinstall` and `prepare` are absent;
- dependency names match the current V2 allowlist;
- frontend runtime dependencies are present and have passed build/runtime HTTP checks;
- Tauri runtime dependencies are present and Tauri dev shell is verified.

### NPM Install Runtime Attempt

Runtime log:

```text
siganusmorph_v2/docs/v2_npm_install_runtime_log.md
```

Frontend install command:

```powershell
cd siganusmorph_v2/frontend
npm install
```

Current result:

```text
frontend npm install command completed cleanly: no
frontend dependencies available on disk: yes
frontend npm audit completed: no
frontend build: pass
frontend Vite runtime HTTP check: pass
Tauri npm install completed: no
Tauri error: ECONNRESET
```

Frontend `node_modules` and `package-lock.json` are now present, and React/Vite build plus HTTP runtime checks pass. `npm audit` did not complete because the audit endpoint was unreachable. Tauri dependencies are still missing because downloading `@tauri-apps/cli` failed with `ECONNRESET`.

Following the controlled-install rule, the registry was not changed and no automatic major upgrades or workaround installation paths were attempted. A second controlled Tauri `npm install` attempt on 2026-06-20 failed with the same `ECONNRESET`; `desktop/tauri/node_modules` and `desktop/tauri/package-lock.json` remain absent.

### Frontend Build and Runtime

Frontend build:

```powershell
cd siganusmorph_v2/frontend
npm run build
```

Current result:

```text
passed
dist/index.html present
```

Frontend runtime HTTP check:

```powershell
python siganusmorph_v2/frontend/runtime_http_check.py
```

Current result:

```text
total_checks: 7
passed_checks: 7
failed_checks: 0
```

Verified:

- Vite dev server serves the root page at `http://127.0.0.1:5173`;
- backend `/api/health` is reachable while the frontend is running;
- HomePage and SingleFishPage source modules transform through Vite;
- Apple-inspired CSS tokens are served;
- user-visible root HTML does not expose local paths or Python tracebacks.

Browser-rendered click/drag interaction remains unverified because the in-app browser control tool was unavailable in this session.

Reports:

```text
results/v2_user_outputs/v2_npm_dependency_audit.json
results/v2_user_outputs/v2_npm_dependency_audit.csv
```

### Streamlit Admin Console

Created:

```text
siganusmorph_v2/admin_streamlit/app.py
```

Admin Console provides an entry/status page for:

- legacy Streamlit backend;
- Review Queue;
- Advanced overlay;
- model evaluation;
- dataset export;
- training dataset planning workflows.

It also shows V2 output boundaries and startup commands while keeping ordinary users directed to the React/Tauri app.

### V2 Documentation

Generated:

- `docs/v2_architecture.md`
- `docs/v2_api_contract.md`
- `docs/v2_frontend_style_guide.md`
- `docs/v2_user_manual_draft.md`
- `docs/v2_admin_manual_draft.md`
- `docs/v2_software_copyright_notes.md`
- `docs/v2_screenshot_checklist.md`
- `docs/v2_build_and_run.md`
- `docs/v2_completion_audit.md`

## 3. Verification Evidence

### Python Compile

Passed:

```powershell
python -m py_compile siganusmorph_v2/backend/app.py siganusmorph_v2/backend/run_backend.py siganusmorph_v2/backend/routers/api.py siganusmorph_v2/backend/schemas/measurement.py siganusmorph_v2/backend/services/measurement_service.py siganusmorph_v2/backend/services/session_store.py siganusmorph_v2/backend/smoke_test_api.py siganusmorph_v2/admin_streamlit/app.py
```

### Backend Health

Verified with FastAPI TestClient:

```json
{
  "status": "ok",
  "version": "2.0.0",
  "app": "SiganusMorph Local V2.0"
}
```

Also verified with a live temporary Uvicorn server:

```text
python siganusmorph_v2/backend/live_health_check.py
passed_checks: 5
failed_checks: 0
```

Report:

```text
results/v2_user_outputs/v2_live_backend_health_check.json
results/v2_user_outputs/v2_live_backend_health_check.csv
```

### Live HTTP API Flow

Verified through a temporary local Uvicorn server:

```text
python siganusmorph_v2/backend/live_api_flow_check.py
passed_checks: 8
failed_checks: 0
```

The live HTTP flow exercises:

```text
upload -> calibrate -> measure -> update-points -> export -> session restore -> export listing
```

Report:

```text
results/v2_user_outputs/v2_live_api_flow_check.json
results/v2_user_outputs/v2_live_api_flow_check.csv
```

### Backend Smoke Test

Command:

```powershell
python siganusmorph_v2/backend/smoke_test_api.py
```

Report:

```text
results/v2_user_outputs/v2_backend_smoke_test_report.json
```

Current result:

```text
total_checks: 11
passed_checks: 11
failed_checks: 0
```

Verified:

- health endpoint returns ok;
- settings endpoint returns V2.0 version, relative output directories and model availability;
- upload endpoint creates V2 sessions;
- invalid session ids are rejected;
- unsafe upload filenames are sanitized and stored inside the V2 session directory;
- automatic calibration failure blocks measurement;
- no fallback `mm_per_pixel` is accepted as success;
- manual-corner calibration succeeds with `mm_per_pixel_source = manual_board_corners`;
- premature export after calibration but before measurement is blocked;
- synthetic fish-like image can run through `measure`;
- `update-points` succeeds;
- `export` succeeds for CSV, Excel/XLSX, JSON and preview PNG and returns project-relative output paths;
- exported CSV, XLSX, JSON and preview files exist under the V2 user output directory;
- `exports` listing returns V2 export folders with project-relative paths;
- `session` lookup returns a public V2 payload without internal original or warped image paths;
- exported files are written under `results/v2_user_outputs/exports/`.

Note: the synthetic image is only an endpoint-plumbing test. Real measurement quality still requires real calibration-board fish images.

### Frontend Static Audit

Command:

```powershell
python siganusmorph_v2/frontend/static_audit.py
```

Reports:

```text
results/v2_user_outputs/v2_frontend_static_audit.json
results/v2_user_outputs/v2_frontend_static_audit.csv
```

Current result:

```text
total_checks: 30
passed_checks: 30
failed_checks: 0
```

Verified from source without installing npm dependencies:

- required React pages exist;
- API client contains required endpoint calls, including `/api/session/{session_id}`;
- API client filters backend tracebacks into user-friendly messages;
- required Chinese user-facing labels are present;
- Apple-inspired design tokens are present;
- MeasurementCanvas includes required draggable points;
- P7V is derived and not a direct point in the required draggable set;
- display-to-image coordinate mapping is present;
- larger hit radius logic is present;
- hover label, crosshair, magnifier and keyboard fine-tuning are present;
- Apply Changes callback is present;
- SingleFishPage performs upload, calibrate and measure flow;
- SingleFishPage can restore an existing V2 session;
- calibration failure blocks measurement in SingleFishPage;
- BatchPage calls upload, calibrate, measure and export;
- ExportPage reads backend exports with browser-local fallback;
- SettingsPage reads backend settings;
- frontend source has no local absolute paths;
- user-facing page/component/style source has no debug/audit/experimental/traceback labels.
- user-facing frontend source has no known mojibake text patterns.
- frontend relative imports resolve;
- external frontend imports are declared in `package.json`;
- required Apple-style CSS classes are defined.

### API Contract Audit

Command:

```powershell
python siganusmorph_v2/contract_audit.py
```

Reports:

```text
results/v2_user_outputs/v2_contract_audit.json
results/v2_user_outputs/v2_contract_audit.csv
```

Current result:

```text
total_checks: 13
passed_checks: 13
failed_checks: 0
```

Verified:

- backend Pydantic response models exist for the public API responses;
- backend routes declare response models for health, settings, session lookup, upload, calibration, measurement, update-points, export and export listing;
- frontend API response types exist for the same payloads;
- frontend client calls all backend endpoints;
- shared fields such as `version`, `session_id`, `specimen_id`, `weight_g`, `coordinate_space`, `formal_points`, `measurements`, `files` and `saved_to` are aligned;
- `coordinate_space = warped_image` is explicit in backend and frontend;
- backend calibration and export gates are present;
- specimen id and weight flow through upload, session restore and export;
- manual calibration corners are explicit;
- API documentation covers all implemented endpoints.

### OpenAPI Snapshot

Command:

```powershell
python siganusmorph_v2/openapi_snapshot.py
```

Reports:

```text
results/v2_user_outputs/v2_openapi_snapshot.json
results/v2_user_outputs/v2_openapi_summary.json
results/v2_user_outputs/v2_openapi_endpoints.csv
```

Current result:

```text
total_endpoints: 9
total_schemas: 18
total_checks: 5
passed_checks: 5
failed_checks: 0
```

Verified:

- all required V2 endpoints are present in FastAPI OpenAPI output;
- required response schemas are present;
- OpenAPI version metadata is `2.0.0`;
- `MeasureResponse.coordinate_space` is constrained to `warped_image`;
- `ExportResponse` includes `files` and `saved_to`.

### V2 Readiness Aggregation

Command:

```powershell
python siganusmorph_v2/verify_v2_readiness.py
```

Reports:

```text
results/v2_user_outputs/v2_readiness_report.json
results/v2_user_outputs/v2_readiness_report.csv
results/v2_user_outputs/v2_readiness_commands.log
results/v2_user_outputs/v2_completion_audit.json
results/v2_user_outputs/v2_completion_audit.csv
```

Current result:

```text
total_checks: 24
passed_checks: 24
failed_checks: 0
overall_status: pass_browser_visual_workflow_verified
```

Passed:

- Python V2 backend/admin/audit files compile;
- backend API smoke test passes;
- live Uvicorn backend health check passes;
- live HTTP API flow passes;
- frontend static audit passes;
- SingleFishPage and MeasurementCanvas workflow audit passes;
- frontend Vite runtime HTTP check passes;
- backend/frontend/API documentation contract audit passes;
- FastAPI OpenAPI snapshot exports the expected V2 API contract;
- V2 launcher scripts exist and avoid historical/destructive paths;
- npm dependency manifest audit passes before installation approval;
- Tauri runtime diagnostic report is generated;
- Tauri dev shell is manually verified;
- required V2 source and documentation files exist;
- frontend and Tauri package scripts are present;
- frontend dependencies are installed;
- Tauri dependencies are installed;
- Rust Cargo toolchain is available for Tauri;
- no local absolute paths were found in V2 source/docs/build report;
- no common mojibake text patterns were found in V2 source/docs/build report;
- backend smoke exports stay inside `results/v2_user_outputs/exports/`;
- this build report records the current manual-workflow validation gap.

Layered runtime status:

```text
Backend runtime: passed
/api/health: passed
Live HTTP API flow: passed
React/Vite runtime: passed
Tauri dev shell: passed
Tauri release executable build: passed
Tauri installer package: not enabled
Browser-level drag interaction: passed
SingleFishPage full visual workflow: passed_browser_visual_workflow
V1.0 untouched: yes
Historical data untouched: yes
corrected_keypoints untouched: yes
```

The desktop shell now launches in development mode and loads the React/Vite frontend. Browser-level React/FastAPI validation confirms the SingleFishPage workflow, warped canvas display, point drag, Apply Changes, export and ExportPage listing.

Completion audit:

```text
siganusmorph_v2/docs/v2_completion_audit.md
```

Current audit summary:

```text
requirements reviewed: 17
complete or complete with boundary/source/browser evidence: 16
pending functional requirements: 0
nonblocking external pending items: 1
overall_status: pass_browser_visual_workflow_verified
```

Browser visual workflow report:

```text
results/v2_user_outputs/v2_frontend_visual_workflow_check.json
passed_checks: 15
failed_checks: 0
```

## 4. Startup Commands

### Backend

```powershell
python -m uvicorn siganusmorph_v2.backend.app:app --host 127.0.0.1 --port 8000 --reload
```

### Backend Smoke Test

```powershell
python siganusmorph_v2/backend/smoke_test_api.py
```

### Full V2 Readiness Check

```powershell
python siganusmorph_v2/verify_v2_readiness.py
```

### Tauri Runtime Diagnostic

```powershell
python siganusmorph_v2/tauri_runtime_diagnostic.py
```

### Frontend

```powershell
cd siganusmorph_v2/frontend
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

### Tauri

```powershell
cd siganusmorph_v2/desktop/tauri
npm install
npm run tauri:dev
```

### Admin Console

```powershell
streamlit run siganusmorph_v2/admin_streamlit/app.py
```

## 5. API List

- `GET /api/health`
- `GET /api/settings`
- `GET /api/session/{session_id}`
- `GET /api/exports`
- `POST /api/upload`
- `POST /api/calibrate`
- `POST /api/measure`
- `POST /api/update-points`
- `POST /api/export`

## 6. Data Flow

```text
raw image
-> calibration
-> warped image
-> segmentation/preannotation
-> formal points
-> measurements
-> export
```

## 7. V1.0 and Historical Data Protection

Protected:

- V1.0 Streamlit frontend;
- V1.0 release package;
- old Streamlit backend;
- Review Queue;
- Advanced overlay;
- `corrected_keypoints`;
- historical batch measurement;
- `final_analysis_dataset`;
- model weights.

V2 user outputs write only to:

```text
results/v2_user_outputs/
```

Future admin workflows should write to:

```text
results/v2_admin_outputs/
```

## 8. Blocking Issues

1. Tauri dev shell is verified. The desktop window opens with title `蓝子鱼形态测量系统 V2.0`, loads the React/Vite frontend, and displays the HomePage with the expected navigation and Apple-inspired styling.

2. SingleFishPage browser-level visual workflow is verified. Tauri-window screenshots can still be collected manually for product signoff if needed.

3. MeasurementCanvas browser-level interaction is verified for rendered points, hover label, crosshair during drag, local metrics, Apply Changes and export. Keyboard fine-tuning remains covered by source audit.

4. Tauri release executable build is verified. Installer/package generation is still not enabled because `bundle.active = false`.

5. Real-image measurement quality is not yet validated in V2.0. The browser visual workflow uses a synthetic/manual-corner session; a real calibration-board fish photo is still recommended for product-level measurement QA.

## 9. Next TODO

1. Collect Tauri-window screenshots for HomePage, SingleFishPage and ExportPage if needed for presentation/signoff.
2. Test a real calibration-board image through:

   ```text
   upload -> calibrate -> measure -> update-points -> export
   ```

3. Run `npm audit` for the frontend when the audit endpoint is reachable; do not auto-upgrade major versions without review.
4. Decide whether to enable Tauri installer/package generation after the manual workflow is stable.
5. Decide how Tauri should launch or bundle the Python FastAPI backend.
6. Expand BatchPage to call batch backend jobs.
7. Add automated API tests using real fixture images when they are available.

## V2.0 Release Candidate Freeze

Generated: 2026-06-20 20:11:35

- Release candidate package: `release_v2.0_software_copyright/`
- Readiness preserved: 25/25 pass
- Browser visual workflow preserved: 15/15 pass
- Real fish QA record: `results/v2_user_outputs/real_fish_qa_v2_release_candidate.json`
- Functional blocking issues: none
- Nonblocking items: npm audit depends on registry/network availability; Tauri installer/package generation is not enabled.
- V1.0, corrected_keypoints and historical batch/final analysis remain untouched.

