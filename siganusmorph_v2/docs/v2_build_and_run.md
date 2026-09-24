# V2.0 Build and Run

## Backend

Development command:

```powershell
python -m uvicorn siganusmorph_v2.backend.app:app --host 127.0.0.1 --port 8000 --reload
```

V2.0 user-facing React/Tauri clients expect the FastAPI backend at:

```text
http://127.0.0.1:8000
```

Do not run the user-facing backend on the old development port `8765`; otherwise upload/calibrate/measure calls from the browser or Tauri shell will fail.

Launcher script:

```powershell
powershell -ExecutionPolicy Bypass -File siganusmorph_v2/scripts/run_backend.ps1
```

Health check:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/api/health
```

Live backend health check:

```powershell
python siganusmorph_v2/backend/live_health_check.py
```

This starts a temporary local Uvicorn server on `127.0.0.1`, verifies `/api/health`, and then stops the server automatically. It writes:

```text
results/v2_user_outputs/v2_live_backend_health_check.json
results/v2_user_outputs/v2_live_backend_health_check.csv
```

Live HTTP API flow check:

```powershell
python siganusmorph_v2/backend/live_api_flow_check.py
```

This starts a temporary local Uvicorn server and exercises the same HTTP path that the React/Tauri client will use:

```text
upload -> calibrate -> measure -> update-points -> export -> session restore -> export listing
```

It writes:

```text
results/v2_user_outputs/v2_live_api_flow_check.json
results/v2_user_outputs/v2_live_api_flow_check.csv
```

Backend API smoke test:

```powershell
python siganusmorph_v2/backend/smoke_test_api.py
```

The smoke test verifies:

- `/api/health`;
- upload session creation;
- calibration failure blocks measurement;
- manual-corner calibration can produce a valid warped image and `manual_board_corners` scale source;
- measurement, update-points, and export endpoint plumbing on a synthetic fish-like image.

The smoke test writes only under:

```text
results/v2_user_outputs/
```

Frontend static audit:

```powershell
python siganusmorph_v2/frontend/static_audit.py
```

This dependency-free audit checks that the React source contains the required pages, required Chinese user-facing labels, API calls, Apple-inspired tokens, MeasurementCanvas interactions, coordinate mapping, and user-facing safety constraints. It writes:

```text
results/v2_user_outputs/v2_frontend_static_audit.json
results/v2_user_outputs/v2_frontend_static_audit.csv
```

API contract audit:

```powershell
python siganusmorph_v2/contract_audit.py
```

This dependency-free audit checks backend Pydantic response models, backend route response models, frontend TypeScript API types, shared response fields, calibration/export gates, coordinate-space invariants and API documentation coverage. It writes:

```text
results/v2_user_outputs/v2_contract_audit.json
results/v2_user_outputs/v2_contract_audit.csv
```

NPM dependency manifest audit:

```powershell
python siganusmorph_v2/npm_dependency_audit.py
```

This dependency-free audit checks the V2 frontend and Tauri `package.json` files before any npm install is approved. It writes:

```text
results/v2_user_outputs/v2_npm_dependency_audit.json
results/v2_user_outputs/v2_npm_dependency_audit.csv
```

See:

```text
siganusmorph_v2/docs/v2_npm_runtime_approval_guide.md
```

OpenAPI snapshot:

```powershell
python siganusmorph_v2/openapi_snapshot.py
```

This exports the FastAPI OpenAPI document and a compact endpoint table for frontend/API review. It writes:

```text
results/v2_user_outputs/v2_openapi_snapshot.json
results/v2_user_outputs/v2_openapi_summary.json
results/v2_user_outputs/v2_openapi_endpoints.csv
```

Unified readiness check:

```powershell
python siganusmorph_v2/verify_v2_readiness.py
```

Equivalent launcher script:

```powershell
powershell -ExecutionPolicy Bypass -File siganusmorph_v2/scripts/run_v2_readiness.ps1
```

This command refreshes the Python compile check, backend smoke test, live backend health check, live HTTP API flow check, frontend static audit, API contract audit, OpenAPI snapshot, launcher audit, source/documentation file inventory, V2 output-boundary checks, local-path scan and common mojibake scan. It writes:

```text
results/v2_user_outputs/v2_readiness_report.json
results/v2_user_outputs/v2_readiness_report.csv
results/v2_user_outputs/v2_readiness_commands.log
```

Current controlled runtime status is:

```text
overall_status = pass_browser_visual_workflow_verified
```

Backend runtime, `/api/health`, live HTTP API flow, React/Vite runtime, Tauri dev shell, Tauri release executable build, and browser-level SingleFishPage visual workflow are verified. Installer/package generation is not enabled yet.

Browser visual workflow check:

```powershell
python siganusmorph_v2/frontend/visual_workflow_check.py
```

Latest result:

```text
passed_checks: 15
failed_checks: 0
visual_workflow_ready: true
```

Runtime prerequisite check:

```powershell
powershell -ExecutionPolicy Bypass -File siganusmorph_v2/scripts/check_runtime_prereqs.ps1
```

## Frontend

Install dependencies:

```powershell
cd siganusmorph_v2/frontend
npm install
```

Dependency installation requires network access and runs third-party npm package lifecycle scripts. In the current controlled run, frontend dependencies are present on disk and React/Vite build/runtime HTTP checks pass. `npm audit` still needs to be rerun when the audit endpoint is reachable.

Current controlled install status is recorded in:

```text
siganusmorph_v2/docs/v2_npm_install_runtime_log.md
```

Latest controlled runtime notes:

```text
frontend npm install exited with ECONNRESET but dependencies are present and React/Vite build/runtime HTTP checks pass.
Tauri dependencies are now present and Tauri dev shell opens the desktop window.
Rust/Cargo is available through D:/1_postgraduate/.cargo and D:/1_postgraduate/.rustup.
MinGW/GCC is available through D:/1Bio_Soft/mingw64/bin.
npm audit still needs to be rerun when the audit endpoint is reachable.
```

Do not switch npm registry, proxy, or installation source without explicit user approval.

Run:

```powershell
npm run dev
```

Frontend workflow source audit:

```powershell
python siganusmorph_v2/frontend/workflow_audit.py
```

This verifies that SingleFishPage and MeasurementCanvas are wired for upload, calibration gating, warped-image canvas rendering, draggable formal points, P7V derivation, hover label, crosshair, magnifier, keyboard fine-tuning and `/api/update-points`.

Launcher script:

```powershell
powershell -ExecutionPolicy Bypass -File siganusmorph_v2/scripts/run_frontend_dev.ps1
```

Open:

```text
http://127.0.0.1:5173
```

## Tauri

Development mode currently assumes the FastAPI backend is started manually.

Environment setup notes:

```text
siganusmorph_v2/docs/v2_tauri_environment_setup.md
```

Read-only Tauri diagnostic:

```powershell
python siganusmorph_v2/tauri_runtime_diagnostic.py
```

```powershell
cd siganusmorph_v2/desktop/tauri
npm run tauri:dev
```

Port and process-lock note:

```text
The React/Tauri user client uses http://127.0.0.1:8000 for all user-facing API calls.
If a rerun fails with Windows os error 32 while compiling, close any existing
siganusmorph_v2.exe / cargo / tauri dev process that is holding
D:/1_postgraduate/tauri_target_siganusmorph_v2, then rerun tauri:dev.
This is a target-directory file lock, not an API port failure.
```

Launcher script:

```powershell
powershell -ExecutionPolicy Bypass -File siganusmorph_v2/scripts/run_tauri_dev.ps1
```

Future packaging work should decide how to bundle or launch the Python backend from Tauri.

Release executable build:

```powershell
$env:CARGO_HOME='D:\1_postgraduate\.cargo'
$env:RUSTUP_HOME='D:\1_postgraduate\.rustup'
$env:PATH='D:\1_postgraduate\.cargo\bin;D:\1Bio_Soft\mingw64\bin;' + $env:PATH
$env:CARGO_TARGET_DIR='D:\1_postgraduate\tauri_target_siganusmorph_v2'
cd siganusmorph_v2/desktop/tauri
npm run tauri:build
```

Current controlled result:

```text
Tauri release executable build: passed
Output executable: D:/1_postgraduate/tauri_target_siganusmorph_v2/release/siganusmorph_v2.exe
Installer/package: not generated because bundle.active is false
```

Why the dedicated target directory matters:

```text
When CARGO_TARGET_DIR was under the workspace path with a space in its name, MinGW windres split the path and failed.
The D:/1_postgraduate/tauri_target_siganusmorph_v2 target directory avoids the path-space issue.
```

Additional notes:

```text
siganusmorph_v2/desktop/tauri/README.md
```

## Admin Console

```powershell
streamlit run siganusmorph_v2/admin_streamlit/app.py
```

Launcher script:

```powershell
powershell -ExecutionPolicy Bypass -File siganusmorph_v2/scripts/run_admin_console.ps1
```

The Admin Console is an entry/status page for managers and developers. It preserves the legacy Streamlit backend, Review Queue and advanced workflows without making them the ordinary user entry.
