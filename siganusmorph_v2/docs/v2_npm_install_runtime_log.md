# V2.0 NPM Install and Runtime Verification Log

Generated: 2026-06-19

## Scope

This log covers controlled npm dependency installation and runtime verification for SiganusMorph Local V2.0.

Allowed npm working directories:

- `siganusmorph_v2/frontend/`
- `siganusmorph_v2/desktop/tauri/`

Protected boundaries:

- no V1.0 release files are intentionally modified;
- no `corrected_keypoints` are modified;
- no historical batch or final-analysis results are overwritten;
- no model training is performed;
- V2 runtime outputs remain under `results/v2_user_outputs/` or `results/v2_admin_outputs/`.

## Step 1. Pre-Install Audit

Frontend package:

```text
siganusmorph_v2/frontend/package.json
```

Frontend scripts:

- `dev`: `vite --host 127.0.0.1 --port 5173`
- `build`: `tsc && vite build`
- `preview`: `vite preview --host 127.0.0.1 --port 4173`

Frontend dependencies:

- `@vitejs/plugin-react`
- `vite`
- `typescript`
- `react`
- `react-dom`

Tauri package:

```text
siganusmorph_v2/desktop/tauri/package.json
```

Tauri scripts:

- `tauri:dev`: `tauri dev`
- `tauri:build`: `tauri build`

Tauri dev dependencies:

- `@tauri-apps/cli`

Pre-install audit command:

```powershell
python siganusmorph_v2/npm_dependency_audit.py
```

Pre-install audit result:

```text
total_checks: 6
passed_checks: 6
failed_checks: 0
warning_count: 6
```

Notes:

- no local project `preinstall`, `install`, `postinstall`, `prepare` or related lifecycle scripts were found;
- dependency names match the V2 allowlist;
- semver ranges are recorded as warnings;
- frontend lockfile/node_modules are now present, while Tauri lockfile/node_modules remain missing because installation failed.

## Step 2. Controlled Install

Frontend command:

```powershell
cd siganusmorph_v2/frontend
npm install
```

Result:

```text
npm install completed: no
exit_code: 1
error: ECONNRESET
registry: https://registry.npm.taobao.org/
failed_request: https://registry.npm.taobao.org/@vitejs%2fplugin-react
reason: Client network socket disconnected before secure TLS connection was established
```

A first attempt timed out before producing `node_modules` or `package-lock.json`. A second controlled attempt returned a concrete network download failure:

```text
npm error code ECONNRESET
npm error network request to https://registry.npm.taobao.org/@vitejs%2fplugin-react failed
```

Post-failure state:

```text
siganusmorph_v2/frontend/node_modules: present
siganusmorph_v2/frontend/package-lock.json: present
top-level frontend dependencies resolved by npm ls: 5 / 5
residual npm/node processes: none detected
```

Action taken:

- stopped the install workflow;
- did not change npm registry;
- did not bypass the network failure;
- recorded that the install command exited non-zero even though frontend dependencies were later found on disk;
- continued with local build/runtime verification using the installed frontend dependencies.

Tauri install was attempted after frontend build/runtime verification became possible. It failed with the same network class:

```text
npm install completed: no
error: ECONNRESET
registry: https://registry.npm.taobao.org/
failed_request: https://registry.npm.taobao.org/@tauri-apps%2fcli
reason: Client network socket disconnected before secure TLS connection was established
```

Tauri install was retried on 2026-06-20 under the same controlled rule and failed with the same request:

```text
npm install completed: no
error: ECONNRESET
failed_request: https://registry.npm.taobao.org/@tauri-apps%2fcli
siganusmorph_v2/desktop/tauri/node_modules: missing
siganusmorph_v2/desktop/tauri/package-lock.json: missing
```

Later on 2026-06-20 the Tauri environment was repaired outside this log's initial failed attempt:

```text
siganusmorph_v2/desktop/tauri/node_modules: present
siganusmorph_v2/desktop/tauri/package-lock.json: present
@tauri-apps/cli: 2.11.3
npm registry: https://registry.npmmirror.com/
Tauri dev shell: opened successfully
```

Rust/GNU toolchain paths recorded for the successful Tauri dev shell:

```text
RUSTUP_HOME: D:/1_postgraduate/.rustup
CARGO_HOME: D:/1_postgraduate/.cargo
Rust host: x86_64-pc-windows-gnu
MinGW/GCC: D:/1Bio_Soft/mingw64/bin
Recommended CARGO_TARGET_DIR: D:/1_postgraduate/tauri_target_siganusmorph_v2
```

Tauri icon generation:

```text
npx tauri icon
```

See `siganusmorph_v2/docs/v2_tauri_environment_setup.md` and `results/v2_user_outputs/v2_tauri_runtime_diagnostic.json`.

## Step 3. Post-Install Audit

Frontend command:

```powershell
cd siganusmorph_v2/frontend
npm audit --json
```

Result:

```text
npm audit completed: no
error: audit endpoint returned an error
failed_request: https://registry.npmmirror.com/-/npm/v1/security/advisories/bulk
reason: Client network socket disconnected before secure TLS connection was established
```

No high or critical vulnerability result is available because the audit endpoint was unreachable.

The audit command was retried on 2026-06-20 and still failed at the same audit endpoint. Vulnerability severity remains `unknown`; no automatic package upgrade was attempted.

## Step 4. Frontend Runtime Verification

Frontend build command:

```powershell
cd siganusmorph_v2/frontend
npm run build
```

Result:

```text
npm run build: pass
dist/index.html: present
```

Build implementation note:

```text
The build script uses `vite build --configLoader runner` to avoid Vite writing a temporary bundled config file under node_modules/.vite-temp on this Windows workspace.
```

Frontend dev server:

```powershell
cd siganusmorph_v2/frontend
npm run dev
```

Result:

```text
React/Vite dev server works: yes
URL: http://127.0.0.1:5173
```

Runtime HTTP verification:

```powershell
python siganusmorph_v2/frontend/runtime_http_check.py
```

Result:

```text
total_checks: 7
passed_checks: 7
failed_checks: 0
```

## Step 5. Frontend/Backend Runtime Loop

Partially verified.

Verified:

- Vite dev server serves the V2 frontend root page;
- `/src/main.tsx`, `/src/App.tsx` and `/src/pages/SingleFishPage.tsx` transform successfully through Vite;
- frontend runtime can reach backend `/api/health`;
- Apple-inspired CSS tokens are served;
- user-visible frontend root HTML does not expose local absolute paths or Python tracebacks.

Not verified:

- browser-rendered visual interaction with HomePage and SingleFishPage;
- drag/drop point editing through an actual browser session;
- upload/calibrate/measure/export through React UI clicks.

Reason:

```text
The in-app browser control tool was not available in this session. The backend live HTTP loop remains verified separately.
```

Backend-only live HTTP flow remains verified by:

```powershell
python siganusmorph_v2/backend/live_api_flow_check.py
```

Latest known backend live HTTP flow status:

```text
passed_checks: 8
failed_checks: 0
```

## Step 6. Tauri Runtime Verification

Read-only diagnostic command:

```powershell
python siganusmorph_v2/tauri_runtime_diagnostic.py
```

Latest status:

```text
passed_checks: 14
failed_checks: 0
tauri_runtime_ready: true
Tauri dev shell: opened successfully
```

## Step 7. Final Result

```text
frontend npm install command completed cleanly: no
frontend dependencies available on disk: yes
npm audit completed: no
high_or_critical_vulnerabilities: unknown
React dev server works: yes
Frontend calls backend health API: yes
HomePage runtime verified: yes, visible in Tauri desktop window
SingleFishPage runtime verified: partially, via Vite module transform check
Single-fish runtime loop verified: backend HTTP loop yes; React UI click/drag flow not browser-verified
Tauri dev shell works: yes
```

Remaining issues:

```text
npm audit still cannot reach the audit endpoint.
SingleFishPage browser-level visual workflow is verified.
MeasurementCanvas browser-level drag/hover/crosshair/apply/export behavior is verified.
Tauri release executable build is verified.
Tauri installer/package is not generated yet because bundle.active is false.
```

## Step 8. Tauri Release Build Verification

Controlled command:

```powershell
cd siganusmorph_v2/desktop/tauri
npm run tauri:build
```

Environment used:

```text
CARGO_HOME: D:/1_postgraduate/.cargo
RUSTUP_HOME: D:/1_postgraduate/.rustup
PATH includes: D:/1_postgraduate/.cargo/bin and D:/1Bio_Soft/mingw64/bin
CARGO_TARGET_DIR: D:/1_postgraduate/tauri_target_siganusmorph_v2
```

Result:

```text
Tauri release executable build: pass
Output executable: D:/1_postgraduate/tauri_target_siganusmorph_v2/release/siganusmorph_v2.exe
Installer/package: not generated because bundle.active is false
```

Resolved blockers:

```text
frontendDist was corrected from ../../frontend/dist to ../../../frontend/dist.
Using a Cargo target directory under the workspace path with a space caused MinGW windres to fail.
The user-specified D:/1_postgraduate/tauri_target_siganusmorph_v2 target directory avoids the path-space issue.
```

Next safe step:

Open SingleFishPage in the Tauri window and verify upload, calibration, measurement, point drag/update, and export with a real or fixture image.
