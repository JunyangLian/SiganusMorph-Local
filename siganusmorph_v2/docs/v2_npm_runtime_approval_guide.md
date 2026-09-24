# V2.0 NPM Runtime Approval Guide

This guide documents the controlled npm approval boundary used for V2.0 runtime setup. It is kept as a record of why npm dependency installation required explicit approval.

The V2.0 source, backend API, live FastAPI health check and live HTTP API flow can be verified without installing npm packages. React and Tauri runtime verification required explicit approval because npm installation downloads third-party packages and can execute package lifecycle scripts.

## Current Dependency Manifests

Frontend:

```text
siganusmorph_v2/frontend/package.json
```

Runtime role:

- React user interface;
- Vite development server;
- TypeScript build check.

Tauri shell:

```text
siganusmorph_v2/desktop/tauri/package.json
```

Runtime role:

- desktop wrapper development mode;
- future desktop packaging.

## Safe Pre-Install Audit

Run:

```powershell
python siganusmorph_v2/npm_dependency_audit.py
```

Output:

```text
results/v2_user_outputs/v2_npm_dependency_audit.json
results/v2_user_outputs/v2_npm_dependency_audit.csv
```

The audit checks:

- package manifests exist;
- local project `preinstall`, `install`, `postinstall`, `prepare` and related lifecycle scripts are absent;
- dependency names match the current V2 allowlist;
- whether `node_modules` and package lock files are present.

## Approval Boundary

Do not run npm install automatically in this workspace unless the user explicitly approves the dependency installation risk.

If approved, the expected commands are:

```powershell
cd siganusmorph_v2/frontend
npm install
npm run build
npm run dev
```

For Tauri:

```powershell
cd siganusmorph_v2/desktop/tauri
npm install
npm run tauri:dev
```

## Why This Is Separate

NPM installation is different from the Python-only V2 checks because it downloads and executes third-party package code. Keeping this as an explicit approval gate protects the V1.0 frozen release, historical results and local workspace from unreviewed side effects.

## Pre-Install Readiness State

Before npm dependencies were approved and installed, the unified readiness report was expected to show:

```text
overall_status: runtime_blocked
runtime_ready: false
```

This meant backend and source audits could pass, but React/Tauri runtime rendering remained unverified.

## Current Runtime State

After the controlled V2 environment repair, frontend and Tauri dependencies are present, the Tauri dev shell opens successfully, the Tauri release executable build passes, and the browser-level SingleFishPage/MeasurementCanvas visual workflow is verified. Current status is recorded in:

```text
siganusmorph_v2/docs/v2_tauri_environment_setup.md
results/v2_user_outputs/v2_readiness_report.json
```

The remaining optional work is Tauri-window product-signoff screenshots, real calibration-board image QA, npm audit when the registry endpoint is reachable, and installer/package generation if distribution requires it.
