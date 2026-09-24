# SiganusMorph Local V2.0 Tauri Shell

This directory contains the V2.0 desktop wrapper skeleton.

## Current Development Mode

The Python FastAPI backend is started manually:

```powershell
python -m uvicorn siganusmorph_v2.backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Convenience launcher from the project root:

```powershell
powershell -ExecutionPolicy Bypass -File siganusmorph_v2/scripts/run_backend.ps1
```

The Tauri dev shell starts the Vite frontend through `beforeDevCommand`:

```powershell
cd siganusmorph_v2/desktop/tauri
npm install
npm run tauri:dev
```

Convenience launcher from the project root after dependencies are installed:

```powershell
powershell -ExecutionPolicy Bypass -File siganusmorph_v2/scripts/run_tauri_dev.ps1
```

The React frontend connects to:

```text
http://127.0.0.1:8000
```

## Packaging Boundary

This is a skeleton only. It does not yet bundle Python, FastAPI, model weights, or the measurement core into the desktop installer.

Future packaging decisions:

- whether Tauri launches a bundled Python runtime;
- how model weights are located in packaged mode;
- how output directories are selected by ordinary users;
- how admin workflows remain separated from the ordinary user app.

## Safety Boundary

The Tauri shell must not write to historical V1.0 outputs. V2 user exports should remain under:

```text
results/v2_user_outputs/
```
