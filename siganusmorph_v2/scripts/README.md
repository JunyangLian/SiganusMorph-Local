# V2.0 Launcher Scripts

These scripts are convenience launchers for the V2.0 development skeleton.

They do not modify V1.0 release files, historical measurements, or `corrected_keypoints`.

## Scripts

- `check_runtime_prereqs.ps1`: checks Python, Node.js, npm, frontend dependencies and Tauri dependencies.
- `run_backend.ps1`: starts the FastAPI backend on `127.0.0.1:8000`.
- `run_frontend_dev.ps1`: starts the Vite frontend on `127.0.0.1:5173` after checking `node_modules`.
- `run_tauri_dev.ps1`: starts the Tauri development shell after checking `node_modules`.
- `run_admin_console.ps1`: starts the Streamlit Admin Console.
- `run_v2_readiness.ps1`: refreshes the V2 readiness report.

Frontend and Tauri scripts require npm dependencies to be installed first.

Before asking for npm installation approval, run:

```powershell
python siganusmorph_v2/npm_dependency_audit.py
```

To inspect desktop-shell prerequisites without installing dependencies, run:

```powershell
python siganusmorph_v2/tauri_runtime_diagnostic.py
```

See `siganusmorph_v2/docs/v2_npm_runtime_approval_guide.md` for the approval boundary and expected runtime blocker.
