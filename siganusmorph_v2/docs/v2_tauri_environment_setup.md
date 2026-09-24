# V2.0 Tauri Environment Setup

This note records the desktop runtime environment that made the SiganusMorph Local V2.0 Tauri development shell launch successfully.

## Runtime Status

Current status:

```text
Tauri dev shell: passed
React/Vite frontend loaded in desktop window: passed
Tauri release executable build: passed
Tauri installer/package: not enabled
SingleFishPage browser-level visual workflow: passed
```

The opened desktop window shows:

- title: `蓝子鱼形态测量系统 V2.0`
- navigation: 首页, 单鱼测量, 批量测量, 结果导出, 设置
- HomePage rendered with the Apple-inspired visual style

## Rust And GNU Toolchain

Rust/Cargo was moved to:

```text
D:/1_postgraduate/.rustup
D:/1_postgraduate/.cargo
```

Rust host triple:

```text
x86_64-pc-windows-gnu
```

Observed tool versions:

```text
cargo 1.96.0
rustc 1.96.0
gcc 16.1.0
```

MinGW/GCC path:

```text
D:/1Bio_Soft/mingw64/bin
```

Recommended Cargo target output directory:

```text
D:/1_postgraduate/tauri_target_siganusmorph_v2
```

Using a dedicated target directory keeps Tauri build artifacts outside the V1.0 release and historical result folders.

The dedicated target directory is also required on this Windows setup because the workspace path contains a space. A build attempt with a target directory under the workspace reached Rust release compilation but failed in MinGW `windres` path preprocessing. Building with `D:/1_postgraduate/tauri_target_siganusmorph_v2` succeeds.

## Tauri Icons

The Tauri icon issue was resolved by generating standard icons with:

```powershell
npx tauri icon
```

The generated icons are part of the Tauri shell runtime setup. They do not modify measurement algorithms, `corrected_keypoints`, V1.0 release files, or historical results.

## Runtime Checks

Run the read-only diagnostic:

```powershell
python siganusmorph_v2/tauri_runtime_diagnostic.py
```

Latest expected result:

```text
total_checks: 14
passed_checks: 14
failed_checks: 0
tauri_runtime_ready: true
```

Run the full V2 readiness check:

```powershell
python siganusmorph_v2/verify_v2_readiness.py
```

Latest expected status:

```text
overall_status: pass_browser_visual_workflow_verified
```

Release build verification:

```powershell
cd siganusmorph_v2/desktop/tauri
npm run tauri:build
```

Latest controlled result:

```text
release executable build: passed
output: D:/1_postgraduate/tauri_target_siganusmorph_v2/release/siganusmorph_v2.exe
installer package: not generated because bundle.active is false
```

## Remaining Validation

Still recommended for product signoff:

- SingleFishPage screenshots inside the Tauri window
- real calibration-board image QA
- Tauri installer/package generation, if needed for distribution

All V2 runtime outputs remain under:

```text
results/v2_user_outputs/
```
