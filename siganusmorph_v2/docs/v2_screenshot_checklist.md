# V2.0 Screenshot Checklist

Recommended screenshots for the current V2.0 desktop-runtime stage:

1. Tauri desktop window with title `蓝子鱼形态测量系统 V2.0`.
2. HomePage with Apple-inspired cards and CTA buttons.
3. Left navigation: 首页, 单鱼测量, 批量测量, 结果导出, 设置.
4. SingleFishPage upload and calibration panel.
5. Warped image measurement workbench.
6. MeasurementCanvas hover label, crosshair and magnifier.
7. Measurement result cards.
8. Export success panel.
9. BatchPage queue and status badges.
10. ExportPage.
11. SettingsPage showing backend health.
12. Admin Console entry page.

Current verified screenshot:

- Tauri desktop HomePage and navigation.
- Tauri release executable build evidence can be documented with `results/v2_user_outputs/v2_tauri_build_report.json`.
- Browser-level SingleFishPage workflow screenshots:
  - `results/v2_user_outputs/visual_smoke/v2_browser_homepage.png`
  - `results/v2_user_outputs/visual_smoke/v2_browser_single_fish_after_drag_export.png`

Screenshots still recommended:

- SingleFishPage full workflow inside the Tauri desktop window;
- export success from the Tauri desktop window;
- a real calibration-board fish image run, if available.

Avoid screenshots with:

- Python traceback
- local absolute paths
- debug/audit/experimental user-facing labels
- unreviewed training labels
