# V2.0 Software Copyright Notes

SiganusMorph Local V2.0 separates the ordinary user application from the administrator/developer console.

Current V2.0 implementation includes:

- React/Vite/TypeScript user frontend;
- FastAPI backend API;
- Tauri desktop shell;
- Streamlit Admin Console entry;
- API, architecture, build, runtime and screenshot documentation.

Current validation status:

- the Tauri desktop shell starts successfully;
- the HomePage renders in the desktop window;
- React/Vite frontend runtime is available;
- Tauri release executable build passes;
- FastAPI `/api/health` passes;
- backend HTTP loop passes `upload -> calibrate -> measure -> update-points -> export`;
- browser-level SingleFishPage visual workflow and MeasurementCanvas interaction pass;
- Streamlit Admin Console is preserved as the advanced review/debug backend.

Still recommended before a formal V2.0 software-copyright submission:

- Tauri-window screenshots of SingleFishPage and ExportPage;
- real calibration-board image QA screenshots;
- installer/package generation, if a distributable installer is required.

V1.0 remains the software-copyright frozen release until V2.0 validation is complete.

Suggested wording:

> 系统 V2.0 包含 React 用户端、FastAPI 后端、Tauri 桌面壳和 Streamlit 管理者后台。当前用户端桌面壳已可启动，Tauri release 可执行文件已可构建，后端 API、单鱼测量 HTTP 闭环和浏览器级测量工作台交互已通过验证，管理者后台保留完整复核与调试功能。

V2.0 does not modify:

- V1.0 release package
- corrected_keypoints
- historical batch measurement
- final_analysis_dataset
- existing model weights
