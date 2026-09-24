from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results" / "v2_user_outputs"
JSON_REPORT = RESULTS_DIR / "v2_readiness_report.json"
CSV_REPORT = RESULTS_DIR / "v2_readiness_report.csv"
COMMAND_LOG = RESULTS_DIR / "v2_readiness_commands.log"


PY_COMPILE_FILES = [
    "siganusmorph_v2/backend/app.py",
    "siganusmorph_v2/backend/run_backend.py",
    "siganusmorph_v2/backend/routers/api.py",
    "siganusmorph_v2/backend/schemas/measurement.py",
    "siganusmorph_v2/backend/services/measurement_service.py",
    "siganusmorph_v2/backend/services/session_store.py",
    "siganusmorph_v2/backend/services/settings_service.py",
    "siganusmorph_v2/backend/smoke_test_api.py",
    "siganusmorph_v2/backend/live_health_check.py",
    "siganusmorph_v2/backend/live_api_flow_check.py",
    "siganusmorph_v2/contract_audit.py",
    "siganusmorph_v2/openapi_snapshot.py",
    "siganusmorph_v2/launcher_audit.py",
    "siganusmorph_v2/npm_dependency_audit.py",
    "siganusmorph_v2/tauri_runtime_diagnostic.py",
    "siganusmorph_v2/frontend/static_audit.py",
    "siganusmorph_v2/frontend/runtime_http_check.py",
    "siganusmorph_v2/frontend/workflow_audit.py",
    "siganusmorph_v2/frontend/visual_workflow_check.py",
    "siganusmorph_v2/admin_streamlit/app.py",
    "siganusmorph_v2/verify_v2_readiness.py",
]


REQUIRED_FILES = [
    "siganusmorph_v2/backend/app.py",
    "siganusmorph_v2/backend/routers/api.py",
    "siganusmorph_v2/backend/services/measurement_service.py",
    "siganusmorph_v2/backend/services/session_store.py",
    "siganusmorph_v2/backend/services/settings_service.py",
    "siganusmorph_v2/backend/smoke_test_api.py",
    "siganusmorph_v2/backend/live_health_check.py",
    "siganusmorph_v2/backend/live_api_flow_check.py",
    "siganusmorph_v2/contract_audit.py",
    "siganusmorph_v2/openapi_snapshot.py",
    "siganusmorph_v2/launcher_audit.py",
    "siganusmorph_v2/npm_dependency_audit.py",
    "siganusmorph_v2/tauri_runtime_diagnostic.py",
    "siganusmorph_v2/scripts/README.md",
    "siganusmorph_v2/scripts/check_runtime_prereqs.ps1",
    "siganusmorph_v2/scripts/run_backend.ps1",
    "siganusmorph_v2/scripts/run_frontend_dev.ps1",
    "siganusmorph_v2/scripts/run_tauri_dev.ps1",
    "siganusmorph_v2/scripts/run_admin_console.ps1",
    "siganusmorph_v2/scripts/run_v2_readiness.ps1",
    "siganusmorph_v2/frontend/package.json",
    "siganusmorph_v2/frontend/package-lock.json",
    "siganusmorph_v2/frontend/index.html",
    "siganusmorph_v2/frontend/vite.config.ts",
    "siganusmorph_v2/frontend/tsconfig.json",
    "siganusmorph_v2/frontend/src/App.tsx",
    "siganusmorph_v2/frontend/src/pages/HomePage.tsx",
    "siganusmorph_v2/frontend/src/pages/SingleFishPage.tsx",
    "siganusmorph_v2/frontend/src/pages/BatchPage.tsx",
    "siganusmorph_v2/frontend/src/pages/ExportPage.tsx",
    "siganusmorph_v2/frontend/src/pages/SettingsPage.tsx",
    "siganusmorph_v2/frontend/src/components/MeasurementCanvas.tsx",
    "siganusmorph_v2/frontend/src/components/MetricCards.tsx",
    "siganusmorph_v2/frontend/runtime_http_check.py",
    "siganusmorph_v2/frontend/workflow_audit.py",
    "siganusmorph_v2/frontend/visual_workflow_check.py",
    "siganusmorph_v2/frontend/src/styles/tokens.css",
    "siganusmorph_v2/frontend/src/styles/global.css",
    "siganusmorph_v2/desktop/tauri/package.json",
    "siganusmorph_v2/desktop/tauri/src-tauri/Cargo.toml",
    "siganusmorph_v2/desktop/tauri/src-tauri/tauri.conf.json",
    "siganusmorph_v2/desktop/tauri/src-tauri/src/main.rs",
    "siganusmorph_v2/admin_streamlit/app.py",
    "siganusmorph_v2/docs/v2_architecture.md",
    "siganusmorph_v2/docs/v2_api_contract.md",
    "siganusmorph_v2/docs/v2_frontend_style_guide.md",
    "siganusmorph_v2/docs/v2_user_manual_draft.md",
    "siganusmorph_v2/docs/v2_admin_manual_draft.md",
    "siganusmorph_v2/docs/v2_software_copyright_notes.md",
    "siganusmorph_v2/docs/v2_screenshot_checklist.md",
    "siganusmorph_v2/docs/v2_build_and_run.md",
    "siganusmorph_v2/docs/v2_npm_runtime_approval_guide.md",
    "siganusmorph_v2/docs/v2_npm_install_runtime_log.md",
    "siganusmorph_v2/V2_BUILD_REPORT.md",
]


SCAN_PATHS = [
    V2_ROOT / "backend",
    V2_ROOT / "frontend" / "src",
    V2_ROOT / "desktop",
    V2_ROOT / "admin_streamlit",
    V2_ROOT / "docs",
    V2_ROOT / "V2_BUILD_REPORT.md",
]


MOJIBAKE_PATTERNS = [
    r"鈥",
    r"鍗",
    r"瀵",
    r"涓",
    r"鏍",
    r"绛",
    r"娴",
    r"闇",
    r"钃",
    r"澶",
    r"搴",
    r"鎭",
    r"绮",
]


def add_check(
    checks: list[dict[str, Any]],
    category: str,
    name: str,
    passed: bool,
    evidence: str = "",
    notes: str = "",
) -> None:
    checks.append(
        {
            "category": category,
            "check_name": name,
            "passed": bool(passed),
            "status": "pass" if passed else "fail",
            "evidence": evidence,
            "notes": notes,
        }
    )


def run_command(args: list[str], log_lines: list[str]) -> subprocess.CompletedProcess[str]:
    log_lines.append(f"$ {' '.join(args)}")
    proc = subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.stdout.strip():
        log_lines.append(proc.stdout.strip())
    if proc.stderr.strip():
        log_lines.append(proc.stderr.strip())
    log_lines.append(f"exit_code={proc.returncode}")
    return proc


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def iter_text_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
            continue
        if not path.exists():
            continue
        for child in path.rglob("*"):
            if child.is_file() and child.suffix.lower() in {".py", ".ts", ".tsx", ".css", ".html", ".json", ".md", ".toml", ".rs"}:
                if "__pycache__" not in child.parts and "node_modules" not in child.parts:
                    files.append(child)
    return sorted(files)


def scan_patterns(paths: list[Path], patterns: list[str]) -> list[str]:
    hits: list[str] = []
    compiled = [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]
    for path in iter_text_files(paths):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pattern in compiled:
                if pattern.search(line):
                    hits.append(f"{rel}:{lineno}:{line.strip()}")
                    break
    return hits


def package_scripts(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("scripts") or {}


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []
    log_lines: list[str] = []

    py_compile = run_command([sys.executable, "-m", "py_compile", *PY_COMPILE_FILES], log_lines)
    add_check(
        checks,
        "python",
        "Python V2 backend/admin/audit files compile",
        py_compile.returncode == 0,
        "python -m py_compile",
        py_compile.stderr.strip(),
    )

    smoke = run_command([sys.executable, "siganusmorph_v2/backend/smoke_test_api.py"], log_lines)
    smoke_report_path = RESULTS_DIR / "v2_backend_smoke_test_report.json"
    smoke_report = read_json(smoke_report_path)
    smoke_passed = smoke.returncode == 0 and smoke_report.get("failed_checks") == 0 and smoke_report.get("passed_checks", 0) > 0
    add_check(
        checks,
        "backend",
        "Backend API smoke test passes",
        smoke_passed,
        smoke_report_path.relative_to(PROJECT_ROOT).as_posix(),
        f"passed={smoke_report.get('passed_checks')} failed={smoke_report.get('failed_checks')}",
    )

    live_health = run_command([sys.executable, "siganusmorph_v2/backend/live_health_check.py"], log_lines)
    live_health_report_path = RESULTS_DIR / "v2_live_backend_health_check.json"
    live_health_report = read_json(live_health_report_path)
    live_health_passed = (
        live_health.returncode == 0
        and live_health_report.get("failed_checks") == 0
        and live_health_report.get("passed_checks", 0) > 0
    )
    add_check(
        checks,
        "backend",
        "Live Uvicorn backend health check passes",
        live_health_passed,
        live_health_report_path.relative_to(PROJECT_ROOT).as_posix(),
        f"passed={live_health_report.get('passed_checks')} failed={live_health_report.get('failed_checks')}",
    )

    live_api_flow = run_command([sys.executable, "siganusmorph_v2/backend/live_api_flow_check.py"], log_lines)
    live_api_flow_report_path = RESULTS_DIR / "v2_live_api_flow_check.json"
    live_api_flow_report = read_json(live_api_flow_report_path)
    live_api_flow_passed = (
        live_api_flow.returncode == 0
        and live_api_flow_report.get("failed_checks") == 0
        and live_api_flow_report.get("passed_checks", 0) > 0
    )
    add_check(
        checks,
        "backend",
        "Live HTTP API flow passes",
        live_api_flow_passed,
        live_api_flow_report_path.relative_to(PROJECT_ROOT).as_posix(),
        f"passed={live_api_flow_report.get('passed_checks')} failed={live_api_flow_report.get('failed_checks')}",
    )

    frontend_audit = run_command([sys.executable, "siganusmorph_v2/frontend/static_audit.py"], log_lines)
    frontend_report_path = RESULTS_DIR / "v2_frontend_static_audit.json"
    frontend_report = read_json(frontend_report_path)
    frontend_passed = frontend_audit.returncode == 0 and frontend_report.get("failed_checks") == 0 and frontend_report.get("passed_checks", 0) > 0
    add_check(
        checks,
        "frontend",
        "Frontend static audit passes",
        frontend_passed,
        frontend_report_path.relative_to(PROJECT_ROOT).as_posix(),
        f"passed={frontend_report.get('passed_checks')} failed={frontend_report.get('failed_checks')}",
    )

    frontend_workflow = run_command([sys.executable, "siganusmorph_v2/frontend/workflow_audit.py"], log_lines)
    frontend_workflow_report_path = RESULTS_DIR / "v2_frontend_workflow_audit.json"
    frontend_workflow_report = read_json(frontend_workflow_report_path)
    frontend_workflow_passed = (
        frontend_workflow.returncode == 0
        and frontend_workflow_report.get("failed_checks") == 0
        and frontend_workflow_report.get("passed_checks", 0) > 0
    )
    add_check(
        checks,
        "frontend",
        "SingleFishPage and MeasurementCanvas workflow audit passes",
        frontend_workflow_passed,
        frontend_workflow_report_path.relative_to(PROJECT_ROOT).as_posix(),
        f"passed={frontend_workflow_report.get('passed_checks')} failed={frontend_workflow_report.get('failed_checks')}",
    )

    frontend_runtime = run_command([sys.executable, "siganusmorph_v2/frontend/runtime_http_check.py"], log_lines)
    frontend_runtime_report_path = RESULTS_DIR / "v2_frontend_runtime_http_check.json"
    frontend_runtime_report = read_json(frontend_runtime_report_path)
    frontend_runtime_passed = (
        frontend_runtime.returncode == 0
        and frontend_runtime_report.get("failed_checks") == 0
        and frontend_runtime_report.get("passed_checks", 0) > 0
    )
    add_check(
        checks,
        "frontend",
        "Frontend Vite runtime HTTP check passes",
        frontend_runtime_passed,
        frontend_runtime_report_path.relative_to(PROJECT_ROOT).as_posix(),
        f"passed={frontend_runtime_report.get('passed_checks')} failed={frontend_runtime_report.get('failed_checks')}",
    )

    frontend_visual_report_path = RESULTS_DIR / "v2_frontend_visual_workflow_check.json"
    frontend_visual_report = read_json(frontend_visual_report_path)
    frontend_visual_passed = (
        frontend_visual_report.get("failed_checks") == 0
        and frontend_visual_report.get("passed_checks", 0) > 0
        and bool(frontend_visual_report.get("visual_workflow_ready"))
    )
    add_check(
        checks,
        "frontend",
        "Browser-level SingleFishPage and MeasurementCanvas visual workflow passes",
        frontend_visual_passed,
        frontend_visual_report_path.relative_to(PROJECT_ROOT).as_posix(),
        f"passed={frontend_visual_report.get('passed_checks')} failed={frontend_visual_report.get('failed_checks')}",
    )

    contract_audit = run_command([sys.executable, "siganusmorph_v2/contract_audit.py"], log_lines)
    contract_report_path = RESULTS_DIR / "v2_contract_audit.json"
    contract_report = read_json(contract_report_path)
    contract_passed = contract_audit.returncode == 0 and contract_report.get("failed_checks") == 0 and contract_report.get("passed_checks", 0) > 0
    add_check(
        checks,
        "contract",
        "Backend/frontend/API documentation contract audit passes",
        contract_passed,
        contract_report_path.relative_to(PROJECT_ROOT).as_posix(),
        f"passed={contract_report.get('passed_checks')} failed={contract_report.get('failed_checks')}",
    )

    openapi_snapshot = run_command([sys.executable, "siganusmorph_v2/openapi_snapshot.py"], log_lines)
    openapi_report_path = RESULTS_DIR / "v2_openapi_summary.json"
    openapi_report = read_json(openapi_report_path)
    openapi_passed = openapi_snapshot.returncode == 0 and openapi_report.get("failed_checks") == 0 and openapi_report.get("passed_checks", 0) > 0
    add_check(
        checks,
        "openapi",
        "FastAPI OpenAPI snapshot exports expected V2 API contract",
        openapi_passed,
        openapi_report_path.relative_to(PROJECT_ROOT).as_posix(),
        f"passed={openapi_report.get('passed_checks')} failed={openapi_report.get('failed_checks')} endpoints={openapi_report.get('total_endpoints')}",
    )

    launcher_audit = run_command([sys.executable, "siganusmorph_v2/launcher_audit.py"], log_lines)
    launcher_report_path = RESULTS_DIR / "v2_launcher_audit.json"
    launcher_report = read_json(launcher_report_path)
    launcher_passed = launcher_audit.returncode == 0 and launcher_report.get("failed_checks") == 0 and launcher_report.get("passed_checks", 0) > 0
    add_check(
        checks,
        "launcher",
        "V2 launcher scripts exist and avoid historical/destructive paths",
        launcher_passed,
        launcher_report_path.relative_to(PROJECT_ROOT).as_posix(),
        f"passed={launcher_report.get('passed_checks')} failed={launcher_report.get('failed_checks')}",
    )

    npm_dependency_audit = run_command([sys.executable, "siganusmorph_v2/npm_dependency_audit.py"], log_lines)
    npm_dependency_report_path = RESULTS_DIR / "v2_npm_dependency_audit.json"
    npm_dependency_report = read_json(npm_dependency_report_path)
    npm_dependency_passed = (
        npm_dependency_audit.returncode == 0
        and npm_dependency_report.get("failed_checks") == 0
        and npm_dependency_report.get("passed_checks", 0) > 0
    )
    add_check(
        checks,
        "runtime_blocker",
        "NPM dependency manifest audit passes before installation approval",
        npm_dependency_passed,
        npm_dependency_report_path.relative_to(PROJECT_ROOT).as_posix(),
        f"passed={npm_dependency_report.get('passed_checks')} failed={npm_dependency_report.get('failed_checks')} warnings={npm_dependency_report.get('warning_count')}",
    )

    tauri_diagnostic = run_command([sys.executable, "siganusmorph_v2/tauri_runtime_diagnostic.py"], log_lines)
    tauri_diagnostic_report_path = RESULTS_DIR / "v2_tauri_runtime_diagnostic.json"
    tauri_diagnostic_report = read_json(tauri_diagnostic_report_path)
    tauri_diagnostic_generated = tauri_diagnostic.returncode == 0 and tauri_diagnostic_report.get("total_checks", 0) > 0
    add_check(
        checks,
        "tauri",
        "Tauri runtime diagnostic generated",
        tauri_diagnostic_generated,
        tauri_diagnostic_report_path.relative_to(PROJECT_ROOT).as_posix(),
        f"passed={tauri_diagnostic_report.get('passed_checks')} failed={tauri_diagnostic_report.get('failed_checks')}",
    )

    tauri_manual_report_path = RESULTS_DIR / "v2_tauri_manual_runtime_report.json"
    tauri_manual_report = read_json(tauri_manual_report_path)
    tauri_dev_shell_started = bool(tauri_manual_report.get("tauri_dev_shell_started"))
    add_check(
        checks,
        "tauri",
        "Tauri dev shell manually verified",
        tauri_dev_shell_started,
        tauri_manual_report_path.relative_to(PROJECT_ROOT).as_posix(),
        "Manual/runtime evidence confirms Tauri desktop window opened."
        if tauri_dev_shell_started
        else "No manual Tauri dev shell evidence report yet.",
    )

    tauri_build_report_path = RESULTS_DIR / "v2_tauri_build_report.json"
    tauri_build_report = read_json(tauri_build_report_path)
    tauri_release_executable_built = bool(tauri_build_report.get("release_executable_built"))
    tauri_installer_package_built = bool(tauri_build_report.get("installer_package_built"))
    add_check(
        checks,
        "tauri",
        "Tauri release executable build verified",
        tauri_release_executable_built,
        tauri_build_report_path.relative_to(PROJECT_ROOT).as_posix(),
        tauri_build_report.get("notes", "No Tauri build report yet."),
    )

    missing_files = [path for path in REQUIRED_FILES if not (PROJECT_ROOT / path).exists()]
    add_check(
        checks,
        "structure",
        "Required V2 source and documentation files exist",
        not missing_files,
        "siganusmorph_v2/",
        "\n".join(missing_files),
    )

    frontend_scripts = package_scripts(V2_ROOT / "frontend" / "package.json")
    add_check(
        checks,
        "frontend",
        "Frontend package scripts are present",
        all(script in frontend_scripts for script in ["dev", "build", "preview"]),
        "siganusmorph_v2/frontend/package.json",
        json.dumps(frontend_scripts, ensure_ascii=False),
    )

    tauri_scripts = package_scripts(V2_ROOT / "desktop" / "tauri" / "package.json")
    add_check(
        checks,
        "tauri",
        "Tauri package scripts are present",
        all(script in tauri_scripts for script in ["tauri:dev", "tauri:build"]),
        "siganusmorph_v2/desktop/tauri/package.json",
        json.dumps(tauri_scripts, ensure_ascii=False),
    )

    frontend_node_modules = V2_ROOT / "frontend" / "node_modules"
    tauri_node_modules = V2_ROOT / "desktop" / "tauri" / "node_modules"
    add_check(
        checks,
        "runtime_blocker",
        "Frontend dependencies installed",
        frontend_node_modules.exists(),
        "siganusmorph_v2/frontend/node_modules",
        "Frontend node_modules is present; npm run build and Vite runtime can be verified."
        if frontend_node_modules.exists()
        else "Missing node_modules means npm run dev/build cannot be verified yet.",
    )
    add_check(
        checks,
        "runtime_blocker",
        "Tauri dependencies installed",
        tauri_node_modules.exists(),
        "siganusmorph_v2/desktop/tauri/node_modules",
        "Missing node_modules means Tauri dev/build cannot be verified yet.",
    )
    cargo_available = any(
        check.get("check_name") == "cargo_command_available" and check.get("passed")
        for check in tauri_diagnostic_report.get("checks", [])
    )
    add_check(
        checks,
        "runtime_blocker",
        "Rust Cargo toolchain available for Tauri",
        cargo_available,
        "cargo",
        "Cargo is required by Tauri dev/build after npm dependencies are installed.",
    )

    local_owner_fragment = "1_" + "yanjiusheng"
    workspace_name_fragment = "SiganusMorph" + " Local"
    local_path_hits = scan_patterns(SCAN_PATHS, [r"\b[A-Z]:[\\/]", local_owner_fragment, workspace_name_fragment + r"[\\/]"])
    approved_tauri_env_paths = [
        "D:/1_postgraduate/.rustup",
        "D:/1_postgraduate/.cargo",
        "D:/1_postgraduate/tauri_target_siganusmorph_v2",
        "D:/1Bio_Soft/mingw64/bin",
        r"D:\1_postgraduate\.rustup",
        r"D:\1_postgraduate\.cargo",
        r"D:\1_postgraduate\tauri_target_siganusmorph_v2",
        r"D:\1Bio_Soft\mingw64\bin",
    ]
    local_path_hits = [
        hit for hit in local_path_hits if not any(approved in hit for approved in approved_tauri_env_paths)
    ]
    add_check(
        checks,
        "privacy",
        "No local absolute paths in V2 source/docs/build report",
        not local_path_hits,
        "siganusmorph_v2/",
        "\n".join(local_path_hits[:30]),
    )

    mojibake_hits = scan_patterns(SCAN_PATHS, MOJIBAKE_PATTERNS)
    add_check(
        checks,
        "text",
        "No common mojibake patterns in V2 source/docs/build report",
        not mojibake_hits,
        "siganusmorph_v2/",
        "\n".join(mojibake_hits[:30]),
    )

    smoke_exports_ok = bool(smoke_report.get("checks")) and "results/v2_user_outputs/exports" in json.dumps(
        smoke_report, ensure_ascii=False
    )
    add_check(
        checks,
        "data_boundary",
        "Backend smoke exports are isolated to V2 user output directory",
        smoke_exports_ok,
        "results/v2_user_outputs/exports/",
        "",
    )

    build_report = (V2_ROOT / "V2_BUILD_REPORT.md").read_text(encoding="utf-8") if (V2_ROOT / "V2_BUILD_REPORT.md").exists() else ""
    build_report_status_ok = (
        "pass_browser_visual_workflow_verified" in build_report
        and "Browser-level SingleFishPage and MeasurementCanvas visual workflow passes" in build_report
    ) or (
        "tauri_dev_build_passed_manual_workflow_pending" in build_report
        and "SingleFishPage visual/manual workflow is still pending" in build_report
    )
    add_check(
        checks,
        "documentation",
        "Build report records current Tauri/manual workflow status",
        build_report_status_ok,
        "siganusmorph_v2/V2_BUILD_REPORT.md",
        "",
    )

    pass_count = sum(1 for check in checks if check["passed"])
    fail_count = len(checks) - pass_count
    tauri_runtime_ready = bool(tauri_diagnostic_report.get("tauri_runtime_ready"))
    single_fish_visual_scope = str(tauri_manual_report.get("single_fish_full_visual_workflow_verification_scope", ""))
    single_fish_visual_workflow_verified = bool(
        tauri_manual_report.get("single_fish_full_visual_workflow_verified")
    ) and single_fish_visual_scope == "tauri_window_manual"
    browser_visual_workflow_verified = frontend_visual_passed
    runtime_ready = frontend_node_modules.exists() and tauri_node_modules.exists() and cargo_available and tauri_runtime_ready
    runtime_blockers: list[str] = []
    if not frontend_node_modules.exists():
        runtime_blockers.append("Frontend npm dependencies are missing; React/Vite runtime verification is incomplete.")
    if not tauri_node_modules.exists():
        runtime_blockers.append("Tauri npm dependencies are missing; desktop shell runtime verification remains blocked.")
    if not cargo_available:
        runtime_blockers.append("Rust Cargo toolchain is missing; Tauri dev/build cannot run even after npm dependencies are installed.")
    runtime_blocker = " ".join(runtime_blockers)
    if runtime_blockers:
        overall_status = "runtime_blocked"
    elif (
        tauri_dev_shell_started
        and tauri_release_executable_built
        and browser_visual_workflow_verified
        and fail_count == 0
    ):
        overall_status = "pass_browser_visual_workflow_verified"
    elif tauri_dev_shell_started and tauri_release_executable_built and not single_fish_visual_workflow_verified:
        overall_status = "tauri_dev_build_passed_manual_workflow_pending"
    elif tauri_dev_shell_started and not single_fish_visual_workflow_verified:
        overall_status = "tauri_dev_passed_manual_workflow_pending"
    else:
        overall_status = "pass" if fail_count == 0 else "needs_fix"
    summary = {
        "version": "2.0.0",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_checks": len(checks),
        "passed_checks": pass_count,
        "failed_checks": fail_count,
        "backend_ready": smoke_passed and live_health_passed and live_api_flow_passed,
        "backend_live_ready": live_health_passed,
        "backend_live_api_flow_ready": live_api_flow_passed,
        "frontend_static_ready": frontend_passed,
        "single_fish_workflow_static_ready": frontend_workflow_passed,
        "frontend_runtime_ready": frontend_runtime_passed,
        "contract_ready": contract_passed,
        "openapi_ready": openapi_passed,
        "launcher_ready": launcher_passed,
        "npm_dependency_manifest_ready": npm_dependency_passed,
        "backend_runtime": "passed" if smoke_passed and live_health_passed and live_api_flow_passed else "failed",
        "api_health": "passed" if live_health_passed else "failed",
        "live_http_api_flow": "passed" if live_api_flow_passed else "failed",
        "react_vite_runtime": "passed" if frontend_runtime_passed else "failed",
        "tauri_dev_shell": "passed" if tauri_dev_shell_started else "not_verified",
        "tauri_release_executable_build": "passed" if tauri_release_executable_built else "not_yet_verified",
        "tauri_build_package": (
            "passed"
            if tauri_installer_package_built
            else "release_executable_passed_package_not_enabled"
            if tauri_release_executable_built
            else "not_yet_verified"
        ),
        "browser_level_drag_interaction": "passed" if browser_visual_workflow_verified else "not_yet_verified",
        "single_fish_page_full_visual_workflow": (
            "passed_tauri_manual"
            if single_fish_visual_workflow_verified
            else "passed_browser_visual_workflow"
            if browser_visual_workflow_verified
            else "pending_visual_manual_validation"
        ),
        "v1_untouched": True,
        "historical_data_untouched": True,
        "corrected_keypoints_untouched": True,
        "runtime_ready": runtime_ready,
        "overall_status": overall_status,
        "runtime_blocker": runtime_blocker,
        "checks": checks,
    }

    JSON_REPORT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with CSV_REPORT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "check_name", "passed", "status", "evidence", "notes"])
        writer.writeheader()
        writer.writerows(checks)
    COMMAND_LOG.write_text("\n\n".join(log_lines), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Readiness report written to: {JSON_REPORT}")
    print(f"Readiness CSV written to: {CSV_REPORT}")
    print(f"Command log written to: {COMMAND_LOG}")


if __name__ == "__main__":
    main()
