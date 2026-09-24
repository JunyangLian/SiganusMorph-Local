from __future__ import annotations

import csv
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


VERSION = "2.0.0"
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
TAURI_ROOT = ROOT / "desktop" / "tauri"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "v2_user_outputs"
RUSTUP_HOME = Path("D:/1_postgraduate/.rustup")
CARGO_HOME = Path("D:/1_postgraduate/.cargo")
CARGO_BIN = CARGO_HOME / "bin" / "cargo.exe"
RUSTC_BIN = CARGO_HOME / "bin" / "rustc.exe"
MINGW_BIN = Path("D:/1Bio_Soft/mingw64/bin")
GCC_BIN = MINGW_BIN / "gcc.exe"


def _sanitize_path(value: str) -> str:
    text = value.replace("\\", "/")
    root_text = str(PROJECT_ROOT).replace("\\", "/")
    return text.replace(root_text, "<project_root>")


def _run_readonly(command: list[str], cwd: Path) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
        return {
            "exit_code": proc.returncode,
            "stdout": _sanitize_path(proc.stdout.strip()),
            "stderr": _sanitize_path(proc.stderr.strip()),
        }
    except Exception as exc:  # pragma: no cover - diagnostic only
        return {"exit_code": None, "stdout": "", "stderr": str(exc)}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    package_path = TAURI_ROOT / "package.json"
    config_path = TAURI_ROOT / "src-tauri" / "tauri.conf.json"
    cargo_path = TAURI_ROOT / "src-tauri" / "Cargo.toml"
    node_modules_path = TAURI_ROOT / "node_modules"
    lock_path = TAURI_ROOT / "package-lock.json"

    package = _load_json(package_path)
    config = _load_json(config_path)
    npm_path = shutil.which("npm")
    node_path = shutil.which("node")
    cargo_path_bin = shutil.which("cargo") or (str(CARGO_BIN) if CARGO_BIN.exists() else None)
    rustc_path_bin = shutil.which("rustc") or (str(RUSTC_BIN) if RUSTC_BIN.exists() else None)
    gcc_path_bin = shutil.which("gcc") or (str(GCC_BIN) if GCC_BIN.exists() else None)

    npm_registry = _run_readonly([npm_path, "config", "get", "registry"], TAURI_ROOT) if npm_path else {}
    npm_ls = _run_readonly([npm_path, "ls", "--depth=0", "--json"], TAURI_ROOT) if npm_path else {}
    cargo_version = _run_readonly([cargo_path_bin, "--version"], TAURI_ROOT) if cargo_path_bin else {}
    rustc_version = _run_readonly([rustc_path_bin, "--version"], TAURI_ROOT) if rustc_path_bin else {}
    gcc_version = _run_readonly([gcc_path_bin, "--version"], TAURI_ROOT) if gcc_path_bin else {}

    scripts = package.get("scripts", {}) if isinstance(package, dict) else {}
    dev_deps = package.get("devDependencies", {}) if isinstance(package, dict) else {}
    build = config.get("build", {}) if isinstance(config, dict) else {}
    windows = config.get("app", {}).get("windows", []) if isinstance(config, dict) else []
    window_title = windows[0].get("title") if windows else None

    checks = [
        {
            "check_name": "tauri_package_json_exists",
            "passed": package_path.exists(),
            "evidence": "siganusmorph_v2/desktop/tauri/package.json",
            "notes": "",
        },
        {
            "check_name": "tauri_conf_exists",
            "passed": config_path.exists(),
            "evidence": "siganusmorph_v2/desktop/tauri/src-tauri/tauri.conf.json",
            "notes": "",
        },
        {
            "check_name": "tauri_cargo_manifest_exists",
            "passed": cargo_path.exists(),
            "evidence": "siganusmorph_v2/desktop/tauri/src-tauri/Cargo.toml",
            "notes": "",
        },
        {
            "check_name": "tauri_dev_script_present",
            "passed": scripts.get("tauri:dev") == "tauri dev",
            "evidence": json.dumps(scripts, ensure_ascii=False),
            "notes": "",
        },
        {
            "check_name": "tauri_cli_dependency_declared",
            "passed": "@tauri-apps/cli" in dev_deps,
            "evidence": json.dumps(dev_deps, ensure_ascii=False),
            "notes": "",
        },
        {
            "check_name": "frontend_dev_url_points_to_local_vite",
            "passed": build.get("devUrl") == "http://127.0.0.1:5173",
            "evidence": str(build.get("devUrl")),
            "notes": "",
        },
        {
            "check_name": "tauri_window_title_utf8_ok",
            "passed": window_title == "蓝子鱼形态测量系统 V2.0",
            "evidence": str(window_title),
            "notes": "",
        },
        {
            "check_name": "node_command_available",
            "passed": bool(node_path),
            "evidence": "node",
            "notes": "",
        },
        {
            "check_name": "npm_command_available",
            "passed": bool(npm_path),
            "evidence": "npm",
            "notes": "",
        },
        {
            "check_name": "cargo_command_available",
            "passed": bool(cargo_path_bin),
            "evidence": _sanitize_path(str(cargo_path_bin or "cargo")),
            "notes": "Required by Tauri dev/build after npm dependencies are installed.",
        },
        {
            "check_name": "rustc_command_available",
            "passed": bool(rustc_path_bin),
            "evidence": _sanitize_path(str(rustc_path_bin or "rustc")),
            "notes": "Rust compiler used by Tauri dev/build.",
        },
        {
            "check_name": "mingw_gcc_available",
            "passed": bool(gcc_path_bin),
            "evidence": _sanitize_path(str(gcc_path_bin or "gcc")),
            "notes": "GNU host toolchain for x86_64-pc-windows-gnu.",
        },
        {
            "check_name": "tauri_node_modules_present",
            "passed": node_modules_path.exists(),
            "evidence": "siganusmorph_v2/desktop/tauri/node_modules",
            "notes": "Missing until Tauri npm install completes.",
        },
        {
            "check_name": "tauri_package_lock_present",
            "passed": lock_path.exists(),
            "evidence": "siganusmorph_v2/desktop/tauri/package-lock.json",
            "notes": "Missing until Tauri npm install completes.",
        },
    ]

    failed = [item for item in checks if not item["passed"]]
    report = {
        "version": VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_checks": len(checks),
        "passed_checks": len(checks) - len(failed),
        "failed_checks": len(failed),
        "tauri_runtime_ready": not failed,
        "rustup_home": _sanitize_path(str(RUSTUP_HOME)),
        "cargo_home": _sanitize_path(str(CARGO_HOME)),
        "rust_host_triple": "x86_64-pc-windows-gnu",
        "mingw_bin": _sanitize_path(str(MINGW_BIN)),
        "recommended_cargo_target_dir": "D:/1_postgraduate/tauri_target_siganusmorph_v2",
        "npm_registry": npm_registry,
        "npm_ls_depth0": npm_ls,
        "cargo_version": cargo_version,
        "rustc_version": rustc_version,
        "gcc_version": gcc_version,
        "checks": checks,
        "blocking_issue": (
            "Tauri npm dependencies are missing; npm install currently fails with network ECONNRESET."
            if not node_modules_path.exists()
            else ""
        ),
    }

    json_path = OUTPUT_ROOT / "v2_tauri_runtime_diagnostic.json"
    csv_path = OUTPUT_ROOT / "v2_tauri_runtime_diagnostic.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["check_name", "passed", "evidence", "notes"])
        writer.writeheader()
        writer.writerows(checks)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Tauri runtime diagnostic report written to: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Tauri runtime diagnostic CSV written to: {csv_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
