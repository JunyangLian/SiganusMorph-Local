from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results" / "v2_user_outputs"
JSON_REPORT = RESULTS_DIR / "v2_npm_dependency_audit.json"
CSV_REPORT = RESULTS_DIR / "v2_npm_dependency_audit.csv"


PACKAGE_FILES = [
    V2_ROOT / "frontend" / "package.json",
    V2_ROOT / "desktop" / "tauri" / "package.json",
]

ALLOWED_PACKAGES = {
    "@vitejs/plugin-react",
    "vite",
    "typescript",
    "react",
    "react-dom",
    "@tauri-apps/cli",
}

RISKY_SCRIPT_NAMES = {
    "preinstall",
    "install",
    "postinstall",
    "prepublish",
    "prepare",
    "prepack",
    "postpack",
}


def _rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _add_check(
    checks: list[dict[str, Any]],
    package_file: Path,
    name: str,
    passed: bool,
    evidence: str = "",
    notes: str = "",
    severity: str = "error",
) -> None:
    checks.append(
        {
            "package_file": _rel(package_file),
            "check_name": name,
            "passed": bool(passed),
            "status": "pass" if passed else "fail",
            "severity": severity,
            "evidence": evidence,
            "notes": notes,
        }
    )


def _version_is_exact(version: str) -> bool:
    return bool(re.fullmatch(r"\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?", version.strip()))


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    install_state: dict[str, bool] = {}

    for package_file in PACKAGE_FILES:
        exists = package_file.exists()
        _add_check(checks, package_file, "package_json_exists", exists, _rel(package_file) if exists else "", "")
        if not exists:
            continue

        package = json.loads(package_file.read_text(encoding="utf-8"))
        scripts = package.get("scripts") or {}
        dependencies = package.get("dependencies") or {}
        dev_dependencies = package.get("devDependencies") or {}
        all_dependencies = {**dependencies, **dev_dependencies}

        risky_scripts = sorted(name for name in scripts if name in RISKY_SCRIPT_NAMES)
        _add_check(
            checks,
            package_file,
            "no_local_npm_lifecycle_scripts",
            not risky_scripts,
            ", ".join(risky_scripts),
            "preinstall/install/postinstall/prepare scripts would run during npm install.",
        )

        unknown_dependencies = sorted(name for name in all_dependencies if name not in ALLOWED_PACKAGES)
        _add_check(
            checks,
            package_file,
            "dependency_names_match_v2_allowlist",
            not unknown_dependencies,
            ", ".join(sorted(all_dependencies)),
            f"Unknown dependencies: {', '.join(unknown_dependencies)}" if unknown_dependencies else "",
        )

        range_dependencies = sorted(f"{name}@{version}" for name, version in all_dependencies.items() if not _version_is_exact(str(version)))
        warnings.append(
            {
                "package_file": _rel(package_file),
                "warning_name": "non_exact_dependency_versions",
                "count": len(range_dependencies),
                "evidence": ", ".join(range_dependencies),
                "notes": "Current package.json uses semver ranges. A reviewed package-lock should be generated before runtime verification.",
            }
        )

        lockfile = package_file.with_name("package-lock.json")
        package_key = "frontend" if "frontend" in package_file.parts else "tauri"
        install_state[f"{package_key}_package_lock_present"] = lockfile.exists()
        warnings.append(
            {
                "package_file": _rel(package_file),
                "warning_name": "package_lock_missing" if not lockfile.exists() else "package_lock_present",
                "count": 0 if lockfile.exists() else 1,
                "evidence": _rel(lockfile),
                "notes": "A package-lock file should be produced only after explicit npm dependency-install approval.",
            }
        )

        node_modules = package_file.parent / "node_modules"
        install_state[f"{package_key}_node_modules_present"] = node_modules.exists()
        warnings.append(
            {
                "package_file": _rel(package_file),
                "warning_name": "node_modules_missing" if not node_modules.exists() else "node_modules_present",
                "count": 0 if node_modules.exists() else 1,
                "evidence": _rel(node_modules),
                "notes": "Missing node_modules is expected until npm installation is explicitly approved.",
            }
        )

    failed = sum(1 for check in checks if not check["passed"])
    passed = len(checks) - failed
    frontend_installed = install_state.get("frontend_node_modules_present", False) and install_state.get("frontend_package_lock_present", False)
    tauri_installed = install_state.get("tauri_node_modules_present", False) and install_state.get("tauri_package_lock_present", False)
    if frontend_installed and tauri_installed:
        runtime_dependency_status = "frontend_and_tauri_installed"
    elif frontend_installed:
        runtime_dependency_status = "frontend_installed_tauri_missing"
    elif tauri_installed:
        runtime_dependency_status = "tauri_installed_frontend_missing"
    else:
        runtime_dependency_status = "not_installed"
    report = {
        "version": "2.0.0",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_checks": len(checks),
        "passed_checks": passed,
        "failed_checks": failed,
        "warning_count": sum(int(warning.get("count") or 0) for warning in warnings),
        "runtime_dependency_status": runtime_dependency_status,
        "frontend_dependencies_installed": frontend_installed,
        "tauri_dependencies_installed": tauri_installed,
        "install_requires_explicit_approval": True,
        "recommended_review_command": "npm install only after explicit approval; prefer reviewing package-lock before runtime verification.",
        "checks": checks,
        "warnings": warnings,
    }

    JSON_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with CSV_REPORT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["package_file", "check_name", "passed", "status", "severity", "evidence", "notes"])
        writer.writeheader()
        writer.writerows(checks)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"NPM dependency audit report written to: {_rel(JSON_REPORT)}")
    print(f"NPM dependency audit CSV written to: {_rel(CSV_REPORT)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
