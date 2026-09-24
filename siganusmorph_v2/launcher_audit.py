from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = Path(__file__).resolve().parent
SCRIPTS_ROOT = V2_ROOT / "scripts"
RESULTS_DIR = PROJECT_ROOT / "results" / "v2_user_outputs"
JSON_REPORT = RESULTS_DIR / "v2_launcher_audit.json"
CSV_REPORT = RESULTS_DIR / "v2_launcher_audit.csv"


EXPECTED_SCRIPTS = {
    "check_runtime_prereqs.ps1": ["node_modules", "Runtime status", "npm dependencies"],
    "run_backend.ps1": ["uvicorn", "siganusmorph_v2.backend.app:app", "127.0.0.1", "8000"],
    "run_frontend_dev.ps1": ["node_modules", "npm run dev", "127.0.0.1", "5173"],
    "run_tauri_dev.ps1": ["node_modules", "npm run tauri:dev", "127.0.0.1:8000"],
    "run_admin_console.ps1": ["streamlit run", "siganusmorph_v2/admin_streamlit/app.py"],
    "run_v2_readiness.ps1": ["verify_v2_readiness.py"],
}


FORBIDDEN_PATTERNS = [
    r"corrected_keypoints",
    r"final_analysis_dataset",
    r"batch_measurement_v0\.6\.4",
    r"Remove-Item",
    r"rm\s+",
    r"git\s+reset",
    r"\b[A-Z]:[\\/]",
]

APPROVED_TAURI_ENV_PATHS = [
    r"D:\1_postgraduate\.cargo",
    r"D:\1_postgraduate\.rustup",
    r"D:\1_postgraduate\tauri_target_siganusmorph_v2",
    r"D:\1Bio_Soft\mingw64\bin",
]


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, evidence: str, notes: str = "") -> None:
    checks.append(
        {
            "check_name": name,
            "passed": bool(passed),
            "status": "pass" if passed else "fail",
            "evidence": evidence,
            "notes": notes,
        }
    )


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []

    for filename, needles in EXPECTED_SCRIPTS.items():
        path = SCRIPTS_ROOT / filename
        add_check(
            checks,
            f"{filename} exists",
            path.exists(),
            path.relative_to(PROJECT_ROOT).as_posix(),
        )
        if not path.exists():
            continue
        text = read(path)
        missing = [needle for needle in needles if needle not in text]
        add_check(
            checks,
            f"{filename} contains expected command markers",
            not missing,
            path.relative_to(PROJECT_ROOT).as_posix(),
            "\n".join(missing),
        )
        forbidden_hits = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            if any(approved in line for approved in APPROVED_TAURI_ENV_PATHS):
                continue
            for pattern in FORBIDDEN_PATTERNS:
                if re.search(pattern, line, flags=re.IGNORECASE):
                    forbidden_hits.append(f"{lineno}: {line.strip()}")
                    break
        add_check(
            checks,
            f"{filename} avoids historical/destructive paths and commands",
            not forbidden_hits,
            path.relative_to(PROJECT_ROOT).as_posix(),
            "\n".join(forbidden_hits),
        )

    readme_path = SCRIPTS_ROOT / "README.md"
    readme = read(readme_path) if readme_path.exists() else ""
    add_check(
        checks,
        "Launcher README documents all scripts",
        all(filename in readme for filename in EXPECTED_SCRIPTS),
        readme_path.relative_to(PROJECT_ROOT).as_posix(),
    )

    summary = {
        "version": "2.0.0",
        "total_checks": len(checks),
        "passed_checks": sum(1 for check in checks if check["passed"]),
        "failed_checks": sum(1 for check in checks if not check["passed"]),
        "checks": checks,
    }
    JSON_REPORT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with CSV_REPORT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["check_name", "passed", "status", "evidence", "notes"])
        writer.writeheader()
        writer.writerows(checks)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Launcher audit written to: {JSON_REPORT}")
    print(f"Launcher audit CSV written to: {CSV_REPORT}")


if __name__ == "__main__":
    main()
