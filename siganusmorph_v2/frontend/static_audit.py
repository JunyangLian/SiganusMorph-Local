from __future__ import annotations

import csv
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_ROOT = Path(__file__).resolve().parent
SRC_ROOT = FRONTEND_ROOT / "src"
REPORT_DIR = PROJECT_ROOT / "results" / "v2_user_outputs"
JSON_REPORT = REPORT_DIR / "v2_frontend_static_audit.json"
CSV_REPORT = REPORT_DIR / "v2_frontend_static_audit.csv"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def has_all(text: str, needles: list[str]) -> bool:
    return all(needle in text for needle in needles)


def add_check(checks: list[dict[str, object]], name: str, passed: bool, evidence: str, details: str = "") -> None:
    checks.append(
        {
            "check_name": name,
            "passed": bool(passed),
            "evidence": evidence,
            "details": details,
        }
    )


def scan_for_patterns(paths: list[Path], patterns: list[str]) -> list[str]:
    hits: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        text = read(path)
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pattern in patterns:
                if re.search(pattern, line, flags=re.IGNORECASE):
                    hits.append(f"{path.relative_to(PROJECT_ROOT).as_posix()}:{lineno}:{line.strip()}")
    return hits


def frontend_source_files() -> list[Path]:
    return sorted(
        list(SRC_ROOT.glob("*.ts"))
        + list(SRC_ROOT.glob("*.tsx"))
        + list((SRC_ROOT / "api").glob("*.ts"))
        + list((SRC_ROOT / "components").glob("*.tsx"))
        + list((SRC_ROOT / "pages").glob("*.tsx"))
    )


def resolve_relative_import(base: Path, specifier: str) -> Path | None:
    candidate = (base.parent / specifier).resolve()
    suffixes = ["", ".ts", ".tsx", ".css", "/index.ts", "/index.tsx"]
    for suffix in suffixes:
        path = Path(str(candidate) + suffix)
        if path.exists():
            return path
    return None


def audit_imports(paths: list[Path], package_json_text: str) -> tuple[list[str], list[str]]:
    import_pattern = re.compile(r"""import(?:\s+[^'"]+\s+from\s+|\s*)['"]([^'"]+)['"]""")
    missing_relative: list[str] = []
    missing_external: list[str] = []
    package_data = json.loads(package_json_text)
    declared = set((package_data.get("dependencies") or {}).keys()) | set((package_data.get("devDependencies") or {}).keys())
    for path in paths:
        text = read(path)
        for match in import_pattern.finditer(text):
            specifier = match.group(1)
            if specifier.startswith("."):
                if resolve_relative_import(path, specifier) is None:
                    missing_relative.append(f"{path.relative_to(PROJECT_ROOT).as_posix()} -> {specifier}")
            elif not specifier.startswith("vite/"):
                package = specifier.split("/")[0] if not specifier.startswith("@") else "/".join(specifier.split("/")[:2])
                if package not in declared:
                    missing_external.append(f"{path.relative_to(PROJECT_ROOT).as_posix()} -> {specifier}")
    return missing_relative, missing_external


def main() -> None:
    checks: list[dict[str, object]] = []

    pages = {
        "HomePage": SRC_ROOT / "pages" / "HomePage.tsx",
        "SingleFishPage": SRC_ROOT / "pages" / "SingleFishPage.tsx",
        "BatchPage": SRC_ROOT / "pages" / "BatchPage.tsx",
        "ExportPage": SRC_ROOT / "pages" / "ExportPage.tsx",
        "SettingsPage": SRC_ROOT / "pages" / "SettingsPage.tsx",
    }
    for name, path in pages.items():
        add_check(checks, f"{name} exists", path.exists(), path.relative_to(PROJECT_ROOT).as_posix())

    app_text = read(SRC_ROOT / "App.tsx")
    page_component_text = "\n".join(
        [app_text]
        + [read(path) for path in pages.values()]
        + [read(path) for path in (SRC_ROOT / "components").glob("*.tsx")]
        + [read(SRC_ROOT / "api" / "client.ts")]
    )
    required_user_labels = [
        "蓝子鱼形态测量系统",
        "首页",
        "单鱼测量",
        "批量测量",
        "结果导出",
        "Excel",
        "设置",
        "开始测量",
        "样本编号",
        "样品重量",
        "校准失败",
        "测量完成",
        "应用修改",
        "清空列表",
        "处理失败",
    ]
    missing_user_labels = [label for label in required_user_labels if label not in page_component_text]
    add_check(
        checks,
        "Required Chinese user-facing labels are present",
        not missing_user_labels,
        "src/App.tsx + src/pages + src/components + src/api/client.ts",
        "\n".join(missing_user_labels),
    )

    api = read(SRC_ROOT / "api" / "client.ts")
    add_check(
        checks,
        "API client has required endpoints",
        has_all(api, ["/api/health", "/api/settings", "/api/session/", "/api/upload", "/api/calibrate", "/api/measure", "/api/update-points", "/api/export", "/api/exports"]),
        "src/api/client.ts",
    )
    add_check(
        checks,
        "API client normalizes backend errors",
        has_all(api, ["safeErrorMessage", "Traceback", "处理失败"]),
        "src/api/client.ts",
        "Traceback is filtered in code and should not be displayed to ordinary users.",
    )

    tokens = read(SRC_ROOT / "styles" / "tokens.css")
    required_tokens = ["#f5f5f7", "#ffffff", "#1d1d1f", "#707070", "#0071e3", "#0066cc", "#e8e8ed", "28px", "36px"]
    add_check(checks, "Apple-inspired tokens implemented", has_all(tokens, required_tokens), "src/styles/tokens.css")

    canvas = read(SRC_ROOT / "components" / "MeasurementCanvas.tsx")
    draggable_points = [
        "P1_snout_tip",
        "P2_eye_anterior",
        "P3_operculum_posterior",
        "P4_peduncle_anterior",
        "P5_caudal_base",
        "P6_caudal_fork",
        "P7U_upper_lobe_tip",
        "P7L_lower_lobe_tip",
        "body_depth_upper",
        "body_depth_lower",
        "peduncle_depth_upper",
        "peduncle_depth_lower",
    ]
    add_check(checks, "MeasurementCanvas exposes required draggable points", has_all(canvas, draggable_points), "src/components/MeasurementCanvas.tsx")
    add_check(checks, "P7V is derived in frontend preview", has_all(canvas, ["deriveP7V", "P7V", "polygon"]), "src/components/MeasurementCanvas.tsx")
    add_check(checks, "Display to image coordinate mapping present", has_all(canvas, ["eventToImage", "getBoundingClientRect", "imageWidth", "imageHeight"]), "src/components/MeasurementCanvas.tsx")
    add_check(checks, "Large invisible hit radius logic present", has_all(canvas, ["nearestPoint", "hitRadius", "18 *"]), "src/components/MeasurementCanvas.tsx")
    add_check(checks, "Hover label and crosshair present", has_all(canvas, ["hover", "crosshair", "x=", "y="]), "src/components/MeasurementCanvas.tsx")
    add_check(checks, "Magnifier present", has_all(canvas, ["magnifier", "backgroundImage", "backgroundPosition"]), "src/components/MeasurementCanvas.tsx")
    add_check(checks, "Keyboard fine tuning present", has_all(canvas, ["handleKeyDown", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "shiftKey"]), "src/components/MeasurementCanvas.tsx")
    add_check(checks, "Apply changes callback present", has_all(canvas, ["onApply", "应用修改", "localPoints"]), "src/components/MeasurementCanvas.tsx")

    single = read(pages["SingleFishPage"])
    add_check(
        checks,
        "SingleFishPage performs upload-calibrate-measure flow",
        has_all(single, ["uploadImage", "calibrate", "measure", "cal.status !== \"success\"", "MeasurementCanvas"]),
        "src/pages/SingleFishPage.tsx",
    )
    add_check(
        checks,
        "SingleFishPage can restore existing V2 sessions",
        has_all(single, ["getSession", "restoreSession", "恢复会话", "setCalibration", "setMeasurement"]),
        "src/pages/SingleFishPage.tsx",
    )
    add_check(
        checks,
        "SingleFishPage blocks measurement when calibration fails",
        has_all(single, ["cal.status !== \"success\"", "return;"]),
        "src/pages/SingleFishPage.tsx",
    )

    batch = read(pages["BatchPage"])
    add_check(
        checks,
        "BatchPage calls backend queue steps",
        has_all(batch, ["uploadImage", "calibrate", "measure", "exportSession", "calibration_failed"]),
        "src/pages/BatchPage.tsx",
    )

    export_page = read(pages["ExportPage"])
    add_check(
        checks,
        "ExportPage reads backend exports with local fallback",
        has_all(export_page, ["listExports", "loadExportHistory", "saved_to", "files", "清空列表"]),
        "src/pages/ExportPage.tsx",
    )

    settings_page = read(pages["SettingsPage"])
    add_check(
        checks,
        "SettingsPage reads backend settings",
        has_all(settings_page, ["getSettings", "model_paths", "user_output_root", "admin_output_root"]),
        "src/pages/SettingsPage.tsx",
    )

    package_json = read(FRONTEND_ROOT / "package.json")
    add_check(checks, "Frontend package scripts present", has_all(package_json, ['"dev"', '"build"', '"preview"']), "frontend/package.json")

    source_files = frontend_source_files()
    missing_relative, missing_external = audit_imports(source_files, package_json)
    add_check(
        checks,
        "Frontend relative imports resolve",
        not missing_relative,
        "src/**/*.ts(x)",
        "\n".join(missing_relative),
    )
    add_check(
        checks,
        "Frontend external imports are declared",
        not missing_external,
        "frontend/package.json",
        "\n".join(missing_external),
    )

    css = read(SRC_ROOT / "styles" / "global.css")
    required_classes = [
        ".app-shell",
        ".sidebar",
        ".brand-title",
        ".brand-subtitle",
        ".nav",
        ".main",
        ".hero",
        ".eyebrow",
        ".grid",
        ".card",
        ".section-title",
        ".muted",
        ".field",
        ".button-row",
        ".btn",
        ".status",
        ".metrics",
        ".metric",
        ".metric-label",
        ".metric-value",
        ".image-box",
        ".table",
        ".canvas-wrap",
        ".measurement-svg",
        ".crosshair",
        ".magnifier",
        ".error-box",
        ".success-box",
    ]
    missing_classes = [class_name for class_name in required_classes if class_name not in css]
    add_check(checks, "Required CSS classes are defined", not missing_classes, "src/styles/global.css", "\n".join(missing_classes))

    user_facing_paths = [SRC_ROOT / "App.tsx"] + list((SRC_ROOT / "pages").glob("*.tsx")) + list((SRC_ROOT / "components").glob("*.tsx")) + list((SRC_ROOT / "styles").glob("*.css"))
    local_owner_fragment = "1_" + "yanjiusheng"
    workspace_name_fragment = "SiganusMorph" + " Local"
    local_path_hits = scan_for_patterns(
        user_facing_paths + [SRC_ROOT / "api" / "client.ts"],
        [r"\b[A-Z]:[\\/]", local_owner_fragment, workspace_name_fragment + r"[\\/]"],
    )
    add_check(checks, "No local absolute paths in frontend source", not local_path_hits, "src/", "\n".join(local_path_hits))

    user_label_hits = scan_for_patterns(user_facing_paths, [r"\bdebug\b", r"\baudit\b", r"\bexperimental\b", r"Traceback"])
    add_check(checks, "No debug/audit/experimental/traceback labels in user UI source", not user_label_hits, "src/pages + src/components + src/styles", "\n".join(user_label_hits))

    mojibake_hits = scan_for_patterns(
        user_facing_paths + [SRC_ROOT / "api" / "client.ts"],
        [r"鈥", r"鍗", r"瀵", r"涓", r"鏍", r"绛", r"娴", r"闇", r"钃", r"澶", r"搴", r"鎭", r"绮"],
    )
    add_check(
        checks,
        "No mojibake text in user-facing frontend source",
        not mojibake_hits,
        "src/pages + src/components + src/api",
        "\n".join(mojibake_hits),
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "version": "2.0.0",
        "total_checks": len(checks),
        "passed_checks": sum(1 for check in checks if check["passed"]),
        "failed_checks": sum(1 for check in checks if not check["passed"]),
        "checks": checks,
    }
    JSON_REPORT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with CSV_REPORT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["check_name", "passed", "evidence", "details"])
        writer.writeheader()
        writer.writerows(checks)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Frontend static audit written to: {JSON_REPORT}")
    print(f"Frontend static audit CSV written to: {CSV_REPORT}")


if __name__ == "__main__":
    main()
