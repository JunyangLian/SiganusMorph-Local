from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results" / "v2_user_outputs"
JSON_REPORT = RESULTS_DIR / "v2_contract_audit.json"
CSV_REPORT = RESULTS_DIR / "v2_contract_audit.csv"

BACKEND_SCHEMA = V2_ROOT / "backend" / "schemas" / "measurement.py"
BACKEND_ROUTER = V2_ROOT / "backend" / "routers" / "api.py"
MEASUREMENT_SERVICE = V2_ROOT / "backend" / "services" / "measurement_service.py"
FRONTEND_CLIENT = V2_ROOT / "frontend" / "src" / "api" / "client.ts"
API_DOC = V2_ROOT / "docs" / "v2_api_contract.md"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def add_check(
    checks: list[dict[str, Any]],
    category: str,
    name: str,
    passed: bool,
    evidence: str,
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


def has_all(text: str, needles: list[str]) -> bool:
    return all(needle in text for needle in needles)


def missing(text: str, needles: list[str]) -> list[str]:
    return [needle for needle in needles if needle not in text]


def route_has_response_model(router_text: str, path: str, model: str) -> bool:
    pattern = re.compile(rf"@router\.(get|post)\(\s*['\"]{re.escape(path)}['\"].*?response_model\s*=\s*{model}", re.DOTALL)
    return bool(pattern.search(router_text))


def main() -> None:
    schema = read(BACKEND_SCHEMA)
    router = read(BACKEND_ROUTER)
    service = read(MEASUREMENT_SERVICE)
    client = read(FRONTEND_CLIENT)
    api_doc = read(API_DOC)
    checks: list[dict[str, Any]] = []

    schema_models = [
        "HealthResponse",
        "SettingsResponse",
        "UploadResponse",
        "CalibrateResponse",
        "MeasureResponse",
        "UpdatePointsResponse",
        "ExportResponse",
        "ExportListResponse",
        "SessionResponse",
    ]
    add_check(
        checks,
        "backend_schema",
        "Backend Pydantic response models exist",
        has_all(schema, [f"class {model}" for model in schema_models]),
        "siganusmorph_v2/backend/schemas/measurement.py",
        "\n".join(missing(schema, [f"class {model}" for model in schema_models])),
    )

    expected_routes = {
        "/health": "HealthResponse",
        "/settings": "SettingsResponse",
        "/session/{session_id}": "SessionResponse",
        "/upload": "UploadResponse",
        "/calibrate": "CalibrateResponse",
        "/measure": "MeasureResponse",
        "/update-points": "UpdatePointsResponse",
        "/export": "ExportResponse",
        "/exports": "ExportListResponse",
    }
    route_failures = [f"{path} -> {model}" for path, model in expected_routes.items() if not route_has_response_model(router, path, model)]
    add_check(
        checks,
        "backend_routes",
        "Backend routes declare response models",
        not route_failures,
        "siganusmorph_v2/backend/routers/api.py",
        "\n".join(route_failures),
    )

    frontend_types = [
        "HealthResponse",
        "SettingsResponse",
        "UploadResponse",
        "CalibrateResponse",
        "MeasureResponse",
        "ExportResponse",
        "ExportHistoryEntry",
        "ExportListResponse",
        "SessionResponse",
    ]
    add_check(
        checks,
        "frontend_types",
        "Frontend API response types exist",
        has_all(client, [f"type {name}" for name in frontend_types]),
        "siganusmorph_v2/frontend/src/api/client.ts",
        "\n".join(missing(client, [f"type {name}" for name in frontend_types])),
    )

    endpoint_needles = [
        "/api/health",
        "/api/settings",
        "/api/session/",
        "/api/upload",
        "/api/calibrate",
        "/api/measure",
        "/api/update-points",
        "/api/export",
        "/api/exports",
    ]
    add_check(
        checks,
        "frontend_api",
        "Frontend client calls all backend endpoints",
        has_all(client, endpoint_needles),
        "siganusmorph_v2/frontend/src/api/client.ts",
        "\n".join(missing(client, endpoint_needles)),
    )

    shared_fields = [
        "version",
        "session_id",
        "image_name",
        "specimen_id",
        "weight_g",
        "coordinate_space",
        "warped_image",
        "formal_points",
        "derived_points",
        "measurements",
        "qc",
        "files",
        "saved_to",
    ]
    shared_failures = []
    for field in shared_fields:
        if field not in schema or field not in client:
            shared_failures.append(field)
    add_check(
        checks,
        "field_alignment",
        "Shared backend/frontend response fields are aligned",
        not shared_failures,
        "schemas/measurement.py + src/api/client.ts",
        "\n".join(shared_failures),
    )

    add_check(
        checks,
        "coordinate_space",
        "Measurement coordinate space is explicitly warped_image in backend and frontend",
        has_all(schema, ['Literal["warped_image"]']) and has_all(client, ['coordinate_space: "warped_image"']),
        "schemas/measurement.py + src/api/client.ts",
        "",
    )

    add_check(
        checks,
        "calibration_gate",
        "Backend blocks measurement unless calibration succeeded",
        has_all(service, ['cal.get("status") != "success"', "Calibration has not succeeded", 'mm_per_pixel_source") == "fallback_default"', "warped_path.exists()"]),
        "siganusmorph_v2/backend/services/measurement_service.py",
        "",
    )

    add_check(
        checks,
        "export_gate",
        "Backend blocks export before measurement and warped coordinate-space readiness",
        has_all(service, ["require_measurement_ready", "Measurement has not produced formal points", 'coordinate_space") != "warped_image"']),
        "siganusmorph_v2/backend/services/measurement_service.py",
        "",
    )

    add_check(
        checks,
        "upload_metadata",
        "specimen_id and weight_g flow through upload, session restore and export",
        has_all(router + service + client, ["specimen_id", "weight_g", "form.append(\"weight_g\"", "rememberExport"]),
        "routers/api.py + measurement_service.py + src/api/client.ts",
        "",
    )

    add_check(
        checks,
        "manual_calibration",
        "Manual calibration corners are available but explicit",
        has_all(schema + service + client + api_doc, ["manual_corners", "manual_board_corners", "manual_corners"]),
        "schema + service + client + docs",
        "",
    )

    add_check(
        checks,
        "export_formats",
        "Export supports csv, xlsx, json and preview_png",
        has_all(service + client, ['"csv"', '"xlsx"', '"json"', '"preview_png"']),
        "measurement_service.py + src/api/client.ts",
        "",
    )

    documented_endpoints = [
        "GET /api/health",
        "GET /api/settings",
        "GET /api/session/{session_id}",
        "POST /api/upload",
        "POST /api/calibrate",
        "POST /api/measure",
        "POST /api/update-points",
        "POST /api/export",
        "GET /api/exports",
    ]
    add_check(
        checks,
        "documentation",
        "API contract documents all implemented endpoints",
        has_all(api_doc, documented_endpoints),
        "siganusmorph_v2/docs/v2_api_contract.md",
        "\n".join(missing(api_doc, documented_endpoints)),
    )

    add_check(
        checks,
        "user_safe_errors",
        "Frontend normalizes Python traceback-like backend errors",
        has_all(client, ["safeErrorMessage", "Traceback", "File "]),
        "siganusmorph_v2/frontend/src/api/client.ts",
        "",
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "version": "2.0.0",
        "total_checks": len(checks),
        "passed_checks": sum(1 for check in checks if check["passed"]),
        "failed_checks": sum(1 for check in checks if not check["passed"]),
        "checks": checks,
    }
    JSON_REPORT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with CSV_REPORT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "check_name", "passed", "status", "evidence", "notes"])
        writer.writeheader()
        writer.writerows(checks)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Contract audit written to: {JSON_REPORT}")
    print(f"Contract audit CSV written to: {CSV_REPORT}")


if __name__ == "__main__":
    main()
