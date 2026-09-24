from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph_v2.backend.app import app  # noqa: E402


RESULTS_DIR = PROJECT_ROOT / "results" / "v2_user_outputs"
OPENAPI_JSON = RESULTS_DIR / "v2_openapi_snapshot.json"
SUMMARY_JSON = RESULTS_DIR / "v2_openapi_summary.json"
SUMMARY_CSV = RESULTS_DIR / "v2_openapi_endpoints.csv"


EXPECTED_ENDPOINTS = {
    ("GET", "/api/health"),
    ("GET", "/api/settings"),
    ("GET", "/api/session/{session_id}"),
    ("POST", "/api/upload"),
    ("POST", "/api/calibrate"),
    ("POST", "/api/measure"),
    ("POST", "/api/update-points"),
    ("POST", "/api/export"),
    ("GET", "/api/exports"),
}


EXPECTED_SCHEMAS = {
    "HealthResponse",
    "SettingsResponse",
    "UploadResponse",
    "CalibrateResponse",
    "MeasureResponse",
    "UpdatePointsResponse",
    "ExportResponse",
    "ExportListResponse",
    "SessionResponse",
}


def schema_ref(operation: dict[str, Any]) -> str:
    responses = operation.get("responses") or {}
    ok_response = responses.get("200") or {}
    content = ok_response.get("content") or {}
    json_content = content.get("application/json") or {}
    schema = json_content.get("schema") or {}
    ref = schema.get("$ref") or ""
    return ref.rsplit("/", 1)[-1] if ref else ""


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, evidence: str = "", notes: str = "") -> None:
    checks.append(
        {
            "check_name": name,
            "passed": bool(passed),
            "status": "pass" if passed else "fail",
            "evidence": evidence,
            "notes": notes,
        }
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    openapi = app.openapi()
    OPENAPI_JSON.write_text(json.dumps(openapi, ensure_ascii=False, indent=2), encoding="utf-8")

    rows: list[dict[str, str]] = []
    found_endpoints: set[tuple[str, str]] = set()
    for path, methods in sorted((openapi.get("paths") or {}).items()):
        for method, operation in sorted(methods.items()):
            method_upper = method.upper()
            if method_upper not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
                continue
            found_endpoints.add((method_upper, path))
            rows.append(
                {
                    "method": method_upper,
                    "path": path,
                    "operation_id": str(operation.get("operationId", "")),
                    "response_schema": schema_ref(operation),
                    "summary": str(operation.get("summary", "")),
                }
            )

    schemas = set(((openapi.get("components") or {}).get("schemas") or {}).keys())
    checks: list[dict[str, Any]] = []
    missing_endpoints = sorted(EXPECTED_ENDPOINTS - found_endpoints)
    add_check(
        checks,
        "Required API endpoints are present in OpenAPI",
        not missing_endpoints,
        "OpenAPI paths",
        "\n".join(f"{method} {path}" for method, path in missing_endpoints),
    )

    missing_schemas = sorted(EXPECTED_SCHEMAS - schemas)
    add_check(
        checks,
        "Required response schemas are present in OpenAPI",
        not missing_schemas,
        "OpenAPI components.schemas",
        "\n".join(missing_schemas),
    )

    add_check(
        checks,
        "OpenAPI version metadata is V2.0",
        (openapi.get("info") or {}).get("version") == "2.0.0",
        "OpenAPI info.version",
        str((openapi.get("info") or {}).get("version")),
    )

    measure_schema = ((openapi.get("components") or {}).get("schemas") or {}).get("MeasureResponse", {})
    coordinate_schema = ((measure_schema.get("properties") or {}).get("coordinate_space") or {})
    add_check(
        checks,
        "MeasureResponse constrains coordinate_space to warped_image",
        "warped_image" in json.dumps(coordinate_schema, ensure_ascii=False),
        "MeasureResponse.coordinate_space",
        json.dumps(coordinate_schema, ensure_ascii=False),
    )

    export_schema = ((openapi.get("components") or {}).get("schemas") or {}).get("ExportResponse", {})
    export_properties = export_schema.get("properties") or {}
    add_check(
        checks,
        "ExportResponse includes files and saved_to",
        "files" in export_properties and "saved_to" in export_properties,
        "ExportResponse properties",
        ", ".join(sorted(export_properties.keys())),
    )

    summary = {
        "version": "2.0.0",
        "openapi_file": OPENAPI_JSON.relative_to(PROJECT_ROOT).as_posix(),
        "total_endpoints": len(rows),
        "total_schemas": len(schemas),
        "total_checks": len(checks),
        "passed_checks": sum(1 for check in checks if check["passed"]),
        "failed_checks": sum(1 for check in checks if not check["passed"]),
        "checks": checks,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "path", "operation_id", "response_schema", "summary"])
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"OpenAPI snapshot written to: {OPENAPI_JSON}")
    print(f"OpenAPI endpoint CSV written to: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
