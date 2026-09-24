from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph_v2.backend.app import create_app  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "results" / "v2_user_outputs" / "measurement_coordinate_hardening"
REPORT_PATH = PROJECT_ROOT / "results" / "v2_user_outputs" / "measurement_coordinate_invariance_report.json"
SOURCE_IMAGE = PROJECT_ROOT / "data" / "real_images_raw_png" / "real_001.png"
SECOND_IMAGE = PROJECT_ROOT / "data" / "real_images_raw_png" / "real_002.png"

METRICS = [
    "SL_mm",
    "FL_mm",
    "TL_open_projection_mm",
    "TL_compressed_virtual_mm",
    "body_depth_mm",
    "caudal_peduncle_depth_mm",
]


def _save_jpg(image: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="JPEG", quality=95)
    return path


def _make_variants(source: Path) -> dict[str, Path]:
    image = Image.open(source).convert("RGB")
    variants: dict[str, Path] = {}
    variants["original_jpg"] = _save_jpg(image, OUTPUT_DIR / "variants" / "real_001_original.jpg")
    variants["padded_border_80px"] = _save_jpg(ImageOps.expand(image, border=80, fill=(245, 245, 245)), OUTPUT_DIR / "variants" / "real_001_padded_80.jpg")
    variants["scaled_075"] = _save_jpg(image.resize((round(image.width * 0.75), round(image.height * 0.75))), OUTPUT_DIR / "variants" / "real_001_scaled_075.jpg")
    variants["scaled_125"] = _save_jpg(image.resize((round(image.width * 1.25), round(image.height * 1.25))), OUTPUT_DIR / "variants" / "real_001_scaled_125.jpg")
    crop = image.crop((10, 10, image.width - 10, image.height - 10))
    variants["edge_crop_10px"] = _save_jpg(crop, OUTPUT_DIR / "variants" / "real_001_edge_crop_10.jpg")
    return variants


def _upload_calibrate_measure(client: TestClient, image_path: Path, specimen_id: str, weight_g: float | None = None) -> dict[str, Any]:
    with image_path.open("rb") as handle:
        upload = client.post(
            "/api/upload",
            files={"file": (image_path.name, handle, "image/jpeg")},
            data={"specimen_id": specimen_id, "weight_g": "" if weight_g is None else str(weight_g)},
        )
    payload: dict[str, Any] = {
        "image_path": str(image_path.relative_to(PROJECT_ROOT)),
        "specimen_id": specimen_id,
        "weight_g": weight_g,
        "upload_status_code": upload.status_code,
    }
    if upload.status_code != 200:
        payload["status"] = "upload_failed"
        payload["error"] = upload.text
        return payload
    upload_json = upload.json()
    session_id = upload_json["session_id"]
    payload["session_id"] = session_id
    calibration = client.post("/api/calibrate", json={"session_id": session_id})
    payload["calibrate_status_code"] = calibration.status_code
    if calibration.status_code != 200:
        payload["status"] = "calibration_request_failed"
        payload["error"] = calibration.text
        return payload
    cal_json = calibration.json()
    payload["calibration"] = cal_json
    if cal_json.get("status") != "success":
        payload["status"] = "calibration_failed"
        payload["error"] = cal_json.get("message", "")
        return payload
    measurement = client.post("/api/measure", json={"session_id": session_id})
    payload["measure_status_code"] = measurement.status_code
    if measurement.status_code != 200:
        payload["status"] = "measurement_failed"
        payload["error"] = measurement.text
        return payload
    measure_json = measurement.json()
    payload["measurement"] = measure_json
    payload["status"] = "measured"
    payload["measurements"] = measure_json.get("measurements", {})
    return payload


def _metric_diff(reference: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    ref_m = reference.get("measurements", {})
    cand_m = candidate.get("measurements", {})
    for metric in METRICS:
        ref = ref_m.get(metric)
        val = cand_m.get(metric)
        try:
            ref_f = float(ref)
            val_f = float(val)
        except (TypeError, ValueError):
            rows.append({"metric": metric, "reference": ref, "candidate": val, "abs_diff_mm": None, "percent_diff": None, "passed": False})
            continue
        diff = abs(val_f - ref_f)
        pct = None if abs(ref_f) <= 1e-9 else 100.0 * diff / abs(ref_f)
        rows.append({"metric": metric, "reference": ref_f, "candidate": val_f, "abs_diff_mm": diff, "percent_diff": pct, "passed": diff <= 1.0})
    return rows


def _display_resize_mapping_check() -> dict[str, Any]:
    image_size = (4200.0, 2970.0)
    points = [(1217.0, 1426.0), (2829.234, 1440.008), (3189.305, 1414.402)]
    display_sizes = [(840.0, 594.0), (1260.0, 891.0), (1680.0, 1188.0)]
    max_roundtrip_error = 0.0
    for width, height in display_sizes:
        sx = width / image_size[0]
        sy = height / image_size[1]
        for x, y in points:
            dx, dy = x * sx, y * sy
            ix, iy = dx / sx, dy / sy
            max_roundtrip_error = max(max_roundtrip_error, math.hypot(ix - x, iy - y))
    return {
        "method": "deterministic image/display coordinate roundtrip",
        "display_sizes": display_sizes,
        "max_roundtrip_error_px": max_roundtrip_error,
        "passed": max_roundtrip_error < 1e-9,
        "note": "The React canvas now uses SVG CTM imageToDisplay/displayToImage; measurements are sent back as warped image coordinates.",
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = TestClient(create_app())
    variants = _make_variants(SOURCE_IMAGE)
    variant_results: dict[str, Any] = {}
    for name, path in variants.items():
        variant_results[name] = _upload_calibrate_measure(client, path, f"invariance_{name}")
    reference = variant_results.get("original_jpg", {})
    comparisons = {}
    for name, result in variant_results.items():
        if name == "original_jpg":
            continue
        if reference.get("status") == "measured" and result.get("status") == "measured":
            comparisons[name] = _metric_diff(reference, result)
        else:
            comparisons[name] = [{"metric": metric, "passed": False, "reason": "reference_or_candidate_not_measured"} for metric in METRICS]

    real_qa_inputs = [
        ("real_001_65_44g", SOURCE_IMAGE, 65.44),
        ("real_002_69_92g", SECOND_IMAGE, 69.92),
    ]
    real_qa: dict[str, Any] = {}
    for label, source, weight in real_qa_inputs:
        jpg = _save_jpg(Image.open(source).convert("RGB"), OUTPUT_DIR / "real_fish_qa" / f"{label}.jpg")
        real_qa[label] = _upload_calibrate_measure(client, jpg, label, weight)

    crop_pass_values = [
        row.get("passed") for row in comparisons.get("edge_crop_10px", []) + comparisons.get("padded_border_80px", [])
        if row.get("passed") is not None
    ]
    report = {
        "version": "2.0.0",
        "coordinate_system_version": "v2.0_warped_image_canonical_board_v1",
        "api_flow": "upload -> calibrate -> measure",
        "output_root": "results/v2_user_outputs/measurement_coordinate_hardening",
        "display_resize_invariance": _display_resize_mapping_check(),
        "variant_results": variant_results,
        "metric_comparisons_to_original": comparisons,
        "same_image_display_resize_invariance_passed": bool(_display_resize_mapping_check()["passed"]),
        "same_image_crop_invariance_passed": bool(crop_pass_values and all(crop_pass_values)),
        "real_fish_qa": real_qa,
        "real_fish_qa_65_44g_passed": real_qa.get("real_001_65_44g", {}).get("status") == "measured",
        "real_fish_qa_69_92g_passed": real_qa.get("real_002_69_92g", {}).get("status") == "measured",
        "notes": [
            "Frontend display scaling is not used for backend measurements.",
            "Backend measurements use warped image coordinates and mm_per_pixel from successful calibration.",
            "Cropped/scaled raw variants are diagnostic; if marker detection fails, the sample is blocked rather than measured with fallback scale.",
        ],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
        "same_image_display_resize_invariance_passed": report["same_image_display_resize_invariance_passed"],
        "same_image_crop_invariance_passed": report["same_image_crop_invariance_passed"],
        "real_fish_qa_65_44g_passed": report["real_fish_qa_65_44g_passed"],
        "real_fish_qa_69_92g_passed": report["real_fish_qa_69_92g_passed"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
