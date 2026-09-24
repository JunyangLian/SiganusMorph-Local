from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from siganusmorph.formal_ui import (
    calibrate_with_aruco,
    dataframe_to_csv_bytes,
    dataframe_to_excel_bytes,
    draw_formal_measurement_overlay,
    formal_points_to_full_keypoints,
    image_bytes_png,
    manual_calibrate_from_corners,
    recalculate_formal_measurements,
    run_recommended_measurement,
    simplified_measurement_row,
)
from siganusmorph.image_utils import image_from_bytes, load_image_file

from .session_store import EXPORT_ROOT, PROJECT_ROOT, asset_url, jsonable, load_session, save_session, session_dir


def _project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _safe_upload_filename(filename: str) -> str:
    """Store uploaded bytes under the session directory, never under caller-provided paths."""
    name = Path(str(filename or "uploaded_image")).name
    if not name or name in {".", ".."}:
        name = "uploaded_image"
    return name


def _coordinate_system_metadata(result: dict[str, Any], warped_url: str | None) -> dict[str, Any]:
    """Return explicit V2 coordinate metadata without changing calibration math."""
    info = result.get("info", {}) if isinstance(result.get("info", {}), dict) else {}
    warped = result.get("warped_image")
    warped_width = int(warped.shape[1]) if warped_url and warped is not None else None
    warped_height = int(warped.shape[0]) if warped_url and warped is not None else None
    mm_per_pixel = result.get("mm_per_pixel")
    board_width_mm = None
    board_height_mm = None
    if warped_width and mm_per_pixel not in ("", None):
        board_width_mm = float(mm_per_pixel) * float(warped_width)
    if warped_height and mm_per_pixel not in ("", None):
        board_height_mm = float(mm_per_pixel) * float(warped_height)
    return {
        "calibration_status": result.get("status", "failed"),
        "detected_marker_count": int(result.get("marker_count") or 0),
        "homography_raw_to_warped": result.get("homography_raw_to_warped"),
        "detected_markers": result.get("detected_markers", []),
        "detected_board_corners_raw": result.get("detected_board_corners_raw", []),
        "warped_width_px": warped_width,
        "warped_height_px": warped_height,
        "board_physical_width_mm": board_width_mm,
        "board_physical_height_mm": board_height_mm,
        "canonical_warp_used": bool(result.get("status") == "success" and warped_width and warped_height),
        "measurement_roi_bounds_warped": None,
        "fish_bbox_warped": None,
        "fish_margin_to_roi_px": None,
        "coordinate_system_version": "v2.0_warped_image_canonical_board_v1",
        "canonical_warp_note": str(info.get("anchor_note", "")),
    }


def _to_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
        except (TypeError, ValueError):
            return None
    return None


def _fish_margin_to_roi(fish_bbox: list[float] | None, roi: list[float] | None) -> dict[str, float] | None:
    if not fish_bbox or not roi:
        return None
    x1, y1, x2, y2 = fish_bbox
    rx1, ry1, rx2, ry2 = roi
    return {
        "left": float(x1 - rx1),
        "top": float(y1 - ry1),
        "right": float(rx2 - x2),
        "bottom": float(ry2 - y2),
    }


def save_upload(session: dict[str, Any], filename: str, content: bytes) -> dict[str, Any]:
    directory = session_dir(str(session["session_id"]))
    safe_filename = _safe_upload_filename(filename)
    raw_original = directory / safe_filename
    raw_original.write_bytes(content)
    image = image_from_bytes(content)
    raw_png = directory / "raw.png"
    raw_png.write_bytes(image_bytes_png(image))
    session["raw_image"] = {
        "filename": safe_filename,
        "original_path": str(raw_original),
        "path": str(raw_png),
        "url": asset_url(str(session["session_id"]), "raw.png"),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
    }
    save_session(session)
    return session


def _manual_corner_rows(manual_corners: list[list[float]]) -> pd.DataFrame:
    if len(manual_corners) < 4:
        raise ValueError("manual_corners must contain four points in top-left, top-right, bottom-right, bottom-left order.")
    names = ["top_left", "top_right", "bottom_right", "bottom_left"]
    rows = []
    for name, xy in zip(names, manual_corners[:4]):
        if len(xy) < 2:
            raise ValueError("Each manual calibration corner must contain x and y values.")
        rows.append({"corner": name, "x": float(xy[0]), "y": float(xy[1])})
    return pd.DataFrame(rows)


def run_calibration(session_id: str, manual_corners: list[list[float]] | None = None) -> dict[str, Any]:
    session = load_session(session_id)
    raw_path = Path(session["raw_image"]["path"])
    image = load_image_file(raw_path)
    if manual_corners:
        result = manual_calibrate_from_corners(image, _manual_corner_rows(manual_corners))
        calibration_mode = "manual_corners"
    else:
        result = calibrate_with_aruco(image)
        calibration_mode = "automatic_marker_detection"
    directory = session_dir(session_id)
    overlay_url = None
    warped_url = None
    if result.get("raw_marker_overlay") is not None:
        overlay_path = directory / "raw_marker_overlay.png"
        overlay_path.write_bytes(image_bytes_png(result["raw_marker_overlay"]))
        overlay_url = asset_url(session_id, "raw_marker_overlay.png")
    if result.get("status") == "success" and result.get("warped_image") is not None:
        warped_path = directory / "warped.png"
        warped_image = result["warped_image"]
        warped_path.write_bytes(image_bytes_png(warped_image))
        warped_url = asset_url(session_id, "warped.png")
        session["status"] = "calibrated"
    else:
        session["status"] = "calibration_failed"

    session["calibration_state"] = {
        "status": result.get("status", "failed"),
        "success": bool(result.get("success")),
        "calibration_mode": calibration_mode,
        "message": result.get("calibration_message") or result.get("status_text") or "",
        "marker_count": int(result.get("marker_count") or 0),
        "detected_markers": result.get("detected_markers", []),
        "raw_marker_overlay_url": overlay_url,
        "warped_image_url": warped_url,
        "warped_image_path": str(directory / "warped.png") if warped_url else None,
        "warped_image_width": int(result["warped_image"].shape[1]) if warped_url else None,
        "warped_image_height": int(result["warped_image"].shape[0]) if warped_url else None,
        "mm_per_pixel": result.get("mm_per_pixel"),
        "mm_per_pixel_source": result.get("mm_per_pixel_source", ""),
        "homography_raw_to_warped": result.get("homography_raw_to_warped"),
        "detected_board_corners_raw": result.get("detected_board_corners_raw", []),
        **_coordinate_system_metadata(result, warped_url),
    }
    save_session(session)
    return session


def require_successful_calibration(session: dict[str, Any]) -> tuple[Path, float]:
    cal = session.get("calibration_state", {})
    if cal.get("status") != "success":
        raise ValueError("Calibration has not succeeded; measurement is blocked.")
    mm_per_pixel = cal.get("mm_per_pixel")
    if mm_per_pixel in ("", None):
        raise ValueError("Calibration succeeded without a valid mm_per_pixel.")
    if cal.get("mm_per_pixel_source") == "fallback_default":
        raise ValueError("Fallback mm_per_pixel is not allowed as successful calibration.")
    warped_path = Path(cal.get("warped_image_path") or "")
    if not warped_path.exists():
        raise ValueError("Missing warped image for measurement.")
    return warped_path, float(mm_per_pixel)


def require_measurement_ready(session: dict[str, Any]) -> None:
    if not session.get("formal_points"):
        raise ValueError("Measurement has not produced formal points; export is blocked.")
    if not session.get("measurements"):
        raise ValueError("Measurement results are missing; export is blocked.")
    if session.get("coordinate_space") != "warped_image":
        raise ValueError("Measurement coordinate space must be warped_image before export.")


def public_session_payload(session_id: str) -> dict[str, Any]:
    session = load_session(session_id)
    raw = session.get("raw_image", {})
    cal = session.get("calibration_state", {})
    calibration = {
        "version": "2.0.0",
        "session_id": session_id,
        "status": cal.get("status", "not_run"),
        "calibration_mode": cal.get("calibration_mode", ""),
        "warped_image_url": cal.get("warped_image_url"),
        "raw_marker_overlay_url": cal.get("raw_marker_overlay_url"),
        "warped_image_width": cal.get("warped_image_width"),
        "warped_image_height": cal.get("warped_image_height"),
        "mm_per_pixel": cal.get("mm_per_pixel"),
        "mm_per_pixel_source": cal.get("mm_per_pixel_source", ""),
        "marker_count": cal.get("marker_count", 0),
        "message": cal.get("message", ""),
        "calibration_status": cal.get("calibration_status", cal.get("status", "not_run")),
        "detected_marker_count": cal.get("detected_marker_count", cal.get("marker_count", 0)),
        "homography_raw_to_warped": cal.get("homography_raw_to_warped"),
        "detected_markers": cal.get("detected_markers", []),
        "detected_board_corners_raw": cal.get("detected_board_corners_raw", []),
        "warped_width_px": cal.get("warped_width_px", cal.get("warped_image_width")),
        "warped_height_px": cal.get("warped_height_px", cal.get("warped_image_height")),
        "board_physical_width_mm": cal.get("board_physical_width_mm"),
        "board_physical_height_mm": cal.get("board_physical_height_mm"),
        "canonical_warp_used": cal.get("canonical_warp_used", False),
        "measurement_roi_bounds_warped": cal.get("measurement_roi_bounds_warped"),
        "fish_bbox_warped": cal.get("fish_bbox_warped"),
        "fish_margin_to_roi_px": cal.get("fish_margin_to_roi_px"),
        "coordinate_system_version": cal.get("coordinate_system_version", ""),
    }
    measurement = None
    if session.get("formal_points") and session.get("measurements"):
        measurement = {
            "version": "2.0.0",
            "session_id": session_id,
            "coordinate_space": "warped_image",
            "formal_points": session.get("formal_points", {}),
            "derived_points": session.get("derived_points", {}),
            "measurements": session.get("measurements", {}),
            "qc": session.get("qc", {}),
            "status": session.get("status", ""),
        }
    return {
        "version": "2.0.0",
        "session_id": session_id,
        "image_name": session.get("image_name", ""),
        "specimen_id": session.get("specimen_id", ""),
        "weight_g": session.get("weight_g"),
        "status": session.get("status", ""),
        "coordinate_space": session.get("coordinate_space", "warped_image"),
        "raw_image_url": raw.get("url"),
        "raw_image_width": raw.get("width"),
        "raw_image_height": raw.get("height"),
        "calibration": calibration,
        "measurement": measurement,
        "exports": session.get("exports", {}),
        "created_at": session.get("created_at"),
        "updated_at": session.get("updated_at"),
    }


def run_measurement(session_id: str) -> dict[str, Any]:
    session = load_session(session_id)
    warped_path, mm_per_pixel = require_successful_calibration(session)
    image = load_image_file(warped_path)
    result = run_recommended_measurement(
        image,
        str(session.get("image_name") or "uploaded_image"),
        Path(__file__).resolve().parents[3],
        mm_per_pixel=mm_per_pixel,
    )
    session["status"] = "measured"
    session["coordinate_space"] = "warped_image"
    session["formal_points"] = jsonable(result.get("formal_measurement_points", {}))
    session["derived_points"] = jsonable(result.get("formal_derived_points", {}))
    session["measurements"] = jsonable(result.get("formal_measurements", {}))
    session["metadata"] = jsonable(result.get("metadata", {}))
    session["full_keypoints"] = jsonable(result.get("full_keypoints", {}))
    cal = session.get("calibration_state", {})
    metadata = result.get("metadata", {}) if isinstance(result.get("metadata", {}), dict) else {}
    segmentation_quality = result.get("raw_result", {}).get("segmentation_quality", {})
    if not isinstance(segmentation_quality, dict):
        segmentation_quality = metadata.get("segmentation_quality", {}) if isinstance(metadata.get("segmentation_quality", {}), dict) else {}
    fish_bbox = _to_float_list(result.get("raw_result", {}).get("fish_bbox"))
    if fish_bbox is None:
        fish_bbox = _to_float_list(metadata.get("fish_bbox"))
    roi = _to_float_list(segmentation_quality.get("search_roi_xyxy"))
    if roi is None and cal.get("warped_image_width") and cal.get("warped_image_height"):
        roi = [0.0, 0.0, float(cal["warped_image_width"]), float(cal["warped_image_height"])]
    cal["measurement_roi_bounds_warped"] = roi
    cal["fish_bbox_warped"] = fish_bbox
    cal["fish_margin_to_roi_px"] = _fish_margin_to_roi(fish_bbox, roi)
    session["calibration_state"] = cal
    session["qc"] = {
        "measurement_status": "needs_review",
        "warnings": result.get("metadata", {}).get("qc_warnings", []),
    }
    save_session(session)
    return session


def update_points(session_id: str, formal_points: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    session = load_session(session_id)
    _, mm_per_pixel = require_successful_calibration(session)
    if not formal_points:
        raise ValueError("No formal_points provided.")
    metadata = session.get("metadata", {})
    measurements, derived = recalculate_formal_measurements(
        formal_points,
        metadata,
        mm_per_pixel=mm_per_pixel,
    )
    session["formal_points"] = jsonable(formal_points)
    session["derived_points"] = jsonable(derived)
    session["measurements"] = jsonable(measurements)
    session["measurement_overrides"] = jsonable(overrides or {})
    session["unsaved_changes"] = True
    save_session(session)
    return session


def export_session(session_id: str, formats: list[str]) -> dict[str, Any]:
    session = load_session(session_id)
    warped_path, _mm_per_pixel = require_successful_calibration(session)
    require_measurement_ready(session)
    image = load_image_file(warped_path)
    export_dir = EXPORT_ROOT / session_id
    export_dir.mkdir(parents=True, exist_ok=True)
    measurements = session.get("measurements", {})
    formal_points = session.get("formal_points", {})
    full_keypoints = formal_points_to_full_keypoints(formal_points, session.get("full_keypoints", {}))
    row = simplified_measurement_row(
        measurements,
        image_name=session.get("image_name", ""),
        specimen_id=session.get("specimen_id", ""),
        weight_g=session.get("weight_g", ""),
        notes="V2.0 export; manual review recommended",
    )
    row["version"] = "2.0.0"
    export_df = pd.DataFrame([row])
    preview = draw_formal_measurement_overlay(
        image,
        formal_points,
        measurements,
        session.get("metadata", {}),
        image_name=session.get("image_name", ""),
        specimen_id=session.get("specimen_id", ""),
    )
    files: dict[str, str] = {}
    if "csv" in formats:
        path = export_dir / "measurement.csv"
        path.write_bytes(dataframe_to_csv_bytes(export_df))
        files["csv"] = _project_relative(path)
    if "xlsx" in formats:
        path = export_dir / "measurement.xlsx"
        path.write_bytes(dataframe_to_excel_bytes(export_df))
        files["xlsx"] = _project_relative(path)
    if "json" in formats:
        path = export_dir / "formal_points.json"
        payload = {
            "version": "2.0.0",
            "session_id": session_id,
            "image_name": session.get("image_name", ""),
            "specimen_id": session.get("specimen_id", ""),
            "weight_g": session.get("weight_g"),
            "coordinate_space": "warped_image",
            "formal_measurement_points": formal_points,
            "corrected_keypoints": full_keypoints,
            "derived_points": session.get("derived_points", {}),
            "measurements": measurements,
            "measurement_overrides": session.get("measurement_overrides", {}),
        }
        path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        files["json"] = _project_relative(path)
    if "preview_png" in formats:
        path = export_dir / "preview.png"
        path.write_bytes(image_bytes_png(preview))
        files["preview_png"] = _project_relative(path)
    session["exports"] = files
    session["unsaved_changes"] = False
    save_session(session)
    return {"version": "2.0.0", "session_id": session_id, "files": files, "saved_to": _project_relative(export_dir)}


def list_exports() -> dict[str, Any]:
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    exports: list[dict[str, Any]] = []
    for directory in sorted(EXPORT_ROOT.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
        if not directory.is_dir():
            continue
        session_id = directory.name
        files = {
            key: _project_relative(path)
            for key, path in {
                "csv": directory / "measurement.csv",
                "xlsx": directory / "measurement.xlsx",
                "json": directory / "formal_points.json",
                "preview_png": directory / "preview.png",
            }.items()
            if path.exists()
        }
        image_name = ""
        specimen_id = ""
        weight_g: Any = None
        version = "2.0.0"
        csv_path = directory / "measurement.csv"
        json_path = directory / "formal_points.json"
        if csv_path.exists():
            try:
                rows = pd.read_csv(csv_path).to_dict(orient="records")
                if rows:
                    first = rows[0]
                    image_name = str(first.get("image_name", "") or "")
                    specimen_id = str(first.get("specimen_id", "") or "")
                    weight_g = first.get("weight_g")
                    version = str(first.get("version", version) or version)
            except Exception:
                pass
        if json_path.exists() and not image_name:
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                image_name = str(payload.get("image_name", "") or "")
                specimen_id = str(payload.get("specimen_id", "") or "")
                weight_g = payload.get("weight_g")
                version = str(payload.get("version", version) or version)
            except Exception:
                pass
        exports.append(
            {
                "version": version,
                "session_id": session_id,
                "image_name": image_name,
                "specimen_id": specimen_id,
                "weight_g": weight_g,
                "saved_to": _project_relative(directory),
                "files": files,
                "exported_at": directory.stat().st_mtime,
            }
        )
    return {"version": "2.0.0", "exports": exports}
