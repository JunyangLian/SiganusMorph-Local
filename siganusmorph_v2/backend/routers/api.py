from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..schemas.measurement import (
    CalibrateRequest,
    CalibrateResponse,
    ExportRequest,
    ExportListResponse,
    ExportResponse,
    HealthResponse,
    MeasureRequest,
    MeasureResponse,
    SettingsResponse,
    SessionResponse,
    UpdatePointsRequest,
    UpdatePointsResponse,
    UploadResponse,
)
from ..services.measurement_service import (
    export_session,
    list_exports,
    public_session_payload,
    run_calibration,
    run_measurement,
    save_upload,
    update_points,
)
from ..services.settings_service import get_v2_settings
from ..services.session_store import asset_url, create_session, jsonable


router = APIRouter(prefix="/api")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version="2.0.0", app="SiganusMorph Local V2.0")


@router.get("/settings", response_model=SettingsResponse)
def settings() -> SettingsResponse:
    return SettingsResponse(**get_v2_settings())


@router.post("/upload", response_model=UploadResponse)
async def upload_image(
    file: Annotated[UploadFile, File()],
    specimen_id: Annotated[str, Form()] = "",
    weight_g: Annotated[float | None, Form()] = None,
) -> UploadResponse:
    try:
        content = await file.read()
        session = create_session(file.filename or "uploaded_image", specimen_id=specimen_id, weight_g=weight_g)
        session = save_upload(session, file.filename or "uploaded_image", content)
        return UploadResponse(
            version="2.0.0",
            session_id=str(session["session_id"]),
            image_name=str(session["image_name"]),
            raw_image_url=asset_url(str(session["session_id"]), "raw.png"),
            status="uploaded",
        )
    except Exception as exc:  # noqa: BLE001 - normalized API error.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/session/{session_id}", response_model=SessionResponse)
def get_session_route(session_id: str) -> dict:
    try:
        return jsonable(public_session_payload(session_id))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - normalized API error.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/calibrate", response_model=CalibrateResponse)
def calibrate(request: CalibrateRequest) -> dict:
    try:
        session = run_calibration(request.session_id, request.manual_corners)
        calibration = session.get("calibration_state", {})
        return {
            "version": "2.0.0",
            "session_id": request.session_id,
            "status": calibration.get("status", "failed"),
            "calibration_mode": calibration.get("calibration_mode", ""),
            "warped_image_url": calibration.get("warped_image_url"),
            "raw_marker_overlay_url": calibration.get("raw_marker_overlay_url"),
            "warped_image_width": calibration.get("warped_image_width"),
            "warped_image_height": calibration.get("warped_image_height"),
            "mm_per_pixel": calibration.get("mm_per_pixel"),
            "mm_per_pixel_source": calibration.get("mm_per_pixel_source", ""),
            "marker_count": calibration.get("marker_count", 0),
            "message": calibration.get("message", ""),
            "calibration_status": calibration.get("calibration_status", calibration.get("status", "failed")),
            "detected_marker_count": calibration.get("detected_marker_count", calibration.get("marker_count", 0)),
            "homography_raw_to_warped": calibration.get("homography_raw_to_warped"),
            "detected_markers": calibration.get("detected_markers", []),
            "detected_board_corners_raw": calibration.get("detected_board_corners_raw", []),
            "warped_width_px": calibration.get("warped_width_px", calibration.get("warped_image_width")),
            "warped_height_px": calibration.get("warped_height_px", calibration.get("warped_image_height")),
            "board_physical_width_mm": calibration.get("board_physical_width_mm"),
            "board_physical_height_mm": calibration.get("board_physical_height_mm"),
            "canonical_warp_used": calibration.get("canonical_warp_used", False),
            "measurement_roi_bounds_warped": calibration.get("measurement_roi_bounds_warped"),
            "fish_bbox_warped": calibration.get("fish_bbox_warped"),
            "fish_margin_to_roi_px": calibration.get("fish_margin_to_roi_px"),
            "coordinate_system_version": calibration.get("coordinate_system_version", ""),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - normalized API error.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/measure", response_model=MeasureResponse)
def measure(request: MeasureRequest) -> dict:
    try:
        session = run_measurement(request.session_id)
        return {
            "version": "2.0.0",
            "session_id": request.session_id,
            "coordinate_space": "warped_image",
            "formal_points": session.get("formal_points", {}),
            "derived_points": session.get("derived_points", {}),
            "measurements": session.get("measurements", {}),
            "qc": session.get("qc", {}),
            "status": session.get("status", ""),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - normalized API error.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/update-points", response_model=UpdatePointsResponse)
def update_points_route(request: UpdatePointsRequest) -> dict:
    try:
        session = update_points(request.session_id, request.formal_points, request.measurement_overrides)
        return {
            "version": "2.0.0",
            "session_id": request.session_id,
            "coordinate_space": "warped_image",
            "formal_points": session.get("formal_points", {}),
            "derived_points": session.get("derived_points", {}),
            "measurements": session.get("measurements", {}),
            "qc": session.get("qc", {}),
            "unsaved_changes": session.get("unsaved_changes", False),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - normalized API error.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/export", response_model=ExportResponse)
def export_route(request: ExportRequest) -> dict:
    try:
        return jsonable(export_session(request.session_id, request.formats))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - normalized API error.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/exports", response_model=ExportListResponse)
def exports_route() -> dict:
    try:
        return jsonable(list_exports())
    except Exception as exc:  # noqa: BLE001 - normalized API error.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
