from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    app: str


class ModelPathInfo(BaseModel):
    name: str
    path: str
    exists: bool


class SettingsResponse(BaseModel):
    version: str
    app: str
    core_engine: str
    backend_output_root: str
    user_output_root: str
    admin_output_root: str
    model_paths: list[ModelPathInfo]
    notes: list[str]


class UploadResponse(BaseModel):
    version: str
    session_id: str
    image_name: str
    raw_image_url: str
    status: str


class CalibrateRequest(BaseModel):
    session_id: str
    manual_corners: list[list[float]] | None = None


class CalibrateResponse(BaseModel):
    version: str
    session_id: str
    status: str
    calibration_mode: str = ""
    warped_image_url: str | None = None
    raw_marker_overlay_url: str | None = None
    warped_image_width: int | None = None
    warped_image_height: int | None = None
    mm_per_pixel: float | None = None
    mm_per_pixel_source: str = ""
    marker_count: int = 0
    message: str = ""
    calibration_status: str = ""
    detected_marker_count: int = 0
    homography_raw_to_warped: list[list[float]] | None = None
    detected_markers: list[Any] = Field(default_factory=list)
    detected_board_corners_raw: list[Any] = Field(default_factory=list)
    warped_width_px: int | None = None
    warped_height_px: int | None = None
    board_physical_width_mm: float | None = None
    board_physical_height_mm: float | None = None
    canonical_warp_used: bool = False
    measurement_roi_bounds_warped: list[float] | None = None
    fish_bbox_warped: list[float] | None = None
    fish_margin_to_roi_px: dict[str, float] | None = None
    coordinate_system_version: str = ""


class MeasureResponse(BaseModel):
    version: str
    session_id: str
    coordinate_space: Literal["warped_image"]
    formal_points: dict[str, Any] = Field(default_factory=dict)
    derived_points: dict[str, Any] = Field(default_factory=dict)
    measurements: dict[str, Any] = Field(default_factory=dict)
    qc: dict[str, Any] = Field(default_factory=dict)
    status: str = ""


class UpdatePointsResponse(MeasureResponse):
    unsaved_changes: bool = False


class ExportResponse(BaseModel):
    version: str
    session_id: str
    files: dict[str, str] = Field(default_factory=dict)
    saved_to: str


class ExportHistoryEntry(ExportResponse):
    image_name: str = ""
    specimen_id: str = ""
    weight_g: Any = None
    exported_at: float | str


class ExportListResponse(BaseModel):
    version: str
    exports: list[ExportHistoryEntry] = Field(default_factory=list)


class SessionResponse(BaseModel):
    version: str
    session_id: str
    image_name: str
    specimen_id: str = ""
    weight_g: Any = None
    status: str = ""
    coordinate_space: Literal["warped_image"]
    raw_image_url: str | None = None
    raw_image_width: int | None = None
    raw_image_height: int | None = None
    calibration: CalibrateResponse
    measurement: MeasureResponse | None = None
    exports: dict[str, str] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


class MeasureRequest(BaseModel):
    session_id: str
    preannotation_mode: str = "recommended"
    measurement_axis_mode: str = "auto_qc_gated"


class UpdatePointsRequest(BaseModel):
    session_id: str
    formal_points: dict[str, Any] = Field(default_factory=dict)
    measurement_overrides: dict[str, Any] = Field(default_factory=dict)


class ExportRequest(BaseModel):
    session_id: str
    formats: list[str] = Field(default_factory=lambda: ["csv", "xlsx", "json", "preview_png"])
    save_confirmed_session: bool = True
