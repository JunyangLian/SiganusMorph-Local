from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import datetime
from math import hypot
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import ImageDraw

from siganusmorph.components import image_clicker
from siganusmorph.config import KEYPOINT_DEFS
from siganusmorph.image_utils import load_image_file, prepare_zoom_display
from siganusmorph.measurements import calculate_measurements
from siganusmorph.dual_axis_measurement import (
    MEASUREMENT_AXIS_AUTO_QC,
    MEASUREMENT_AXIS_BODY_MIDLINE,
    MEASUREMENT_AXIS_MODEL,
    compute_measurement_axis_metadata,
)
from siganusmorph.body_contour_midline import estimate_body_contour_midline_points
from siganusmorph.local_normal_measurement import (
    estimate_body_depth_by_body_midline_normals,
    estimate_peduncle_depth_by_axis_normals,
)
from siganusmorph.segmentation import segment_fish_from_blue_board
from siganusmorph.realworld_review import (
    KEYPOINT_KEYS,
    REVIEW_DIRNAME,
    corrected_json_path,
    corrected_preview_path,
    full_to_short_keypoints,
    load_review_sample,
    path_from_project,
    preannotation_json_path,
    read_json,
    review_root,
    sample_manifest_path,
    save_corrected_review_annotation,
    short_to_full_keypoints,
    update_sample_status,
    xy_from_payload,
)
from siganusmorph.visualization import draw_enhanced_preannotation_overlay, draw_keypoints_and_measurements


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MM_PER_PIXEL = 0.1
DEBUG_LOG_PATH = PROJECT_ROOT / "logs" / "review_interaction_debug.log"
REVIEW_BATCHES = {
    "5.15 real-world review v0.4.1": "realworld_review_v0.4.1",
    "5.24 new fish review v0.6.5": "realworld_review_5_24_v0.6.5",
}
BODY_DEPTH_GEOMETRY_SOURCE_V066 = "body_midline_normal_trunk_boundary_max_width"
BODY_DEPTH_GEOMETRY_VERSION_V066 = "v0.6.9_morphometric_definition_fix"
MEASUREMENT_AXIS_OPTIONS = (
    "model_axis 当前默认 / 兼容旧结果",
    "body_midline_axis experimental smoother axis",
    "auto_qc_gated recommended",
)
MEASUREMENT_AXIS_OPTION_VALUES = {
    MEASUREMENT_AXIS_OPTIONS[0]: MEASUREMENT_AXIS_MODEL,
    MEASUREMENT_AXIS_OPTIONS[1]: MEASUREMENT_AXIS_BODY_MIDLINE,
    MEASUREMENT_AXIS_OPTIONS[2]: MEASUREMENT_AXIS_AUTO_QC,
}


def _load_preannotation(image_name: str) -> dict[str, Any]:
    path = preannotation_json_path(PROJECT_ROOT, image_name, _current_review_dirname())
    return read_json(path) if path.exists() else {}


def _load_corrected(image_name: str) -> dict[str, Any] | None:
    path = corrected_json_path(PROJECT_ROOT, image_name, _current_review_dirname())
    if not path.exists():
        return None
    return read_json(path)


def _current_review_dirname() -> str:
    label = st.session_state.get("rw_review_batch_label", "5.15 real-world review v0.4.1")
    return REVIEW_BATCHES.get(str(label), REVIEW_DIRNAME)


@st.cache_data(show_spinner=False)
def _load_image_cached(path_text: str) -> Any:
    return load_image_file(path_text)


def _log_interaction(
    *,
    image_name: str,
    selected_keypoint: str,
    event_type: str,
    click_xy: Any = "",
    pending_click_xy: Any = "",
    applied: bool = False,
    preannotation_recomputed: bool = False,
    overlay_recomputed: bool = False,
    coordinate_debug: Any = "",
) -> None:
    DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "image_name": image_name,
        "selected_keypoint": selected_keypoint,
        "event_type": event_type,
        "click_xy": click_xy,
        "pending_click_xy": pending_click_xy,
        "applied": applied,
        "rerun_count": st.session_state.get("rw_rerun_counter", 0),
        "last_click_id_processed": st.session_state.get("last_click_id_processed", ""),
        "preannotation_recomputed": preannotation_recomputed,
        "overlay_recomputed": overlay_recomputed,
        "coordinate_debug": coordinate_debug,
    }
    if isinstance(coordinate_debug, dict):
        for key in (
            "raw_click_x",
            "raw_click_y",
            "raw_event_x",
            "raw_event_y",
            "display_width",
            "display_height",
            "canvas_width",
            "canvas_height",
            "rendered_width",
            "rendered_height",
            "original_width",
            "original_height",
            "image_offset_x",
            "image_offset_y",
            "mapped_original_x",
            "mapped_original_y",
            "mapped_back_display_x",
            "mapped_back_display_y",
            "roundtrip_error_px",
            "coordinate_mapping_warning",
        ):
            line[key] = coordinate_debug.get(key, "")
    DEBUG_LOG_PATH.write_text(
        DEBUG_LOG_PATH.read_text(encoding="utf-8") + f"{line}\n" if DEBUG_LOG_PATH.exists() else f"{line}\n",
        encoding="utf-8",
    )


def _init_session_for_image(row: dict[str, Any]) -> None:
    image_name = str(row["image_name"])
    if st.session_state.get("rw_current_image") == image_name and st.session_state.get("rw_corrected_full"):
        return
    pre = _load_preannotation(image_name)
    corrected = _load_corrected(image_name)
    if corrected and isinstance(corrected.get("corrected_keypoints"), dict):
        points = dict(corrected["corrected_keypoints"])
        status = "corrected_and_confirmed"
    else:
        points = dict(pre.get("hybrid_keypoints", {}) or {})
        status = "pending_review"
        update_sample_status(PROJECT_ROOT, image_name, "in_progress", review_dirname=_current_review_dirname())
    st.session_state.rw_current_image = image_name
    st.session_state.current_image_name = image_name
    st.session_state.rw_preannotation = pre
    st.session_state.preannotation_loaded = bool(pre)
    st.session_state.preannotation_result = pre
    st.session_state.rw_corrected_full = points
    st.session_state.corrected_keypoints = points
    st.session_state.rw_review_start_time = datetime.now().isoformat(timespec="seconds")
    st.session_state.rw_review_status = status
    st.session_state.annotation_status = "corrected_and_confirmed" if status == "corrected_and_confirmed" else "auto_preannotation_unverified"
    st.session_state.model_keypoints_raw = dict(pre.get("model_keypoints_raw", {}) or {})
    st.session_state.heatmap_keypoints = dict(pre.get("heatmap_keypoints", {}) or {})
    st.session_state.v034_keypoints = dict(pre.get("v034_keypoints", {}) or {})
    st.session_state.mask_suggestions = dict(pre.get("mask_suggestions", {}) or {})
    st.session_state.geometric_suggestions = dict(pre.get("geometric_suggestions", {}) or {})
    st.session_state.derived_points = dict(pre.get("derived_points", {}) or {})
    st.session_state.hybrid_qc_results = dict(pre.get("hybrid_qc_results", {}) or {})
    st.session_state.rw_zoom_mode = False
    st.session_state.rw_zoom_center = None
    st.session_state.rw_edit_log = []
    st.session_state.pending_click_xy = None
    st.session_state.pending_click_keypoint = ""
    st.session_state.pending_click_id = ""
    st.session_state.last_click_id_processed = ""
    st.session_state.last_click_xy_processed = ""
    st.session_state.is_processing_click = False
    st.session_state.rw_edit_counter = 0
    _log_interaction(
        image_name=image_name,
        selected_keypoint="",
        event_type="image_loaded",
        preannotation_recomputed=False,
        overlay_recomputed=False,
    )


def _reset_to_preannotation() -> None:
    pre = st.session_state.get("rw_preannotation", {})
    st.session_state.rw_corrected_full = dict(pre.get("hybrid_keypoints", {}) or {})
    st.session_state.corrected_keypoints = dict(st.session_state.rw_corrected_full)
    st.session_state.rw_edit_log = []
    st.session_state.rw_review_start_time = datetime.now().isoformat(timespec="seconds")
    st.session_state.annotation_status = "auto_preannotation_unverified"
    st.session_state.pending_click_xy = None
    st.session_state.pending_click_keypoint = ""
    st.session_state.pending_click_id = ""


def _update_selected_point(key: str, xy: dict[str, float]) -> None:
    points = dict(st.session_state.get("rw_corrected_full", {}) or {})
    old = xy_from_payload(points.get(key))
    points[key] = [float(xy["x"]), float(xy["y"])]
    st.session_state.rw_corrected_full = points
    st.session_state.corrected_keypoints = points
    st.session_state.annotation_status = "editing"
    if old is not None:
        dx = float(xy["x"]) - old[0]
        dy = float(xy["y"]) - old[1]
        displacement_px = (dx * dx + dy * dy) ** 0.5
    else:
        displacement_px = 0.0
    log = list(st.session_state.get("rw_edit_log", []) or [])
    log.append(
        {
            "keypoint_name": key,
            "old_xy": [float(old[0]), float(old[1])] if old is not None else None,
            "new_xy": [float(xy["x"]), float(xy["y"])],
            "displacement_px": displacement_px,
            "displacement_mm": displacement_px * MM_PER_PIXEL,
            "edited_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    st.session_state.rw_edit_log = log
    st.session_state.rw_edit_counter = int(st.session_state.get("rw_edit_counter", 0)) + 1


def _click_id(image_name: str, selected_key: str, click: dict[str, Any], point: dict[str, float]) -> str:
    button = str(click.get("button", ""))
    # Quantize to 0.1 px so tiny float formatting differences do not create
    # new events, while distinct clicks still get through.
    return f"{image_name}|{selected_key}|{button}|{round(float(point['x']), 1)}|{round(float(point['y']), 1)}|{st.session_state.get('rw_zoom_mode', False)}"


def display_to_original_coord(
    click_x: float,
    click_y: float,
    display_width: float,
    display_height: float,
    original_width: float,
    original_height: float,
    image_offset_x: float = 0.0,
    image_offset_y: float = 0.0,
    zoom_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Map actual rendered display coordinates to original warped-image coordinates."""
    x_display = float(click_x) - float(image_offset_x)
    y_display = float(click_y) - float(image_offset_y)
    display_width = max(1.0, float(display_width))
    display_height = max(1.0, float(display_height))
    original_width = max(1.0, float(original_width))
    original_height = max(1.0, float(original_height))
    outside = x_display < 0 or y_display < 0 or x_display > display_width or y_display > display_height
    zoom_info = dict(zoom_info or {})
    if bool(zoom_info.get("zoom_mode", False)):
        crop_x1 = float(zoom_info.get("offset_x", 0.0))
        crop_y1 = float(zoom_info.get("offset_y", 0.0))
        crop_width = float(zoom_info.get("crop_width", zoom_info.get("scale_x", 1.0) * display_width))
        crop_height = float(zoom_info.get("crop_height", zoom_info.get("scale_y", 1.0) * display_height))
        x_original = crop_x1 + x_display / display_width * crop_width
        y_original = crop_y1 + y_display / display_height * crop_height
    else:
        x_original = x_display / display_width * original_width
        y_original = y_display / display_height * original_height
    x_original = min(max(x_original, 0.0), original_width - 1.0)
    y_original = min(max(y_original, 0.0), original_height - 1.0)
    return {
        "x": float(x_original),
        "y": float(y_original),
        "display_x": float(x_display),
        "display_y": float(y_display),
        "clicked_outside_image": bool(outside),
    }


def original_to_display_coord(
    original_x: float,
    original_y: float,
    display_width: float,
    display_height: float,
    original_width: float,
    original_height: float,
    image_offset_x: float = 0.0,
    image_offset_y: float = 0.0,
    zoom_info: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Map original warped-image coordinates back to actual rendered display coordinates."""
    display_width = max(1.0, float(display_width))
    display_height = max(1.0, float(display_height))
    original_width = max(1.0, float(original_width))
    original_height = max(1.0, float(original_height))
    zoom_info = dict(zoom_info or {})
    if bool(zoom_info.get("zoom_mode", False)):
        crop_x1 = float(zoom_info.get("offset_x", 0.0))
        crop_y1 = float(zoom_info.get("offset_y", 0.0))
        crop_width = max(1.0, float(zoom_info.get("crop_width", zoom_info.get("scale_x", 1.0) * display_width)))
        crop_height = max(1.0, float(zoom_info.get("crop_height", zoom_info.get("scale_y", 1.0) * display_height)))
        x_display = (float(original_x) - crop_x1) / crop_width * display_width
        y_display = (float(original_y) - crop_y1) / crop_height * display_height
    else:
        x_display = float(original_x) / original_width * display_width
        y_display = float(original_y) / original_height * display_height
    return {
        "x": float(x_display + image_offset_x),
        "y": float(y_display + image_offset_y),
    }


def _coordinate_transform_from_click(
    click: dict[str, Any],
    display: Any,
    transform: dict[str, Any],
    original_shape: tuple[int, int, int],
) -> dict[str, Any]:
    original_height, original_width = original_shape[:2]
    canvas_width = float(click.get("canvas_width") or click.get("width") or display.width)
    canvas_height = float(click.get("canvas_height") or click.get("height") or display.height)
    rendered_width = float(click.get("rendered_width") or canvas_width)
    rendered_height = float(click.get("rendered_height") or canvas_height)
    component_width = canvas_width
    component_height = canvas_height
    zoom_info = dict(transform)
    zoom_info["crop_width"] = float(transform.get("scale_x", 1.0)) * float(display.width)
    zoom_info["crop_height"] = float(transform.get("scale_y", 1.0)) * float(display.height)
    mapped = display_to_original_coord(
        float(click["x"]),
        float(click["y"]),
        component_width,
        component_height,
        float(original_width),
        float(original_height),
        image_offset_x=0.0,
        image_offset_y=0.0,
        zoom_info=zoom_info,
    )
    mapped_back = original_to_display_coord(
        mapped["x"],
        mapped["y"],
        component_width,
        component_height,
        float(original_width),
        float(original_height),
        image_offset_x=0.0,
        image_offset_y=0.0,
        zoom_info=zoom_info,
    )
    roundtrip = hypot(float(click["x"]) - mapped_back["x"], float(click["y"]) - mapped_back["y"])
    return {
        "raw_click_x": float(click["x"]),
        "raw_click_y": float(click["y"]),
        "raw_event_x": float(click.get("raw_x", click["x"])),
        "raw_event_y": float(click.get("raw_y", click["y"])),
        "display_width": component_width,
        "display_height": component_height,
        "component_width": component_width,
        "component_height": component_height,
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "rendered_width": rendered_width,
        "rendered_height": rendered_height,
        "pil_display_width": float(display.width),
        "pil_display_height": float(display.height),
        "original_width": float(original_width),
        "original_height": float(original_height),
        "image_offset_x": 0.0,
        "image_offset_y": 0.0,
        "zoom_mode": bool(transform.get("zoom_mode", False)),
        "zoom_crop_box": [
            float(transform.get("offset_x", 0.0)),
            float(transform.get("offset_y", 0.0)),
            float(transform.get("offset_x", 0.0)) + zoom_info["crop_width"],
            float(transform.get("offset_y", 0.0)) + zoom_info["crop_height"],
        ],
        "display_x": mapped["display_x"],
        "display_y": mapped["display_y"],
        "mapped_original_x": mapped["x"],
        "mapped_original_y": mapped["y"],
        "mapped_back_display_x": mapped_back["x"],
        "mapped_back_display_y": mapped_back["y"],
        "roundtrip_error_px": roundtrip,
        "clicked_outside_image": mapped["clicked_outside_image"],
        "coordinate_mapping_warning": bool(roundtrip > 2.0),
    }


def _handle_image_click(
    click: dict[str, Any] | None,
    transform: dict[str, Any],
    image_name: str,
    selected_key: str,
    *,
    display: Any,
    original_shape: tuple[int, int, int],
) -> None:
    if click is None:
        return
    if st.session_state.get("is_processing_click"):
        return
    coord_debug = _coordinate_transform_from_click(click, display, transform, original_shape)
    st.session_state.rw_coordinate_debug = {
        **coord_debug,
        "image_name": image_name,
        "selected_keypoint": selected_key,
        "old_keypoint_original_xy": xy_from_payload(st.session_state.get("rw_corrected_full", {}).get(selected_key)),
    }
    if coord_debug.get("clicked_outside_image"):
        _log_interaction(
            image_name=image_name,
            selected_keypoint=selected_key,
            event_type="clicked_outside_image",
            click_xy={"x": click.get("x"), "y": click.get("y")},
            coordinate_debug=coord_debug,
        )
        return
    point = {"x": float(coord_debug["mapped_original_x"]), "y": float(coord_debug["mapped_original_y"])}
    cid = _click_id(image_name, selected_key, click, point)
    if cid == st.session_state.get("last_click_id_processed"):
        _log_interaction(
            image_name=image_name,
            selected_keypoint=selected_key,
            event_type="duplicate_click_ignored",
            click_xy=point,
            pending_click_xy=st.session_state.get("pending_click_xy"),
            coordinate_debug=coord_debug,
        )
        return
    st.session_state.is_processing_click = True
    try:
        if click.get("button") == "right":
            st.session_state.last_click_id_processed = cid
            st.session_state.last_click_xy_processed = [point["x"], point["y"]]
            if st.session_state.get("rw_zoom_mode"):
                st.session_state.rw_zoom_mode = False
                st.session_state.rw_zoom_center = None
                event = "right_click_reset_view"
            else:
                st.session_state.rw_zoom_mode = True
                st.session_state.rw_zoom_center = (point["x"], point["y"])
                event = "right_click_zoom"
            _log_interaction(
                image_name=image_name,
                selected_keypoint=selected_key,
                event_type=event,
                click_xy=point,
                coordinate_debug=coord_debug,
            )
            st.session_state.is_processing_click = False
            st.rerun()
        elif click.get("button") == "left":
            st.session_state.pending_click_xy = {"x": float(point["x"]), "y": float(point["y"])}
            st.session_state.pending_click_keypoint = selected_key
            st.session_state.pending_click_id = cid
            st.session_state.last_click_id_processed = cid
            st.session_state.last_click_xy_processed = [point["x"], point["y"]]
            _log_interaction(
                image_name=image_name,
                selected_keypoint=selected_key,
                event_type="pending_click_recorded",
                click_xy=point,
                pending_click_xy=st.session_state.pending_click_xy,
                coordinate_debug=coord_debug,
            )
            st.session_state.is_processing_click = False
            st.rerun()
    finally:
        st.session_state.is_processing_click = False


def _apply_pending_click(image_name: str) -> bool:
    pending = st.session_state.get("pending_click_xy")
    key = str(st.session_state.get("pending_click_keypoint", ""))
    if not isinstance(pending, dict) or key not in KEYPOINT_KEYS:
        return False
    _update_selected_point(key, pending)
    _log_interaction(
        image_name=image_name,
        selected_keypoint=key,
        event_type="pending_click_applied",
        pending_click_xy=pending,
        applied=True,
        overlay_recomputed=True,
    )
    st.session_state.pending_click_xy = None
    st.session_state.pending_click_keypoint = ""
    st.session_state.pending_click_id = ""
    return True


def _clear_pending_click(image_name: str) -> None:
    pending = st.session_state.get("pending_click_xy")
    key = str(st.session_state.get("pending_click_keypoint", ""))
    st.session_state.pending_click_xy = None
    st.session_state.pending_click_keypoint = ""
    st.session_state.pending_click_id = ""
    _log_interaction(
        image_name=image_name,
        selected_keypoint=key,
        event_type="pending_click_cleared",
        pending_click_xy=pending,
    )


def _undo_last_edit(image_name: str) -> bool:
    log = list(st.session_state.get("rw_edit_log", []) or [])
    if not log:
        return False
    event = log.pop()
    key = str(event.get("keypoint_name", ""))
    old = event.get("old_xy")
    if key not in KEYPOINT_KEYS or not isinstance(old, (list, tuple)) or len(old) < 2:
        st.session_state.rw_edit_log = log
        return False
    points = dict(st.session_state.get("rw_corrected_full", {}) or {})
    points[key] = [float(old[0]), float(old[1])]
    st.session_state.rw_corrected_full = points
    st.session_state.corrected_keypoints = points
    st.session_state.rw_edit_log = log
    st.session_state.rw_edit_counter = int(st.session_state.get("rw_edit_counter", 0)) + 1
    _log_interaction(
        image_name=image_name,
        selected_keypoint=key,
        event_type="undo_last_edit",
        applied=True,
        overlay_recomputed=True,
    )
    return True


def _metadata_for_overlay() -> dict[str, Any]:
    pre = dict(st.session_state.get("rw_preannotation", {}) or {})
    metadata = dict(pre.get("preannotation_metadata", {}) or {})
    metadata["corrected_keypoints"] = dict(st.session_state.get("rw_corrected_full", {}) or {})
    metadata["show_heatmap_points"] = bool(st.session_state.get("rw_show_heatmap", False))
    metadata["show_v034_points"] = bool(st.session_state.get("rw_show_v034", False))
    metadata["show_mask_suggestions"] = bool(st.session_state.get("rw_show_suggestions", True))
    metadata["show_qc_warnings"] = bool(st.session_state.get("rw_show_qc", True))
    metadata["show_local_normal_measurement_suggestions"] = bool(st.session_state.get("rw_show_local_normal", True))
    metadata["show_width_profiles"] = bool(st.session_state.get("rw_show_width_profiles", True))
    metadata["show_curvature_qc"] = bool(st.session_state.get("rw_show_curvature", True))
    metadata["show_model_axis"] = bool(st.session_state.get("rw_show_model_axis", True))
    metadata["show_body_midline_axis"] = bool(st.session_state.get("rw_show_body_midline_axis", True))
    metadata["show_geometric_c_points"] = bool(st.session_state.get("rw_show_geometric_c_points", True))
    metadata["show_body_midline_contour"] = bool(st.session_state.get("rw_show_body_midline_contour", True))
    metadata["show_dual_axis_comparison"] = bool(st.session_state.get("rw_show_dual_axis", True))
    metadata["show_selected_measurement_axis_only"] = bool(st.session_state.get("rw_show_selected_axis_only", False))
    metadata["show_p6_geometry_qc"] = bool(st.session_state.get("rw_show_p6_geometry_qc", False))
    metadata["show_p6_fallback_suggestion"] = bool(st.session_state.get("rw_show_p6_fallback_suggestion", False))
    metadata["show_geometric_body_depth"] = bool(st.session_state.get("rw_show_geometric_body_depth", False))
    metadata["show_body_depth_width_profile"] = bool(st.session_state.get("rw_show_body_depth_width_profile", False))
    metadata["show_body_depth_comparison_labels"] = bool(st.session_state.get("rw_show_body_depth_comparison_labels", False))
    metadata["show_geometric_peduncle_depth"] = bool(st.session_state.get("rw_show_geometric_peduncle_depth", False))
    metadata["show_peduncle_depth_width_profile"] = bool(st.session_state.get("rw_show_peduncle_depth_width_profile", False))
    metadata["show_peduncle_depth_comparison_labels"] = bool(st.session_state.get("rw_show_peduncle_depth_comparison_labels", False))
    metadata["show_fin_suppression_regions"] = bool(st.session_state.get("rw_show_fin_suppression_regions", False))
    metadata["show_body_depth_boundary_contour"] = bool(st.session_state.get("rw_show_body_depth_boundary_contour", False))
    metadata["show_body_trunk_contours"] = bool(st.session_state.get("rw_show_body_trunk_contours", False))
    metadata["show_p3_operculum_suggestion"] = bool(st.session_state.get("rw_show_p3_operculum_suggestion", False))
    metadata["show_tail_fork_gap_region"] = bool(st.session_state.get("rw_show_tail_fork_gap_region", False))
    metadata["show_current_body_depth_line"] = bool(st.session_state.get("rw_show_current_body_depth_line", False))
    metadata["show_current_peduncle_depth_line"] = bool(st.session_state.get("rw_show_current_peduncle_depth_line", False))
    metadata["show_compressed_tail_p7v"] = bool(st.session_state.get("rw_show_compressed_tail_p7v", False))
    metadata["show_open_projection_p7v"] = bool(st.session_state.get("rw_show_open_projection_p7v", False))
    metadata["show_tail_folding_arc_reference"] = bool(st.session_state.get("rw_show_tail_folding_arc_reference", False))
    metadata["show_tl_comparison_labels"] = bool(st.session_state.get("rw_show_tl_comparison_labels", False))
    metadata["show_v01_heatmap_points"] = bool(st.session_state.get("rw_show_debug_heatmap_points", False))
    metadata["show_v05_heatmap_points"] = bool(st.session_state.get("rw_show_debug_heatmap_points", False))
    metadata["show_point_source_labels"] = bool(st.session_state.get("rw_show_point_source_labels", False))
    metadata["advanced_overlay_mode"] = bool(st.session_state.get("rw_advanced_settings_enabled", False))
    if not bool(st.session_state.get("rw_advanced_settings_enabled", False)):
        metadata.update(
            {
                "show_heatmap_points": False,
                "show_v034_points": False,
                "show_mask_suggestions": False,
                "show_local_normal_measurement_suggestions": False,
                "show_width_profiles": False,
                "show_qc_warnings": True,
                "show_curvature_qc": True,
                "show_model_axis": False,
                "show_body_midline_axis": True,
                "show_geometric_c_points": False,
                "show_body_midline_contour": False,
                "show_dual_axis_comparison": True,
                "show_selected_measurement_axis_only": False,
                "show_p6_geometry_qc": False,
                "show_p6_fallback_suggestion": False,
                "show_geometric_body_depth": True,
                "show_body_depth_width_profile": True,
                "show_body_depth_comparison_labels": False,
                "show_geometric_peduncle_depth": True,
                "show_peduncle_depth_width_profile": True,
                "show_peduncle_depth_comparison_labels": False,
                "show_fin_suppression_regions": False,
                "show_body_depth_boundary_contour": False,
                "show_body_trunk_contours": False,
                "show_p3_operculum_suggestion": False,
                "show_tail_fork_gap_region": False,
                "show_current_body_depth_line": False,
                "show_current_peduncle_depth_line": False,
                "show_compressed_tail_p7v": True,
                "show_open_projection_p7v": False,
                "show_tail_folding_arc_reference": False,
                "show_tl_comparison_labels": False,
                "show_v06_selected_points": True,
                "show_point_source_labels": False,
                "advanced_overlay_mode": False,
            }
        )
    return metadata


def _current_measurement_axis_mode() -> str:
    if not bool(st.session_state.get("rw_advanced_settings_enabled", False)):
        return MEASUREMENT_AXIS_AUTO_QC
    option = st.session_state.get("rw_measurement_axis_mode", MEASUREMENT_AXIS_OPTIONS[0])
    return MEASUREMENT_AXIS_OPTION_VALUES.get(str(option), MEASUREMENT_AXIS_MODEL)


def _is_current_body_depth_geometry(geometry: Any) -> bool:
    if not isinstance(geometry, dict):
        return False
    return (
        geometry.get("body_depth_geometry_version") == BODY_DEPTH_GEOMETRY_VERSION_V066
        or geometry.get("body_depth_source") == BODY_DEPTH_GEOMETRY_SOURCE_V066
    )


def _has_body_depth_geometry_points(geometry: Any) -> bool:
    if not isinstance(geometry, dict):
        return False
    for key in ("P8_geometric", "P9_geometric"):
        value = geometry.get(key)
        if not (isinstance(value, (list, tuple)) and len(value) >= 2 and value[0] is not None and value[1] is not None):
            return False
    return True


def _body_depth_geometry_cache_key(corrected: dict[str, Any]) -> str:
    key_payload = [
        st.session_state.get("rw_current_image", ""),
        [corrected.get(name) for name in KEYPOINT_KEYS],
    ]
    return hashlib.sha1(repr(key_payload).encode("utf-8")).hexdigest()


def _active_body_depth_geometry_for_display(
    image: Any,
    corrected: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    image_name = str(st.session_state.get("rw_current_image", ""))
    cache_key = _body_depth_geometry_cache_key(corrected)
    cache = st.session_state.get("rw_body_depth_geometry_display_cache")
    saved = metadata.get("body_depth_geometry") if isinstance(metadata.get("body_depth_geometry"), dict) else {}
    legacy_detected = bool(saved) and not _is_current_body_depth_geometry(saved)
    info: dict[str, Any] = {
        "active_geometry_version": "",
        "active_body_depth_source": "",
        "geometry_origin": "unavailable",
        "legacy_body_depth_geometry_detected": legacy_detected,
        "legacy_body_depth_geometry_ignored": False,
    }
    if isinstance(cache, dict):
        cached_geometry = cache.get("geometry")
        if (
            cache.get("image_name") == image_name
            and cache.get("cache_key") == cache_key
            and cache.get("geometry_version") == BODY_DEPTH_GEOMETRY_VERSION_V066
            and cache.get("body_depth_source") == BODY_DEPTH_GEOMETRY_SOURCE_V066
            and _is_current_body_depth_geometry(cached_geometry)
            and _has_body_depth_geometry_points(cached_geometry)
        ):
            geometry = dict(cached_geometry)
            info.update(
                {
                    "active_geometry_version": geometry.get("body_depth_geometry_version", BODY_DEPTH_GEOMETRY_VERSION_V066),
                    "active_body_depth_source": geometry.get("body_depth_source", BODY_DEPTH_GEOMETRY_SOURCE_V066),
                    "geometry_origin": "session_display_cache",
                    "legacy_body_depth_geometry_ignored": legacy_detected,
                }
            )
            return geometry, info
        st.session_state.rw_body_depth_geometry_display_cache = {}
    if _is_current_body_depth_geometry(saved) and _has_body_depth_geometry_points(saved):
        geometry = dict(saved)
        geometry.setdefault("body_depth_geometry_version", BODY_DEPTH_GEOMETRY_VERSION_V066)
        geometry.setdefault("body_depth_source", BODY_DEPTH_GEOMETRY_SOURCE_V066)
        info.update(
            {
                "active_geometry_version": geometry.get("body_depth_geometry_version", ""),
                "active_body_depth_source": geometry.get("body_depth_source", ""),
                "geometry_origin": "saved_metadata_v066",
            }
        )
        return geometry, info
    geometry = _compute_body_depth_geometry_for_display(image, corrected)
    if geometry and _is_current_body_depth_geometry(geometry) and _has_body_depth_geometry_points(geometry):
        info.update(
            {
                "active_geometry_version": geometry.get("body_depth_geometry_version", BODY_DEPTH_GEOMETRY_VERSION_V066),
                "active_body_depth_source": geometry.get("body_depth_source", BODY_DEPTH_GEOMETRY_SOURCE_V066),
                "geometry_origin": "recomputed_display_only",
                "legacy_body_depth_geometry_ignored": legacy_detected,
            }
        )
        return geometry, info
    if legacy_detected:
        info["legacy_body_depth_geometry_ignored"] = True
        info["geometry_error"] = "Legacy geometric body-depth data detected, but current v0.6.6 geometry could not be recomputed."
    return {}, info


def _metadata_with_measurement_axes(image: Any) -> dict[str, Any]:
    metadata = _metadata_for_overlay()
    corrected = dict(st.session_state.get("rw_corrected_full", {}) or {})
    if len(corrected) == len(KEYPOINT_KEYS):
        corrected_short_for_tl = full_to_short_keypoints(corrected)
        try:
            tail_measurements = calculate_measurements(corrected_short_for_tl, MM_PER_PIXEL, _current_measurement_axis_mode())
            metadata["compressed_tail_tl"] = {
                key: tail_measurements.get(key)
                for key in (
                    "P7V_open_projection_x",
                    "P7V_open_projection_y",
                    "P7V_compressed_virtual_x",
                    "P7V_compressed_virtual_y",
                    "TL_open_projection_mm",
                    "TL_compressed_virtual_mm",
                    "TL_difference_mm",
                    "TL_difference_percent",
                    "RU_mm",
                    "RL_mm",
                    "Rmax_mm",
                    "upper_lobe_angle_to_axis_deg",
                    "lower_lobe_angle_to_axis_deg",
                    "inter_lobe_open_angle_deg",
                    "upper_lobe_angle_deg",
                    "lower_lobe_angle_deg",
                    "tail_open_angle_deg",
                    "tail_lobe_length_asymmetry_ratio",
                    "compressed_tail_tl_valid_qc_pass",
                    "compressed_tail_tl_qc_pass",
                    "compressed_vs_projection_difference_flag",
                    "compressed_vs_projection_difference_reason",
                    "compressed_tail_tl_review_required",
                    "compressed_tail_tl_review_reason",
                )
            }
        except Exception:
            metadata.setdefault("compressed_tail_tl", {})
        try:
            axis_payload = compute_measurement_axis_metadata(
                image,
                corrected,
                mm_per_pixel=MM_PER_PIXEL,
                measurement_axis_mode=_current_measurement_axis_mode(),
            )
            measurement_axes = dict(axis_payload.get("measurement_axes", {}) or {})
            row_fields = dict(axis_payload.get("measurement_axis_row_fields", {}) or {})
            for key in (
                "TL_curve_selected_mm",
                "SL_curve_selected_mm",
                "curvature_index_selected",
                "selected_measurement_axis",
                "TL_curve_axis_diff_mm",
                "SL_curve_axis_diff_mm",
                "curvature_index_axis_diff",
                "dual_axis_disagreement",
            ):
                if key in row_fields:
                    measurement_axes[key] = row_fields[key]
            metadata["measurement_axes"] = measurement_axes
        except Exception as exc:  # noqa: BLE001 - display failure should not block point edits.
            metadata["measurement_axes"] = {
                "measurement_axis_mode_request": _current_measurement_axis_mode(),
                "selected_measurement_axis": "model_axis",
                "measurements_needs_review": True,
                "measurement_axis_review_reason": f"measurement_axis_failed:{exc}",
            }
        if (
            bool(metadata.get("show_geometric_body_depth", False))
            or bool(metadata.get("show_body_depth_width_profile", False))
            or bool(metadata.get("show_body_depth_comparison_labels", False))
            or bool(metadata.get("show_body_depth_boundary_contour", False))
            or bool(metadata.get("show_body_trunk_contours", False))
            or bool(metadata.get("show_fin_suppression_regions", False))
        ):
            geometry, geometry_info = _active_body_depth_geometry_for_display(image, corrected, metadata)
            metadata["active_body_depth_geometry_info"] = geometry_info
            if geometry:
                metadata["body_depth_geometry"] = geometry
            else:
                metadata.pop("body_depth_geometry", None)
        if (
            bool(metadata.get("show_geometric_peduncle_depth", False))
            or bool(metadata.get("show_peduncle_depth_width_profile", False))
            or bool(metadata.get("show_peduncle_depth_comparison_labels", False))
        ):
            ped_geometry = metadata.get("peduncle_depth_geometry") if isinstance(metadata.get("peduncle_depth_geometry"), dict) else {}
            if not ped_geometry:
                ped_geometry = _compute_peduncle_depth_geometry_for_display(image, corrected)
            if ped_geometry:
                metadata["peduncle_depth_geometry"] = ped_geometry
    return metadata


def _compute_body_depth_geometry_for_display(image: Any, corrected: dict[str, Any]) -> dict[str, Any]:
    """Compute display-only geometric P8/P9 for older review JSON payloads."""
    image_name = str(st.session_state.get("rw_current_image", ""))
    cache_key = _body_depth_geometry_cache_key(corrected)
    try:
        fish_mask, _, _ = segment_fish_from_blue_board(image)
        body_midline = estimate_body_contour_midline_points(fish_mask, corrected, getattr(image, "shape", None) or [])
        geometry = estimate_body_depth_by_body_midline_normals(
            fish_mask,
            body_midline.get("body_core_mask"),
            body_midline,
            corrected,
            getattr(image, "shape", None) or [],
            config={"mm_per_pixel": MM_PER_PIXEL},
        )
        geometry["display_only"] = True
        geometry["body_depth_source"] = BODY_DEPTH_GEOMETRY_SOURCE_V066
        geometry["body_depth_geometry_version"] = BODY_DEPTH_GEOMETRY_VERSION_V066
        st.session_state.rw_body_depth_geometry_display_cache = {
            "image_name": image_name,
            "cache_key": cache_key,
            "geometry_version": BODY_DEPTH_GEOMETRY_VERSION_V066,
            "body_depth_source": BODY_DEPTH_GEOMETRY_SOURCE_V066,
            "geometry": geometry,
        }
        return geometry
    except Exception as exc:  # noqa: BLE001
        st.session_state.rw_body_depth_geometry_display_cache = {
            "image_name": image_name,
            "cache_key": cache_key,
            "geometry_version": BODY_DEPTH_GEOMETRY_VERSION_V066,
            "body_depth_source": BODY_DEPTH_GEOMETRY_SOURCE_V066,
            "geometry": {
                "body_depth_source": BODY_DEPTH_GEOMETRY_SOURCE_V066,
                "body_depth_geometry_version": BODY_DEPTH_GEOMETRY_VERSION_V066,
                "body_depth_geometric_qc_pass": False,
                "body_depth_geometric_review_reason": f"display_geometry_failed:{exc}",
                "display_only": True,
            },
        }
        return {}


def _compute_peduncle_depth_geometry_for_display(image: Any, corrected: dict[str, Any]) -> dict[str, Any]:
    """Compute display-only geometric P10/P11 for review overlays."""
    try:
        fish_mask, _, _ = segment_fish_from_blue_board(image)
        geometry = estimate_peduncle_depth_by_axis_normals(
            fish_mask,
            corrected,
            getattr(image, "shape", None) or [],
            config={"mm_per_pixel": MM_PER_PIXEL},
        )
        geometry["display_only"] = True
        return geometry
    except Exception as exc:  # noqa: BLE001
        return {
            "peduncle_depth_source": "P4_P5_axis_normal_min_width",
            "peduncle_depth_geometry_version": "v0.6.8_peduncle_axis_normals",
            "peduncle_depth_geometric_qc_pass": False,
            "peduncle_depth_geometric_review_reason": f"display_peduncle_geometry_failed:{exc}",
            "display_only": True,
        }


def _draw_cross(draw: ImageDraw.ImageDraw, x: float, y: float, color: tuple[int, int, int], size: int = 16, width: int = 4) -> None:
    draw.line((x - size, y, x + size, y), fill=color, width=width)
    draw.line((x, y - size, x, y + size), fill=color, width=width)


def _draw_pending_click_preview(display: Any) -> Any:
    """Draw raw click and round-tripped mapped-back coordinates on display image."""
    debug = st.session_state.get("rw_coordinate_debug")
    if not isinstance(debug, dict):
        return display
    pending = st.session_state.get("pending_click_xy")
    if not isinstance(pending, dict):
        return display
    image = display.copy()
    draw = ImageDraw.Draw(image)
    raw_x = float(debug.get("raw_click_x", 0.0))
    raw_y = float(debug.get("raw_click_y", 0.0))
    back_x = float(debug.get("mapped_back_display_x", raw_x))
    back_y = float(debug.get("mapped_back_display_y", raw_y))
    _draw_cross(draw, raw_x, raw_y, (0, 120, 255), size=18, width=4)
    _draw_cross(draw, back_x, back_y, (255, 40, 40), size=10, width=3)
    if float(debug.get("roundtrip_error_px", 0.0)) > 2.0:
        draw.text((max(5, raw_x + 20), max(5, raw_y + 20)), "coordinate warning", fill=(255, 40, 40))
    return image


def main() -> None:
    st.set_page_config(page_title="Real-world Review Queue", layout="wide")
    st.session_state.rw_rerun_counter = int(st.session_state.get("rw_rerun_counter", 0)) + 1
    st.title("Real-world Review Queue")
    batch_labels = list(REVIEW_BATCHES)
    st.sidebar.selectbox(
        "Review batch",
        batch_labels,
        index=batch_labels.index(st.session_state.get("rw_review_batch_label", batch_labels[0]))
        if st.session_state.get("rw_review_batch_label", batch_labels[0]) in batch_labels
        else 0,
        key="rw_review_batch_label",
    )
    review_dirname = _current_review_dirname()
    root = review_root(PROJECT_ROOT, review_dirname)
    sample_path = sample_manifest_path(PROJECT_ROOT, review_dirname)
    if not sample_path.exists():
        st.warning(
            "Review queue not found. Run the setup script first. "
            "For 5.24 use `python scripts/setup_realworld_review_5_24_v065.py`."
        )
        return
    sample = load_review_sample(PROJECT_ROOT, review_dirname)
    if sample.empty:
        st.warning("Review sample manifest is empty.")
        return
    if "review_status" not in sample.columns:
        sample["review_status"] = "pending_review"
    status_filter = st.sidebar.multiselect(
        "Status filter",
        ["pending_review", "in_progress", "corrected_and_confirmed", "excluded", "needs_retake", "needs_later_review"],
        default=["pending_review", "in_progress", "corrected_and_confirmed", "needs_later_review"],
    )
    visible = sample[sample["review_status"].isin(status_filter)].copy()
    if visible.empty:
        visible = sample.copy()
    labels = [
        f"{i+1:03d}/{len(visible):03d} {row.review_status} {row.image_name} {row.specimen_id}"
        for i, row in enumerate(visible.itertuples())
    ]
    selected_label = st.sidebar.selectbox("Review image", labels)
    selected_index = labels.index(selected_label)
    row = visible.iloc[selected_index].to_dict()
    _init_session_for_image(row)

    image_name = str(row["image_name"])
    warped_path = path_from_project(PROJECT_ROOT, str(row["warped_image_path"]))
    pre = st.session_state.get("rw_preannotation", {})
    corrected_exists = corrected_json_path(PROJECT_ROOT, image_name, review_dirname).exists()
    st.sidebar.write(
        {
            "image_name": image_name,
            "specimen_id": row.get("specimen_id", ""),
            "source_status": row.get("source_status", ""),
            "review_status": row.get("review_status", ""),
            "corrected_exists": corrected_exists,
        }
    )
    if corrected_exists:
        st.sidebar.info("Existing corrected_keypoints are loaded by default and will not be overwritten unless you save again.")
    st.sidebar.caption(f"Rerun count: {st.session_state.rw_rerun_counter}")
    if st.sidebar.button("Reset page points to AI preannotation"):
        _reset_to_preannotation()
        _log_interaction(
            image_name=image_name,
            selected_keypoint=str(st.session_state.get("rw_selected_key", "")),
            event_type="reset_to_preannotation",
        )
        st.rerun()
    if st.sidebar.button("Exclude this image from review"):
        update_sample_status(PROJECT_ROOT, image_name, "excluded", notes="excluded_in_review_page", review_dirname=review_dirname)
        st.success("Marked excluded.")
        st.rerun()
    status_cols = st.sidebar.columns(2)
    if status_cols[0].button("Needs retake"):
        update_sample_status(PROJECT_ROOT, image_name, "needs_retake", notes="marked_needs_retake_in_review_page", review_dirname=review_dirname)
        st.success("Marked needs_retake.")
        st.rerun()
    if status_cols[1].button("Later review"):
        update_sample_status(PROJECT_ROOT, image_name, "needs_later_review", notes="marked_needs_later_review_in_review_page", review_dirname=review_dirname)
        st.success("Marked needs_later_review.")
        st.rerun()

    if not warped_path.exists():
        st.error(f"Warped image not found: {warped_path}")
        return
    image = _load_image_cached(str(warped_path))
    corrected_full = dict(st.session_state.get("rw_corrected_full", {}) or {})
    corrected_short = full_to_short_keypoints(corrected_full)

    left, right = st.columns([4, 1.35])
    with right:
        st.subheader("Display")
        advanced_settings = st.checkbox(
            "Advanced settings / Developer mode",
            value=bool(st.session_state.get("rw_advanced_settings_enabled", False)),
            key="rw_advanced_settings_enabled",
        )
        if advanced_settings:
            st.markdown("### Measurement axis mode")
            st.selectbox("Measurement axis mode", MEASUREMENT_AXIS_OPTIONS, index=2, key="rw_measurement_axis_mode")
            st.markdown("### Advanced overlay layers")
            st.caption("These switches only control extra visualization layers. They do not overwrite corrected_keypoints or change default measurements.")
            st.caption("TL_compressed_virtual_mm is a parallel candidate for virtual caudal-fin compression. It is kept beside historical projection-based TL until manual validation is complete.")
            show_geometric_p8p9_review = st.checkbox(
                "Show geometric P8/P9",
                value=bool(st.session_state.get("show_geometric_p8p9", False)),
                key="show_geometric_p8p9_review",
            )
            show_body_depth_max_section_review = st.checkbox(
                "Show body-depth max section",
                value=bool(st.session_state.get("show_body_depth_max_section", False)),
                key="show_body_depth_max_section_review",
            )
            show_body_depth_comparison_labels_review = st.checkbox(
                "Show body-depth comparison labels",
                value=bool(st.session_state.get("show_body_depth_comparison_labels", False)),
                key="show_body_depth_comparison_labels_review",
            )
            show_body_depth_outer_boundary_contour_review = st.checkbox(
                "Show body-depth outer boundary contour",
                value=bool(st.session_state.get("show_body_depth_boundary_contour", False)),
                key="show_body_depth_outer_boundary_contour_review",
            )
            show_p6_geometry_qc_review = st.checkbox(
                "Show P6 geometry QC",
                value=bool(st.session_state.get("show_p6_geometry_qc", False)),
                key="show_p6_geometry_qc_review",
            )
            show_p6_fallback_suggestion_review = st.checkbox(
                "Show P6 fallback suggestion",
                value=bool(st.session_state.get("show_p6_fallback_suggestion", False)),
                key="show_p6_fallback_suggestion_review",
            )
            show_tail_fork_gap_region_review = st.checkbox(
                "Show tail fork gap region",
                value=bool(st.session_state.get("show_tail_fork_gap_region", False)),
                key="show_tail_fork_gap_region_review",
            )
            show_geometric_p10p11_review = st.checkbox(
                "Show geometric P10/P11",
                value=bool(st.session_state.get("show_geometric_p10p11", False)),
                key="show_geometric_p10p11_review",
            )
            show_peduncle_depth_min_section_review = st.checkbox(
                "Show peduncle-depth min section",
                value=bool(st.session_state.get("show_peduncle_depth_min_section", False)),
                key="show_peduncle_depth_min_section_review",
            )
            show_peduncle_depth_comparison_labels_review = st.checkbox(
                "Show peduncle-depth comparison labels",
                value=bool(st.session_state.get("show_peduncle_depth_comparison_labels", False)),
                key="show_peduncle_depth_comparison_labels_review",
            )
            show_fin_suppression_regions_review = st.checkbox(
                "Show fin suppression regions",
                value=bool(st.session_state.get("show_fin_suppression_regions", False)),
                key="show_fin_suppression_regions_review",
            )
            show_body_trunk_contours_review = st.checkbox(
                "Show body trunk contours",
                value=bool(st.session_state.get("show_body_trunk_contours", False)),
                key="show_body_trunk_contours_review",
            )
            show_p3_operculum_suggestion_review = st.checkbox(
                "Show P3 operculum suggestion",
                value=bool(st.session_state.get("show_p3_operculum_suggestion", False)),
                key="show_p3_operculum_suggestion_review",
            )
            show_compressed_tail_p7v_review = st.checkbox(
                "Show compressed-tail P7V",
                value=bool(st.session_state.get("show_compressed_tail_p7v", True)),
                key="show_compressed_tail_p7v_review",
            )
            show_open_projection_p7v_review = st.checkbox(
                "Show open-projection P7V",
                value=bool(st.session_state.get("show_open_projection_p7v", False)),
                key="show_open_projection_p7v_review",
            )
            show_tail_folding_arc_reference_review = st.checkbox(
                "Show tail-folding arc reference",
                value=bool(st.session_state.get("show_tail_folding_arc_reference", False)),
                key="show_tail_folding_arc_reference_review",
            )
            show_tl_comparison_labels_review = st.checkbox(
                "Show TL comparison labels",
                value=bool(st.session_state.get("show_tl_comparison_labels", False)),
                key="show_tl_comparison_labels_review",
            )
            show_current_p8p9_line_review = st.checkbox(
                "Show current P8-P9 line",
                value=bool(st.session_state.get("show_current_body_depth_line", False)),
                key="show_current_p8p9_line_review",
            )
            show_current_p10p11_line_review = st.checkbox(
                "Show current P10-P11 line",
                value=bool(st.session_state.get("show_current_peduncle_depth_line", False)),
                key="show_current_p10p11_line_review",
            )
            show_model_axis_review = st.checkbox(
                "Show model axis",
                value=bool(st.session_state.get("rw_show_model_axis", False)),
                key="show_model_axis_overlay_review",
            )
            show_debug_heatmap_points_review = st.checkbox(
                "Show debug heatmap points",
                value=bool(st.session_state.get("rw_show_debug_heatmap_points", False)),
                key="show_debug_heatmap_points_review",
            )
            show_point_source_labels_review = st.checkbox(
                "Show point source labels",
                value=bool(st.session_state.get("rw_show_point_source_labels", False)),
                key="show_point_source_labels_review",
            )
            show_qc_warnings_review = st.checkbox(
                "Show QC warnings",
                value=bool(st.session_state.get("rw_show_qc", True)),
                key="show_qc_warnings_overlay_review",
            )
            st.session_state.show_geometric_p8p9 = show_geometric_p8p9_review
            st.session_state.show_body_depth_max_section = show_body_depth_max_section_review
            st.session_state.show_body_depth_comparison_labels = show_body_depth_comparison_labels_review
            st.session_state.show_body_depth_boundary_contour = show_body_depth_outer_boundary_contour_review
            st.session_state.show_p6_geometry_qc = show_p6_geometry_qc_review
            st.session_state.show_p6_fallback_suggestion = show_p6_fallback_suggestion_review
            st.session_state.show_tail_fork_gap_region = show_tail_fork_gap_region_review
            st.session_state.show_geometric_p10p11 = show_geometric_p10p11_review
            st.session_state.show_peduncle_depth_min_section = show_peduncle_depth_min_section_review
            st.session_state.show_peduncle_depth_comparison_labels = show_peduncle_depth_comparison_labels_review
            st.session_state.show_fin_suppression_regions = show_fin_suppression_regions_review
            st.session_state.show_body_trunk_contours = show_body_trunk_contours_review
            st.session_state.show_p3_operculum_suggestion = show_p3_operculum_suggestion_review
            st.session_state.show_compressed_tail_p7v = show_compressed_tail_p7v_review
            st.session_state.show_open_projection_p7v = show_open_projection_p7v_review
            st.session_state.show_tail_folding_arc_reference = show_tail_folding_arc_reference_review
            st.session_state.show_tl_comparison_labels = show_tl_comparison_labels_review
            st.session_state.show_current_body_depth_line = show_current_p8p9_line_review
            st.session_state.show_current_peduncle_depth_line = show_current_p10p11_line_review
            st.session_state.rw_show_geometric_body_depth = show_geometric_p8p9_review
            st.session_state.rw_show_body_depth_width_profile = show_body_depth_max_section_review
            st.session_state.rw_show_body_depth_comparison_labels = show_body_depth_comparison_labels_review
            st.session_state.rw_show_body_depth_boundary_contour = show_body_depth_outer_boundary_contour_review
            st.session_state.rw_show_p6_geometry_qc = show_p6_geometry_qc_review
            st.session_state.rw_show_p6_fallback_suggestion = show_p6_fallback_suggestion_review
            st.session_state.rw_show_tail_fork_gap_region = show_tail_fork_gap_region_review
            st.session_state.rw_show_geometric_peduncle_depth = show_geometric_p10p11_review
            st.session_state.rw_show_peduncle_depth_width_profile = show_peduncle_depth_min_section_review
            st.session_state.rw_show_peduncle_depth_comparison_labels = show_peduncle_depth_comparison_labels_review
            st.session_state.rw_show_fin_suppression_regions = show_fin_suppression_regions_review
            st.session_state.rw_show_body_trunk_contours = show_body_trunk_contours_review
            st.session_state.rw_show_p3_operculum_suggestion = show_p3_operculum_suggestion_review
            st.session_state.rw_show_compressed_tail_p7v = show_compressed_tail_p7v_review
            st.session_state.rw_show_open_projection_p7v = show_open_projection_p7v_review
            st.session_state.rw_show_tail_folding_arc_reference = show_tail_folding_arc_reference_review
            st.session_state.rw_show_tl_comparison_labels = show_tl_comparison_labels_review
            st.session_state.rw_show_current_body_depth_line = show_current_p8p9_line_review
            st.session_state.rw_show_current_peduncle_depth_line = show_current_p10p11_line_review
            st.session_state.rw_show_model_axis = show_model_axis_review
            st.session_state.rw_show_debug_heatmap_points = show_debug_heatmap_points_review
            st.session_state.rw_show_heatmap = show_debug_heatmap_points_review
            st.session_state.rw_show_point_source_labels = show_point_source_labels_review
            st.session_state.rw_show_qc = show_qc_warnings_review
            st.session_state.rw_overlay_dirty = True
            with st.expander("Overlay debug / audit", expanded=False):
                overlay_state = {
                    "show_geometric_p8p9": show_geometric_p8p9_review,
                    "show_body_depth_max_section": show_body_depth_max_section_review,
                    "show_body_depth_comparison_labels": show_body_depth_comparison_labels_review,
                    "show_body_depth_outer_boundary_contour": show_body_depth_outer_boundary_contour_review,
                    "show_p6_geometry_qc": show_p6_geometry_qc_review,
                    "show_p6_fallback_suggestion": show_p6_fallback_suggestion_review,
                    "show_tail_fork_gap_region": show_tail_fork_gap_region_review,
                    "show_geometric_p10p11": show_geometric_p10p11_review,
                    "show_peduncle_depth_min_section": show_peduncle_depth_min_section_review,
                    "show_peduncle_depth_comparison_labels": show_peduncle_depth_comparison_labels_review,
                    "show_fin_suppression_regions": show_fin_suppression_regions_review,
                    "show_body_trunk_contours": show_body_trunk_contours_review,
                    "show_p3_operculum_suggestion": show_p3_operculum_suggestion_review,
                    "show_compressed_tail_p7v": show_compressed_tail_p7v_review,
                    "show_open_projection_p7v": show_open_projection_p7v_review,
                    "show_tail_folding_arc_reference": show_tail_folding_arc_reference_review,
                    "show_tl_comparison_labels": show_tl_comparison_labels_review,
                    "show_current_p8p9_line": show_current_p8p9_line_review,
                    "show_current_p10p11_line": show_current_p10p11_line_review,
                    "show_model_axis": show_model_axis_review,
                    "show_debug_heatmap_points": show_debug_heatmap_points_review,
                    "show_point_source_labels": show_point_source_labels_review,
                    "show_qc_warnings": show_qc_warnings_review,
                }
                st.write(overlay_state)
                display_metadata = _metadata_with_measurement_axes(image)
                bdg = display_metadata.get("body_depth_geometry", {}) if isinstance(display_metadata.get("body_depth_geometry", {}), Mapping) else {}
                pdg = display_metadata.get("peduncle_depth_geometry", {}) if isinstance(display_metadata.get("peduncle_depth_geometry", {}), Mapping) else {}
                p6g = display_metadata.get("p6_gap_derivation", {}) if isinstance(display_metadata.get("p6_gap_derivation", {}), Mapping) else {}
                tqc = display_metadata.get("tail_qc", {}) if isinstance(display_metadata.get("tail_qc", {}), Mapping) else {}
                oqc = display_metadata.get("operculum_qc", {}) if isinstance(display_metadata.get("operculum_qc", {}), Mapping) else {}
                axes = display_metadata.get("measurement_axes", {}) if isinstance(display_metadata.get("measurement_axes", {}), Mapping) else {}
                st.write(
                    {
                        "body_depth_geometry_available": bool(bdg),
                        "peduncle_depth_geometry_available": bool(pdg),
                        "p6_gap_derivation_available": bool(p6g or tqc.get("P6_fallback_candidate")),
                        "tail_gap_region_available": bool(p6g.get("tail_gap_centerline") or p6g.get("tail_gap_tip")),
                        "fin_suppression_regions_available": bool(bdg.get("fin_suppression_regions")),
                        "body_trunk_contours_available": bool(bdg.get("dorsal_trunk_contour") or bdg.get("ventral_trunk_contour")),
                        "p3_operculum_suggestion_available": bool(oqc.get("P3_operculum_edge_suggestion")),
                        "model_axis_available": bool(axes.get("model_axis")),
                        "heatmap_debug_available": bool(display_metadata.get("comparison_preannotation") or display_metadata.get("heatmap_keypoints")),
                    }
                )
            st.markdown("### Other overlay/debug layers")
            st.checkbox("Show v0.3.4 points", value=False, key="rw_show_v034")
            st.checkbox("Show suggestions", value=True, key="rw_show_suggestions")
            st.checkbox("Show local-normal suggestions", value=True, key="rw_show_local_normal")
            st.checkbox("Show section lines", value=True, key="rw_show_width_profiles")
            st.checkbox("Show curvature QC", value=True, key="rw_show_curvature")
        else:
            st.session_state.show_geometric_p8p9 = False
            st.session_state.show_body_depth_max_section = False
            st.session_state.show_body_depth_comparison_labels = False
            st.session_state.show_body_depth_boundary_contour = False
            st.session_state.show_p6_geometry_qc = False
            st.session_state.show_p6_fallback_suggestion = False
            st.session_state.show_tail_fork_gap_region = False
            st.session_state.show_geometric_p10p11 = True
            st.session_state.show_peduncle_depth_min_section = True
            st.session_state.show_peduncle_depth_comparison_labels = False
            st.session_state.show_body_trunk_contours = False
            st.session_state.show_compressed_tail_p7v = True
            st.session_state.show_open_projection_p7v = False
            st.session_state.show_tail_folding_arc_reference = False
            st.session_state.show_tl_comparison_labels = False
            st.session_state.rw_show_geometric_body_depth = False
            st.session_state.rw_show_body_depth_width_profile = False
            st.session_state.rw_show_body_depth_comparison_labels = False
            st.session_state.rw_show_body_depth_boundary_contour = False
            st.session_state.rw_show_p6_geometry_qc = False
            st.session_state.rw_show_p6_fallback_suggestion = False
            st.session_state.rw_show_tail_fork_gap_region = False
            st.session_state.rw_show_geometric_peduncle_depth = True
            st.session_state.rw_show_peduncle_depth_width_profile = True
            st.session_state.rw_show_peduncle_depth_comparison_labels = False
            st.session_state.rw_show_current_body_depth_line = False
            st.session_state.rw_show_current_peduncle_depth_line = False
            st.session_state.rw_show_body_trunk_contours = False
            st.session_state.rw_show_compressed_tail_p7v = True
            st.session_state.rw_show_open_projection_p7v = False
            st.session_state.rw_show_tail_folding_arc_reference = False
            st.session_state.rw_show_tl_comparison_labels = False
            st.session_state.rw_show_debug_heatmap_points = False
            st.session_state.rw_show_point_source_labels = False
            st.info("Recommended workflow: v0.6 AI-assisted preannotation + auto_qc_gated measurement axis.")
        pre_metadata = pre.get("preannotation_metadata", {}) if isinstance(pre, Mapping) else {}
        tail_qc_v065 = pre_metadata.get("tail_qc", {}) if isinstance(pre_metadata, Mapping) and isinstance(pre_metadata.get("tail_qc", {}), Mapping) else {}
        if tail_qc_v065.get("P6_geometry_qc_pass") is False and not advanced_settings:
            st.warning("P6 可能不可靠，请检查尾鳍分叉点。")
        current_axis_mode = _current_measurement_axis_mode()
        if st.session_state.get("rw_last_measurement_axis_mode") != current_axis_mode:
            st.session_state.rw_last_measurement_axis_mode = current_axis_mode
            st.session_state.rw_overlay_dirty = True
        st.caption("测量轴模式只影响测量轴与派生测量值，不会覆盖已确认的 corrected_keypoints 或 C1–C4 标注点。")
        if advanced_settings:
            st.checkbox("Show model axis", value=True, key="rw_show_model_axis")
            st.checkbox("Show body midline axis", value=True, key="rw_show_body_midline_axis")
            st.checkbox("Show geometric C points", value=True, key="rw_show_geometric_c_points")
            st.checkbox("Show body midline contour", value=True, key="rw_show_body_midline_contour")
            st.checkbox("Show dual-axis comparison", value=True, key="rw_show_dual_axis")
            st.checkbox("Show selected measurement axis only", value=False, key="rw_show_selected_axis_only")
            with st.expander("Geometric body-depth debug", expanded=False):
                display_metadata = _metadata_with_measurement_axes(image)
                body_depth_geometry = display_metadata.get("body_depth_geometry", {}) if isinstance(display_metadata.get("body_depth_geometry", {}), Mapping) else {}
                geometry_info = display_metadata.get("active_body_depth_geometry_info", {}) if isinstance(display_metadata.get("active_body_depth_geometry_info", {}), Mapping) else {}
                legacy_detected = bool(geometry_info.get("legacy_body_depth_geometry_detected", False))
                if legacy_detected:
                    st.caption("Legacy body-depth geometry ignored for overlay. Recomputed v0.6.6 display-only geometry is used when available.")
                if not body_depth_geometry:
                    msg = geometry_info.get("geometry_error") or "No geometric body-depth data available. Run AI-assisted preannotation or recompute body-depth geometry."
                    st.caption(str(msg))
                else:
                    current_measurements = calculate_measurements(corrected_short, MM_PER_PIXEL) if len(corrected_short) == len(KEYPOINT_DEFS) else {}
                    current_mm = current_measurements.get("body_depth_mm") if isinstance(current_measurements, Mapping) else None
                    geo_mm_raw = body_depth_geometry.get("body_depth_geometric_mm")
                    try:
                        geo_mm = float(geo_mm_raw) if geo_mm_raw not in ("", None) else None
                    except (TypeError, ValueError):
                        geo_mm = None
                    try:
                        current_mm_float = float(current_mm) if current_mm not in ("", None) else None
                    except (TypeError, ValueError):
                        current_mm_float = None
                    diff_mm = abs(current_mm_float - geo_mm) if current_mm_float is not None and geo_mm is not None else None
                    st.write(
                        {
                            "P8_geometric": body_depth_geometry.get("P8_geometric", ""),
                            "P9_geometric": body_depth_geometry.get("P9_geometric", ""),
                            "active_geometry_version": geometry_info.get("active_geometry_version", body_depth_geometry.get("body_depth_geometry_version", "")),
                            "active_body_depth_source": geometry_info.get("active_body_depth_source", body_depth_geometry.get("body_depth_source", "")),
                            "geometry_came_from": geometry_info.get("geometry_origin", ""),
                            "legacy_body_depth_geometry_detected": legacy_detected,
                            "body_depth_geometric_mm": geo_mm,
                            "body_depth_geometric_qc_pass": body_depth_geometry.get("body_depth_geometric_qc_pass", ""),
                            "body_depth_geometric_review_reason": body_depth_geometry.get("body_depth_geometric_review_reason", ""),
                            "current_body_depth_mm": current_mm_float,
                            "difference_mm": diff_mm,
                        }
                    )
        selected_key = st.selectbox("Point to edit", KEYPOINT_KEYS, key="rw_selected_key")
        st.session_state.selected_keypoint = selected_key
        specimen_id_value = st.text_input(
            "Specimen ID",
            value=str((st.session_state.get("rw_preannotation", {}) or {}).get("specimen_id", row.get("specimen_id", ""))),
            help="For 5.24 images this can be filled from the handwritten label before saving.",
            key=f"rw_specimen_id_input_{image_name}",
        )
        if isinstance(st.session_state.get("rw_preannotation"), dict):
            st.session_state.rw_preannotation["specimen_id"] = specimen_id_value.strip() or "unknown"
        st.caption("Left click records a pending coordinate. Click Apply to update the selected point. Right click toggles 3x zoom.")
        pending = st.session_state.get("pending_click_xy")
        pending_key = str(st.session_state.get("pending_click_keypoint", ""))
        if isinstance(pending, dict) and pending_key:
            st.info(
                f"Pending new coordinate for {pending_key}: "
                f"x={float(pending['x']):.1f}, y={float(pending['y']):.1f}"
            )
            apply_cols = st.columns(2)
            if apply_cols[0].button("Apply to selected keypoint", type="primary"):
                if _apply_pending_click(image_name):
                    st.rerun()
            if apply_cols[1].button("Clear pending click"):
                _clear_pending_click(image_name)
                st.rerun()
        else:
            st.caption("No pending click yet.")
        if st.button("Undo last edit"):
            if _undo_last_edit(image_name):
                st.rerun()
            else:
                st.warning("No edit to undo.")
        notes = st.text_area("Reviewer notes", key="rw_reviewer_notes")
        if st.button("Save corrected annotation", type="primary"):
            paths = save_corrected_review_annotation(
                PROJECT_ROOT,
                image_name=image_name,
                warped_image=image,
                corrected_full=st.session_state.rw_corrected_full,
                preannotation_payload=st.session_state.get("rw_preannotation", pre),
                review_start_time=str(st.session_state.get("rw_review_start_time", "")),
                reviewer_notes=notes,
                mm_per_pixel=MM_PER_PIXEL,
                measurement_axis_mode=_current_measurement_axis_mode(),
                review_dirname=review_dirname,
            )
            st.success(f"Saved corrected annotation: {paths['json']}")
            st.rerun()
        st.subheader("QC")
        metadata = _metadata_with_measurement_axes(image)
        curvature = metadata.get("curvature_qc", {}) if isinstance(metadata.get("curvature_qc"), dict) else {}
        measurement_axes = metadata.get("measurement_axes", {}) if isinstance(metadata.get("measurement_axes"), dict) else {}
        dual_cmp = measurement_axes.get("dual_axis_comparison", {}) if isinstance(measurement_axes.get("dual_axis_comparison"), dict) else {}
        model_axis_meta = measurement_axes.get("model_axis") if isinstance(measurement_axes.get("model_axis"), dict) else {}
        body_axis_meta = measurement_axes.get("body_midline_axis") if isinstance(measurement_axes.get("body_midline_axis"), dict) else {}
        st.write(
            {
                "review_reason": pre.get("review_reason", ""),
                "p7v_valid": metadata.get("p7v_valid"),
                "curvature_qc_level": curvature.get("curvature_qc_level", ""),
                "curvature_index": curvature.get("curvature_index", ""),
                "body_depth_refined_mm": metadata.get("body_depth_refined_mm", ""),
                "peduncle_depth_refined_mm": metadata.get("caudal_peduncle_depth_refined_mm", ""),
                "selected_measurement_axis": measurement_axes.get("selected_measurement_axis", ""),
                "measurement_axis_mode_request": measurement_axes.get("measurement_axis_mode_request", ""),
                "TL_curve_model_axis_mm": model_axis_meta.get("TL_curve_mm", ""),
                "SL_curve_model_axis_mm": model_axis_meta.get("SL_curve_mm", ""),
                "curvature_index_model_axis": model_axis_meta.get("curvature_index", ""),
                "axis_smoothness_model_axis": model_axis_meta.get("axis_smoothness", ""),
                "TL_curve_body_midline_axis_mm": body_axis_meta.get("TL_curve_mm", ""),
                "SL_curve_body_midline_axis_mm": body_axis_meta.get("SL_curve_mm", ""),
                "curvature_index_body_midline_axis": body_axis_meta.get("curvature_index", ""),
                "axis_smoothness_body_midline_axis": body_axis_meta.get("axis_smoothness", ""),
                "TL_curve_selected_mm": measurement_axes.get("TL_curve_selected_mm", ""),
                "SL_curve_selected_mm": measurement_axes.get("SL_curve_selected_mm", ""),
                "curvature_index_selected": measurement_axes.get("curvature_index_selected", ""),
                "TL_curve_axis_diff_mm": dual_cmp.get("TL_curve_diff_mm", ""),
                "SL_curve_axis_diff_mm": dual_cmp.get("SL_curve_diff_mm", ""),
                "curvature_index_axis_diff": dual_cmp.get("curvature_index_diff", ""),
                "dual_axis_disagreement": dual_cmp.get("dual_axis_disagreement", ""),
                "measurement_axis_review_reason": measurement_axes.get("measurement_axis_review_reason", ""),
            }
        )
        with st.expander("Coordinate debug", expanded=False):
            debug = st.session_state.get("rw_coordinate_debug", {})
            if isinstance(debug, dict) and debug:
                st.write(debug)
            else:
                st.caption("No click has been recorded yet.")

    with left:
        st.subheader(f"{image_name} | {row.get('specimen_id', '')}")
        if len(corrected_short) == len(KEYPOINT_DEFS):
            measurements = calculate_measurements(corrected_short, MM_PER_PIXEL)
        else:
            measurements = {}
        base_overlay = image
        overlay = draw_enhanced_preannotation_overlay(base_overlay, _metadata_with_measurement_axes(image))
        display, transform = prepare_zoom_display(
            overlay,
            max_width=1200,
            zoom_mode=bool(st.session_state.get("rw_zoom_mode", False)),
            zoom_center=st.session_state.get("rw_zoom_center"),
            zoom_factor=3.0,
        )
        display_for_click = _draw_pending_click_preview(display)
        click = image_clicker(display_for_click, key=f"rw_clicker_{image_name}_{st.session_state.get('rw_zoom_mode', False)}")
        _handle_image_click(
            click,
            transform,
            image_name,
            str(st.session_state.get("rw_selected_key", selected_key)),
            display=display_for_click,
            original_shape=image.shape,
        )
        pending = st.session_state.get("pending_click_xy")
        pending_key = str(st.session_state.get("pending_click_keypoint", ""))
        if isinstance(pending, dict) and pending_key:
            st.info(
                f"Pending click recorded for {pending_key}: "
                f"x={float(pending['x']):.1f}, y={float(pending['y']):.1f}. "
                "Use the Apply button on the right to update the point."
            )

        corrected_preview = corrected_preview_path(PROJECT_ROOT, image_name, review_dirname)
        if corrected_exists and corrected_preview.exists():
            st.caption(f"Corrected preview: {corrected_preview}")
        st.caption(f"Workflow output root: {root}")


if __name__ == "__main__":
    main()
