from __future__ import annotations

import json
from hashlib import sha256
from io import BytesIO
from datetime import datetime
import os
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

from siganusmorph.components import measurement_editor
from siganusmorph.formal_ui import (
    calibrate_with_aruco,
    check_formal_coordinate_consistency,
    dataframe_to_csv_bytes,
    dataframe_to_excel_bytes,
    formal_editor_payload,
    formal_points_to_full_keypoints,
    image_bytes_png,
    image_from_upload,
    recalculate_formal_measurements,
    run_recommended_measurement,
    safe_name,
    simplified_measurement_row,
    update_formal_points_from_editor,
)
from siganusmorph.remote_jobs import (
    ACTIVE_STATUSES,
    get_remote_job,
    load_remote_result,
    remote_compute_enabled,
    submit_remote_job,
)
from siganusmorph.ui_styles import (
    apply_formal_theme,
    fmt_mm,
    metric_grid,
    page_header,
    status_badge,
    step_header,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMAL_OUTPUT_ROOT = PROJECT_ROOT / "results" / "formal_v1_user_outputs"
REMOTE_COMPUTE_ENABLED = remote_compute_enabled()

apply_formal_theme()
page_header(
    "单鱼测量",
    "导入单张标准化照片，系统自动完成校准、鱼体识别、形态测量，并支持人工拖拽修正。",
    eyebrow="V1.0 Measurement Workspace",
)

if "formal_single" not in st.session_state:
    st.session_state.formal_single = {}

state = st.session_state.formal_single


def has_uploaded_image() -> bool:
    return bool(state.get("raw_image_bytes")) if REMOTE_COMPUTE_ENABLED else "raw_image" in state


def lightweight_upload_preview(image_bytes: bytes, max_side: int = 1280) -> Image.Image | None:
    """Decode only a display-sized preview on the low-resource web server."""
    try:
        with Image.open(BytesIO(image_bytes)) as source:
            source.draft("RGB", (max_side, max_side))
            preview = ImageOps.exif_transpose(source).convert("RGB")
            preview.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            return preview.copy()
    except Exception:
        return None


def lightweight_array_preview(image: Any, max_side: int = 1280) -> Image.Image | None:
    try:
        preview = Image.fromarray(image).convert("RGB")
        preview.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return preview
    except Exception:
        return None


def draw_lightweight_editor_preview(
    display: Image.Image,
    points: list[dict[str, Any]],
    lines: list[dict[str, Any]],
    polylines: list[dict[str, Any]],
    stars: list[dict[str, Any]],
) -> Image.Image:
    """Render a compact preview from display-space editor objects."""
    canvas = display.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    point_map = {str(point.get("name")): point for point in points}
    for polyline in polylines:
        coords = polyline.get("points", [])
        xy = [
            (float(item[0]), float(item[1]))
            for item in coords
            if isinstance(item, (list, tuple)) and len(item) >= 2
        ]
        if len(xy) >= 2:
            draw.line(
                xy,
                fill=str(polyline.get("color", "#00aee8")),
                width=max(1, int(polyline.get("width", 2))),
            )
    for line in lines:
        a = point_map.get(str(line.get("a")))
        b = point_map.get(str(line.get("b")))
        if a and b:
            draw.line(
                (float(a["x"]), float(a["y"]), float(b["x"]), float(b["y"])),
                fill=str(line.get("color", "#ffffff")),
                width=max(1, int(line.get("width", 2))),
            )
    for point in points:
        x, y = float(point["x"]), float(point["y"])
        radius = max(4, int(point.get("radius", 5)))
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=str(point.get("color", "#ffffff")),
            outline="#202020",
            width=2,
        )
    for star in stars:
        x, y = float(star.get("x", 0)), float(star.get("y", 0))
        radius = max(6, int(star.get("radius", 7)))
        draw.polygon(
            ((x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)),
            fill=str(star.get("color", "#a055ff")),
            outline="#ffffff",
        )
    return canvas


def clear_downstream_state() -> None:
    """Clear measurement state when raw image or calibration changes."""
    for key in (
        "working_image",
        "measurement_result",
        "measurement_error",
        "formal_measurement_points",
        "formal_measurements",
        "formal_derived_points",
        "measurement_overrides",
        "last_formal_editor_event_ts",
        "overlay",
        "applied_changes",
        "unsaved_changes",
        "saved_at",
        "remote_job_id",
        "remote_job_status",
        "remote_job_loaded",
        "raw_image_bytes",
        "raw_image_preview",
        "raw_image_upload_id",
        "warped_display_preview",
    ):
        state.pop(key, None)


def set_calibration(calibration_state: dict[str, Any]) -> None:
    """Set formal calibration state and clear stale measurement results."""
    state["formal_calibration_state"] = calibration_state
    if calibration_state.get("status") == "success":
        state["working_image"] = calibration_state["warped_image"]
        state["mm_per_pixel"] = float(calibration_state["mm_per_pixel"])
    else:
        state.pop("working_image", None)
        state.pop("mm_per_pixel", None)
    for key in (
        "measurement_result",
        "measurement_error",
        "formal_measurement_points",
        "formal_measurements",
        "formal_derived_points",
        "measurement_overrides",
        "last_formal_editor_event_ts",
        "overlay",
        "saved_at",
    ):
        state.pop(key, None)


def measurement_status() -> tuple[str, str]:
    if not has_uploaded_image():
        return "未上传", "neutral"
    if state.get("measurement_result"):
        return "测量完成", "success"
    if state.get("remote_job_status") in ACTIVE_STATUSES:
        labels = {
            "queued": "排队等待",
            "claimed": "计算节点已领取",
            "processing": "正在远程测量",
        }
        return labels.get(str(state.get("remote_job_status")), "远程测量中"), "info"
    if state.get("measurement_error"):
        if "校准" in str(state["measurement_error"]):
            return "校准失败", "error"
        return "需要复核", "warning"
    return "等待测量", "info"


def run_auto_measurement_pipeline() -> None:
    """Run formal calibration and measurement as one user-facing action."""
    raw_image = state.get("raw_image")
    raw_image_bytes = state.get("raw_image_bytes")
    if REMOTE_COMPUTE_ENABLED and not raw_image_bytes:
        state["measurement_error"] = "请先上传单张图片。"
        return
    if not REMOTE_COMPUTE_ENABLED and raw_image is None:
        state["measurement_error"] = "请先上传单张图片。"
        return

    if REMOTE_COMPUTE_ENABLED:
        try:
            job = submit_remote_job(
                bytes(raw_image_bytes),
                image_name=state.get("image_name", "uploaded_image"),
                specimen_id=state.get("specimen_id", ""),
                weight_g=state.get("formal_weight_g") if state.get("formal_weight_g") not in ("", None) else None,
                project_root=PROJECT_ROOT,
            )
        except Exception as exc:  # noqa: BLE001 - normal UI error handling.
            state["measurement_error"] = f"远程测量任务提交失败：{exc}"
            return
        state["remote_job_id"] = job.get("job_id")
        state["remote_job_status"] = job.get("status", "queued")
        state["remote_job_loaded"] = ""
        state["measurement_error"] = ""
        return

    state["measurement_error"] = ""
    with st.status("正在校准图像...", expanded=True) as progress:
        progress.write("正在检测校准标记...")
        calibration_state = calibrate_with_aruco(raw_image)
        set_calibration(calibration_state)
        calibration_ok, calibration_errors = check_formal_coordinate_consistency(calibration_state)
        if not calibration_ok:
            reason = calibration_state.get("calibration_message") or "；".join(calibration_errors)
            state["measurement_error"] = f"校准失败：{reason}"
            progress.update(label="校准失败", state="error")
            return

        progress.write("正在识别鱼体...")
        progress.write("正在计算形态参数...")
        try:
            result = run_recommended_measurement(
                state["working_image"],
                state.get("image_name", "uploaded_image"),
                PROJECT_ROOT,
                mm_per_pixel=float(state["mm_per_pixel"]),
            )
            points_ok, point_errors = check_formal_coordinate_consistency(
                state["formal_calibration_state"],
                formal_points=result.get("formal_measurement_points", {}),
            )
            if not points_ok:
                raise ValueError("; ".join(point_errors))
        except Exception as exc:  # noqa: BLE001 - displayed to normal user.
            state["measurement_error"] = str(exc)
            progress.update(label="自动测量未完成", state="error")
        else:
            state["measurement_result"] = result
            state["formal_measurement_points"] = dict(result.get("formal_measurement_points", {}))
            state["formal_measurements"] = dict(result.get("formal_measurements", {}))
            state["formal_derived_points"] = dict(result.get("formal_derived_points", {}))
            state["measurement_overrides"] = {}
            state["measurement_error"] = ""
            state["unsaved_changes"] = False
            progress.write("测量完成。")
            progress.update(label="测量完成", state="complete")


def _apply_remote_result(payload: Mapping[str, Any]) -> None:
    calibration = dict(payload.get("calibration_state", {}) or {})
    if payload.get("raw_marker_overlay") is not None:
        calibration["raw_marker_overlay"] = payload["raw_marker_overlay"]
    if payload.get("warped_image") is not None:
        calibration["warped_image"] = payload["warped_image"]
        state["warped_display_preview"] = lightweight_array_preview(payload["warped_image"])
    set_calibration(calibration)
    if payload.get("status") != "success":
        state["measurement_error"] = str(
            calibration.get("calibration_message") or "校准失败，需要人工复核或重新拍摄。"
        )
        return
    result = dict(payload.get("measurement_result", {}) or {})
    result.setdefault("raw_result", {})
    result.setdefault("short_keypoints", {})
    state["measurement_result"] = result
    state["formal_measurement_points"] = dict(result.get("formal_measurement_points", {}) or {})
    state["formal_measurements"] = dict(result.get("formal_measurements", {}) or {})
    state["formal_derived_points"] = dict(result.get("formal_derived_points", {}) or {})
    state["measurement_overrides"] = {}
    state["measurement_error"] = ""
    state["unsaved_changes"] = False


@st.fragment(run_every=2.0)
def render_remote_job_monitor() -> None:
    if not REMOTE_COMPUTE_ENABLED:
        return
    job_id = str(state.get("remote_job_id") or "")
    if not job_id:
        st.caption("自动测量由远程计算节点执行，本页保持轻量响应。")
        return
    job = get_remote_job(job_id, project_root=PROJECT_ROOT)
    if not job:
        st.error("找不到远程测量任务。")
        return
    state["remote_job_status"] = job.get("status", "")
    status = str(job.get("status") or "")
    progress = str(job.get("progress") or status)
    if status in ACTIVE_STATUSES:
        st.info(f"计算任务：{progress}")
        return
    if status == "failed":
        state["measurement_error"] = str(job.get("error_message") or "远程测量失败")
        st.error(state["measurement_error"])
        return
    if status == "completed" and state.get("remote_job_loaded") != job_id:
        try:
            payload = load_remote_result(job_id, project_root=PROJECT_ROOT)
            _apply_remote_result(payload)
        except Exception as exc:  # noqa: BLE001 - normal UI error handling.
            state["measurement_error"] = f"读取远程结果失败：{exc}"
            st.error(state["measurement_error"])
            return
        state["remote_job_loaded"] = job_id
        st.rerun()


def current_measurement_row(notes: str) -> dict[str, Any]:
    measurements = state.get("formal_measurements") or {}
    return simplified_measurement_row(
        measurements,
        image_name=state.get("image_name", ""),
        specimen_id=state.get("specimen_id", ""),
        weight_g=state.get("formal_weight_g", ""),
        notes=notes,
    )


def render_result_cards(row: Mapping[str, Any]) -> None:
    metric_grid(
        [
            ("压拢尾鳍全长 TL", fmt_mm(row.get("TL_compressed_virtual_mm")), " mm"),
            ("标准长 SL", fmt_mm(row.get("SL_mm")), " mm"),
            ("叉长 FL", fmt_mm(row.get("FL_mm")), " mm"),
            ("体高", fmt_mm(row.get("body_depth_mm")), " mm"),
            ("尾柄高", fmt_mm(row.get("caudal_peduncle_depth_mm")), " mm"),
            ("历史投影 TL", fmt_mm(row.get("TL_open_projection_mm")), " mm"),
        ]
    )


def save_formal_outputs(
    *,
    export_df: pd.DataFrame,
    export_row: dict[str, Any],
    corrected_payload: dict[str, Any],
    preview: Any,
    export_formal_points: Mapping[str, Any],
    export_measurements: Mapping[str, Any],
) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(state.get("image_name", "single_fish")).stem
    out_dir = FORMAL_OUTPUT_ROOT / "confirmed_sessions"
    preview_dir = FORMAL_OUTPUT_ROOT / "previews"
    measurement_dir = FORMAL_OUTPUT_ROOT / "measurements"
    points_dir = FORMAL_OUTPUT_ROOT / "formal_points"
    metadata_dir = FORMAL_OUTPUT_ROOT / "preview_metadata"
    for directory in (out_dir, preview_dir, measurement_dir, points_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{stem}_{timestamp}.json"
    preview_path = preview_dir / f"{stem}_{timestamp}.png"
    measurement_csv_path = measurement_dir / f"{stem}_{timestamp}_measurement.csv"
    measurement_json_path = measurement_dir / f"{stem}_{timestamp}_measurement.json"
    measurement_xlsx_path = measurement_dir / f"{stem}_{timestamp}_measurement.xlsx"
    points_json_path = points_dir / f"{stem}_{timestamp}_formal_points.json"
    preview_meta_path = metadata_dir / f"{stem}_{timestamp}_preview_metadata.json"

    json_path.write_text(json.dumps(corrected_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    preview_path.write_bytes(image_bytes_png(preview))
    export_df.to_csv(measurement_csv_path, index=False, encoding="utf-8-sig")
    export_df.to_excel(measurement_xlsx_path, index=False)
    measurement_json_path.write_text(json.dumps(export_row, ensure_ascii=False, indent=2), encoding="utf-8")
    points_json_path.write_text(
        json.dumps(
            {
                "image_name": state.get("image_name", ""),
                "specimen_id": state.get("specimen_id", ""),
                "weight_g": state.get("formal_weight_g", ""),
                "coordinate_space": "warped_image",
                "formal_measurement_points": export_formal_points,
                "derived_points": state.get("formal_derived_points", {}),
                "measurement_overrides": state.get("measurement_overrides", {}),
                "measurements": export_measurements,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    preview_meta_path.write_text(
        json.dumps(
            {
                "image_name": state.get("image_name", ""),
                "specimen_id": state.get("specimen_id", ""),
                "weight_g": state.get("formal_weight_g", ""),
                "preview_path": str(preview_path),
                "measurement_csv_path": str(measurement_csv_path),
                "measurement_json_path": str(measurement_json_path),
                "measurement_xlsx_path": str(measurement_xlsx_path),
                "points_json_path": str(points_json_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    state["unsaved_changes"] = False
    state["saved_at"] = datetime.now().isoformat(timespec="seconds")


with st.container(border=True):
    step_header("1", "导入与自动测量", "上传照片，填写样本信息，系统自动完成校准和测量。")
    upload_col, info_col = st.columns([1.15, 1])

    with upload_col:
        uploaded = st.file_uploader("上传单张鱼体照片", type=["png", "jpg", "jpeg", "heic", "tif", "tiff"])
        if uploaded is not None:
            image_name = safe_name(uploaded.name)
            uploaded_bytes = uploaded.getvalue()
            upload_id = f"{image_name}:{len(uploaded_bytes)}:{sha256(uploaded_bytes).hexdigest()[:16]}"
            if state.get("raw_image_upload_id") != upload_id:
                clear_downstream_state()
                state.pop("formal_calibration_state", None)
                state.pop("manual_corner_rows", None)
                state["specimen_id"] = ""
                state["formal_specimen_id"] = ""
                state["formal_weight_g"] = ""
            if REMOTE_COMPUTE_ENABLED:
                state["raw_image_bytes"] = uploaded_bytes
                state["raw_image_preview"] = lightweight_upload_preview(uploaded_bytes)
                state["raw_image_upload_id"] = upload_id
                image = state.get("raw_image_preview")
            else:
                image = image_from_upload(uploaded)
                state["raw_image"] = image
                state["image"] = image
                state["raw_image_upload_id"] = upload_id
            state["image_name"] = image_name
            state["raw_image_name"] = image_name
            state.setdefault("specimen_id", "")
            state.setdefault("formal_specimen_id", state.get("specimen_id", ""))
            state.setdefault("formal_weight_g", "")
            st.success(f"已导入：{image_name}")
            if image is not None:
                st.image(image, caption="原始照片预览（轻量缩略图）", use_container_width=True)
            else:
                st.caption("图片已导入。当前格式不生成云端预览，完整原图仍会发送到计算节点。")
        else:
            st.info("请先上传一张标准化侧位照片。")

    with info_col:
        st.markdown("#### 样本信息")
        specimen_id = st.text_input(
            "样本编号 specimen_id",
            value=state.get("formal_specimen_id", state.get("specimen_id", "")),
            placeholder="例如 fish_001",
        )
        raw_weight = state.get("formal_weight_g", "")
        try:
            weight_default = float(raw_weight) if raw_weight not in ("", None) else None
        except (TypeError, ValueError):
            weight_default = None
        weight_g = st.number_input(
            "样品重量 weight_g（g）",
            min_value=0.0,
            value=weight_default,
            step=0.01,
            format="%.2f",
            placeholder="例如 155.33",
        )
        st.caption("可选。若已称重，可填写样品重量，用于导出记录和后续长度-重量分析。")
        state["formal_specimen_id"] = specimen_id.strip()
        state["specimen_id"] = state["formal_specimen_id"]
        state["formal_weight_g"] = float(weight_g) if weight_g is not None else ""

        label, badge_status = measurement_status()
        st.markdown(status_badge(label, badge_status), unsafe_allow_html=True)

        start_disabled = not has_uploaded_image() or state.get("remote_job_status") in ACTIVE_STATUSES
        if st.button("开始自动测量", type="primary", disabled=start_disabled, use_container_width=True):
            run_auto_measurement_pipeline()

        render_remote_job_monitor()

        if state.get("measurement_error"):
            st.error(f"自动测量未完成：{state['measurement_error']}")
        elif state.get("measurement_result"):
            st.success("测量完成。请在下方工作区复核、修正并导出结果。")

    with st.expander("校准详情", expanded=False):
        calibration = state.get("formal_calibration_state")
        if not has_uploaded_image():
            st.info("上传图片并开始自动测量后，可在这里查看校准详情。")
        elif calibration:
            status_value = calibration.get("status", "unknown")
            if status_value == "success":
                st.success("校准成功，后续测量将在校正图像坐标中进行。")
            else:
                st.error("校准失败，不能进入自动测量。")
            detail_cols = st.columns(3)
            with detail_cols[0]:
                st.metric("校准状态", status_value)
            with detail_cols[1]:
                st.metric("检测到的 marker 数量", calibration.get("marker_count", 0))
            with detail_cols[2]:
                mm_value = calibration.get("mm_per_pixel")
                st.metric("mm/pixel", f"{float(mm_value):.6f}" if mm_value not in ("", None) else "不可用")

            raw_col, warped_col = st.columns(2)
            with raw_col:
                st.markdown("**原始照片检测预览**")
                fallback_raw = state.get("raw_image")
                if fallback_raw is None:
                    fallback_raw = state.get("raw_image_preview")
                marker_overlay = calibration.get("raw_marker_overlay")
                if marker_overlay is None:
                    marker_overlay = fallback_raw
                if marker_overlay is not None:
                    st.image(marker_overlay, use_container_width=True)
            with warped_col:
                st.markdown("**校正后图像**")
                if status_value == "success" and calibration.get("warped_image") is not None:
                    warped = calibration["warped_image"]
                    warped_preview = state.get("warped_display_preview")
                    st.image(
                        warped_preview if warped_preview is not None else warped,
                        caption="后续自动测量使用该图像。",
                        use_container_width=True,
                    )
                    st.metric("校正图像尺寸", f"{warped.shape[1]} × {warped.shape[0]} px")
                else:
                    st.warning("无可用校正图像。请重新拍摄，或进入高级工具检查校准点。")
        else:
            st.info("尚未开始自动测量。")

result = state.get("measurement_result")

with st.container(border=True):
    step_header("2", "复核、修正与导出", "在测量工作区拖拽修正点位，应用修改后保存或导出结果。")
    if not result:
        st.info("请先在 Step 1 完成导入与自动测量。")
    else:
        export_measurements = state.get("formal_measurements") or result.get("formal_measurements", {})
        export_formal_points = state.get("formal_measurement_points") or result.get("formal_measurement_points", {})
        export_keypoints = formal_points_to_full_keypoints(export_formal_points, result["full_keypoints"])
        result_note = "用户修正后结果" if state.get("measurement_overrides") else "自动测量结果，需要人工确认"
        current_row = current_measurement_row(result_note)
        export_df = pd.DataFrame([current_row])
        corrected_payload = {
            "image_name": state.get("image_name", ""),
            "specimen_id": state.get("specimen_id", ""),
            "weight_g": state.get("formal_weight_g", ""),
            "annotation_status": "formal_v1_user_confirmed_session",
            "preannotation_mode": "recommended_formal_workflow",
            "measurement_axis_mode": "auto_qc_gated",
            "coordinate_space": "warped_image",
            "calibration_state": {
                key: value
                for key, value in (state.get("formal_calibration_state") or {}).items()
                if key not in {"warped_image", "raw_marker_overlay"}
            },
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "corrected_keypoints": export_keypoints,
            "formal_measurement_points": export_formal_points,
            "derived_points": state.get("formal_derived_points", {}),
            "measurement_overrides": state.get("measurement_overrides", {}),
            "measurements": export_measurements,
            "note": "由正式前台页面导出的会话文件；未自动写入项目 corrected_keypoints 目录。",
        }

        work_col, result_col = st.columns([0.65, 0.35])
        with work_col:
            status_label = "已保存" if state.get("saved_at") else ("已修改未保存" if state.get("unsaved_changes") else "可拖拽修正")
            status_kind = "success" if state.get("saved_at") else ("warning" if state.get("unsaved_changes") else "info")
            st.markdown(
                f"""
                <div class="sm-workspace-title">
                  <div><strong>测量预览</strong><div class="sm-help">拖动点位可修正测量位置，点击应用修改后更新结果。</div></div>
                  {status_badge(status_label, status_kind)}
                </div>
                """,
                unsafe_allow_html=True,
            )
            editor_display, editor_scale, editor_points, editor_lines, editor_polylines, editor_stars = formal_editor_payload(
                state["working_image"],
                state["formal_measurement_points"],
                state["formal_measurements"],
                result["metadata"],
                max_width=1050,
            )
            preview = draw_lightweight_editor_preview(
                editor_display,
                editor_points,
                editor_lines,
                editor_polylines,
                editor_stars,
            )
            editor_opts = st.columns(2)
            with editor_opts[0]:
                formal_show_labels = st.checkbox("显示点位标签", value=False, key="formal_show_point_labels")
            with editor_opts[1]:
                formal_enable_magnifier = st.checkbox("启用局部放大镜", value=True, key="formal_enable_magnifier")
            editor_value = measurement_editor(
                editor_display,
                points=editor_points,
                lines=editor_lines,
                polylines=editor_polylines,
                stars=editor_stars,
                mm_per_pixel=float(state["mm_per_pixel"]),
                show_labels=formal_show_labels,
                enable_magnifier=formal_enable_magnifier,
                image_coord_width=float(state["working_image"].shape[1]),
                image_coord_height=float(state["working_image"].shape[0]),
                display_to_image_scale=float(editor_scale),
                debug=False,
                key=f"formal_editor_{state.get('image_name', 'image')}",
            )
            st.caption("拖动过程只在前端工作区内更新；点击组件内“应用修改”后，右侧结果会同步刷新。")

        event_ts = None if not isinstance(editor_value, dict) else editor_value.get("timestamp")
        if event_ts and event_ts != state.get("last_formal_editor_event_ts"):
            updated_points, override = update_formal_points_from_editor(
                state["formal_measurement_points"],
                editor_value,
                display_to_original_scale=editor_scale,
            )
            points_ok, point_errors = check_formal_coordinate_consistency(
                state["formal_calibration_state"],
                formal_points=updated_points,
            )
            if not points_ok:
                st.error("应用修改失败：点位坐标不在校正图像范围内。")
                for err in point_errors:
                    st.caption(f"- {err}")
            else:
                state["formal_measurement_points"] = updated_points
                state["formal_measurements"], state["formal_derived_points"] = recalculate_formal_measurements(
                    updated_points,
                    result["metadata"],
                    mm_per_pixel=float(state["mm_per_pixel"]),
                )
                if override:
                    overrides = dict(state.get("measurement_overrides", {}))
                    changed_points = override.get("changed_points", []) or list(updated_points)
                    for point_name in changed_points:
                        overrides[str(point_name)] = {
                            "xy": updated_points.get(str(point_name), []),
                            "modified_by_user": True,
                            "modified_at": override["modified_at"],
                        }
                    state["measurement_overrides"] = overrides
                    state["applied_changes"] = True
                    state["unsaved_changes"] = True
                    state.pop("saved_at", None)
            state["last_formal_editor_event_ts"] = event_ts
            st.rerun()

        with result_col:
            st.markdown("#### 测量结果")
            st.markdown(
                f"""
                <div class="sm-help">样本编号：<strong>{state.get("specimen_id") or "—"}</strong></div>
                <div class="sm-help">样品重量：<strong>{state.get("formal_weight_g") if state.get("formal_weight_g") not in ("", None) else "—"}</strong> g</div>
                """,
                unsafe_allow_html=True,
            )
            render_result_cards(current_row)
            st.markdown(status_badge(current_row.get("measurement_status", "需要复核"), "success" if state.get("saved_at") else "warning"), unsafe_allow_html=True)

            st.markdown("#### 操作")
            if st.button("保存人工确认结果", type="primary", use_container_width=True):
                save_formal_outputs(
                    export_df=export_df,
                    export_row=current_row,
                    corrected_payload=corrected_payload,
                    preview=preview,
                    export_formal_points=export_formal_points,
                    export_measurements=export_measurements,
                )
                st.success("已保存到正式前台输出目录，不会覆盖历史标注或批量分析结果。")
                st.rerun()

            if st.button("重置为自动测量结果", use_container_width=True):
                state["formal_measurement_points"] = dict(result.get("formal_measurement_points", {}))
                state["formal_measurements"], state["formal_derived_points"] = recalculate_formal_measurements(
                    state["formal_measurement_points"],
                    result["metadata"],
                    mm_per_pixel=float(state["mm_per_pixel"]),
                )
                state["measurement_overrides"] = {}
                state["unsaved_changes"] = False
                state.pop("saved_at", None)
                st.rerun()

            st.download_button(
                "导出当前结果 CSV",
                dataframe_to_csv_bytes(export_df),
                file_name="single_fish_measurement.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "导出当前结果 Excel",
                dataframe_to_excel_bytes(export_df),
                file_name="single_fish_measurement.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
            st.download_button(
                "导出标注预览图",
                image_bytes_png(preview),
                file_name="single_fish_warped_preview.png",
                mime="image/png",
                use_container_width=True,
            )
            st.download_button(
                "导出标注 JSON",
                json.dumps(corrected_payload, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="formal_corrected_keypoints_session_export.json",
                mime="application/json",
                use_container_width=True,
            )

            with st.expander("查看完整测量字段", expanded=False):
                st.dataframe(export_df, use_container_width=True, hide_index=True)
            with st.expander("查看已应用的修正点位", expanded=False):
                overrides = state.get("measurement_overrides", {})
                if overrides:
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "point_name": key,
                                    "x": value.get("xy", ["", ""])[0] if len(value.get("xy", [])) >= 2 else "",
                                    "y": value.get("xy", ["", ""])[1] if len(value.get("xy", [])) >= 2 else "",
                                    "modified_at": value.get("modified_at", ""),
                                }
                                for key, value in overrides.items()
                            ]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.write("尚未应用人工修正。")
