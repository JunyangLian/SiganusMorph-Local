from __future__ import annotations

from datetime import datetime
from io import BytesIO
import json
from pathlib import Path

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
from siganusmorph.ui_styles import apply_formal_theme, metric_grid, page_header, status_badge, step_header
from siganusmorph.remote_jobs import (
    ACTIVE_STATUSES,
    get_remote_jobs,
    load_remote_result,
    remote_compute_enabled,
    submit_remote_job,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMAL_BATCH_OUTPUT_ROOT = PROJECT_ROOT / "results" / "formal_v1_user_outputs" / "batch_review"
REMOTE_COMPUTE_ENABLED = remote_compute_enabled()


@st.cache_data(show_spinner=False, max_entries=24)
def lightweight_batch_preview(image_bytes: bytes, max_side: int = 640) -> Image.Image | None:
    """Create a small upload preview without retaining a full decoded image."""
    try:
        with Image.open(BytesIO(image_bytes)) as source:
            source.draft("RGB", (max_side, max_side))
            preview = ImageOps.exif_transpose(source).convert("RGB")
            preview.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            return preview.copy()
    except Exception:
        return None


def draw_batch_review_preview(
    display: Image.Image,
    points: list[dict[str, object]],
    lines: list[dict[str, object]],
    polylines: list[dict[str, object]],
    stars: list[dict[str, object]],
) -> Image.Image:
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
            draw.line(xy, fill=str(polyline.get("color", "#00aee8")), width=max(1, int(polyline.get("width", 2))))
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


def save_batch_review(review: dict[str, object], row: dict[str, object], preview: Image.Image) -> list[Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(safe_name(str(review.get("image_name") or "batch_image"))).stem
    output_dir = FORMAL_BATCH_OUTPUT_ROOT / f"{stem}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = output_dir / "measurement.csv"
    xlsx_path = output_dir / "measurement.xlsx"
    json_path = output_dir / "formal_review.json"
    preview_path = output_dir / "measurement_preview.png"
    frame = pd.DataFrame([row])
    csv_path.write_bytes(dataframe_to_csv_bytes(frame))
    xlsx_path.write_bytes(dataframe_to_excel_bytes(frame))
    preview_path.write_bytes(image_bytes_png(preview))
    result = review.get("result", {}) if isinstance(review.get("result"), dict) else {}
    payload = {
        "image_name": review.get("image_name", ""),
        "job_id": review.get("job_id", ""),
        "annotation_status": "formal_v1_batch_user_confirmed_session",
        "coordinate_space": "warped_image",
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "formal_measurement_points": review.get("formal_measurement_points", {}),
        "corrected_keypoints": formal_points_to_full_keypoints(
            review.get("formal_measurement_points", {}),
            result.get("full_keypoints", {}),
        ),
        "measurement_overrides": review.get("measurement_overrides", {}),
        "measurements": review.get("formal_measurements", {}),
        "note": "批量页人工复核会话输出；未写入项目历史 corrected_keypoints。",
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return [csv_path, xlsx_path, json_path, preview_path]

apply_formal_theme()
page_header(
    "批量测量",
    "一次导入多张标准化照片，批量运行校准、自动测量和结果汇总，适合正式数据整理前的快速处理。",
    eyebrow="Batch Measurement",
)

if "formal_batch_rows" not in st.session_state:
    st.session_state.formal_batch_rows = []
if "formal_batch_jobs" not in st.session_state:
    st.session_state.formal_batch_jobs = []
if "formal_batch_review" not in st.session_state:
    st.session_state.formal_batch_review = {}


def remote_row_from_payload(job: dict[str, object], payload: dict[str, object]) -> dict[str, object]:
    image_name = str(job.get("image_name") or "")
    base = {
        "image_name": image_name,
        "specimen_id": str(job.get("specimen_id") or ""),
        "weight_g": job.get("weight_g") if job.get("weight_g") is not None else "",
        "status": "needs_review",
        "calibration_status": "",
        "measurement_status": "",
        "notes": "",
    }
    if payload.get("status") != "success":
        calibration = payload.get("calibration_state", {})
        message = calibration.get("calibration_message", "") if isinstance(calibration, dict) else ""
        base.update(
            status="calibration_failed",
            calibration_status="calibration_failed",
            measurement_status="needs_review",
            notes=str(message or "校准失败，未运行自动测量"),
        )
        return base
    measurement_result = payload.get("measurement_result", {})
    formal_measurements = (
        measurement_result.get("formal_measurements", {})
        if isinstance(measurement_result, dict)
        else {}
    )
    base.update(
        simplified_measurement_row(
            formal_measurements,
            image_name=image_name,
            specimen_id=str(job.get("specimen_id") or ""),
            weight_g=job.get("weight_g") if job.get("weight_g") is not None else "",
            notes="批量远程自动测量，建议人工复核",
        )
    )
    base.update(status="success", calibration_status="success", measurement_status="exported")
    return base


@st.fragment(run_every=2.0)
def render_remote_batch_monitor() -> None:
    if not REMOTE_COMPUTE_ENABLED:
        return
    batch_jobs = list(st.session_state.get("formal_batch_jobs", []))
    if not batch_jobs:
        st.caption("批量任务将由远程计算节点逐张处理，云端页面保持轻量响应。")
        return
    rows = list(st.session_state.get("formal_batch_rows", []))
    rows_by_id = {str(row.get("job_id")): dict(row) for row in rows}
    jobs_by_id = get_remote_jobs(
        [str(item.get("job_id") or "") for item in batch_jobs],
        project_root=PROJECT_ROOT,
    )
    terminal = 0
    for item in batch_jobs:
        job_id = str(item.get("job_id") or "")
        job = jobs_by_id.get(job_id)
        row = rows_by_id.get(job_id, dict(item))
        if not job:
            row.update(status="needs_review", measurement_status="failed", notes="找不到远程任务")
            terminal += 1
        else:
            status = str(job.get("status") or "")
            if not (status == "completed" and row.get("result_loaded")):
                row["status"] = status
                row["notes"] = str(job.get("progress") or "")
            if status == "completed" and not row.get("result_loaded"):
                try:
                    payload = load_remote_result(job_id, project_root=PROJECT_ROOT, include_warped=False)
                    completed_row = remote_row_from_payload(row, payload)
                    completed_row.update(job_id=job_id, result_loaded=True)
                    row = completed_row
                except Exception as exc:  # noqa: BLE001
                    row.update(status="needs_review", measurement_status="failed", notes=f"读取结果失败：{exc}")
                terminal += 1
            elif status == "completed" and row.get("result_loaded"):
                terminal += 1
            elif status == "failed":
                row.update(status="needs_review", measurement_status="failed", notes=str(job.get("error_message") or "远程测量失败"))
                terminal += 1
            elif status not in ACTIVE_STATUSES:
                terminal += 1
        rows_by_id[job_id] = row
    ordered_rows = [rows_by_id[str(item.get("job_id") or "")] for item in batch_jobs]
    st.session_state.formal_batch_rows = ordered_rows
    completed = sum(1 for row in ordered_rows if row.get("status") in {"success", "needs_review", "calibration_failed"})
    st.progress(completed / max(1, len(ordered_rows)))
    st.dataframe(
        pd.DataFrame(ordered_rows)[[column for column in ("image_name", "status", "notes") if column in pd.DataFrame(ordered_rows).columns]],
        use_container_width=True,
        hide_index=True,
    )
    if terminal == len(batch_jobs) and not st.session_state.get("formal_batch_terminal_rerun"):
        st.session_state.formal_batch_terminal_rerun = True
        st.rerun()

with st.container(border=True):
    step_header("1", "上传与批量处理", "上传多张照片，系统逐张校准并生成测量结果。")
    upload_col, action_col = st.columns([1.35, 1])
    with upload_col:
        uploads = st.file_uploader(
            "上传多张鱼体照片",
            type=["png", "jpg", "jpeg", "heic", "tif", "tiff"],
            accept_multiple_files=True,
        )
        if uploads:
            st.caption(f"已选择 {len(uploads)} 张图片。")
            preview_cols = st.columns(min(4, max(1, len(uploads))))
            for idx, uploaded in enumerate(uploads[:4]):
                with preview_cols[idx % len(preview_cols)]:
                    preview = lightweight_batch_preview(uploaded.getvalue())
                    if preview is not None:
                        st.image(preview, caption=safe_name(uploaded.name), use_container_width=True)
                    else:
                        st.caption(f"{safe_name(uploaded.name)}：已选择，无缩略图")
        else:
            st.info("请上传图片后开始批量测量。")

    with action_col:
        st.markdown("#### 处理状态")
        current_count = len(st.session_state.formal_batch_rows)
        st.markdown(status_badge(f"已生成 {current_count} 条结果", "success" if current_count else "neutral"), unsafe_allow_html=True)
        run_batch = st.button("开始批量测量", type="primary", disabled=not uploads, use_container_width=True)

    if uploads and run_batch and REMOTE_COMPUTE_ENABLED:
        jobs: list[dict[str, object]] = []
        rows: list[dict[str, object]] = []
        for uploaded in uploads:
            image_name = safe_name(uploaded.name)
            try:
                job = submit_remote_job(
                    uploaded.getvalue(),
                    image_name=image_name,
                    project_root=PROJECT_ROOT,
                )
                item = {
                    "job_id": str(job.get("job_id") or ""),
                    "image_name": image_name,
                    "specimen_id": "",
                    "weight_g": "",
                }
                jobs.append(item)
                rows.append(
                    {
                        **item,
                        "status": str(job.get("status") or "queued"),
                        "calibration_status": "",
                        "measurement_status": "",
                        "notes": str(job.get("progress") or "等待计算节点领取"),
                        "result_loaded": False,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "job_id": "",
                        "image_name": image_name,
                        "specimen_id": "",
                        "weight_g": "",
                        "status": "needs_review",
                        "calibration_status": "",
                        "measurement_status": "failed",
                        "notes": f"提交失败：{exc}",
                        "result_loaded": False,
                    }
                )
        st.session_state.formal_batch_jobs = jobs
        st.session_state.formal_batch_rows = rows
        st.session_state.formal_batch_terminal_rerun = False
        st.success(f"已提交 {len(jobs)} 张图片，计算节点将逐张处理。")

    elif uploads and run_batch:
        rows: list[dict[str, object]] = []
        progress = st.progress(0)
        for index, uploaded in enumerate(uploads, start=1):
            image_name = safe_name(uploaded.name)
            row = {
                "image_name": image_name,
                "specimen_id": "",
                "weight_g": "",
                "status": "needs_review",
                "calibration_status": "",
                "measurement_status": "",
                "notes": "",
            }
            try:
                image = image_from_upload(uploaded)
                calibration = calibrate_with_aruco(image)
                row["calibration_status"] = "success" if calibration.get("success") else "calibration_failed"
                if not calibration.get("success"):
                    row["status"] = "calibration_failed"
                    row["measurement_status"] = "needs_review"
                    row["notes"] = "校准失败，未运行自动测量"
                    rows.append(row)
                    progress.progress(index / len(uploads))
                    continue
                working = calibration["warped_image"]
                mm_per_pixel = float(calibration["mm_per_pixel"])
                result = run_recommended_measurement(working, image_name, PROJECT_ROOT, mm_per_pixel=mm_per_pixel)
                row.update(
                    simplified_measurement_row(
                        result["measurements"],
                        image_name=image_name,
                        notes="批量自动测量，建议人工复核",
                    )
                )
                row["status"] = "success"
                row["measurement_status"] = "exported"
            except Exception as exc:  # noqa: BLE001 - batch row status.
                row["status"] = "needs_review"
                row["measurement_status"] = "failed"
                row["notes"] = f"处理失败：{exc}"
            rows.append(row)
            progress.progress(index / len(uploads))
        st.session_state.formal_batch_rows = rows
        st.success("批量处理完成。请检查需要复核的图片。")

    render_remote_batch_monitor()

rows_df = pd.DataFrame(st.session_state.formal_batch_rows)
if rows_df.empty:
    rows_df = pd.DataFrame(
        columns=[
            "image_name",
            "specimen_id",
            "weight_g",
            "status",
            "calibration_status",
            "measurement_status",
            "notes",
        ]
    )

with st.container(border=True):
    step_header("2", "结果摘要与导出", "查看处理状态，导出批量测量表。")
    total = len(rows_df)
    success_count = int((rows_df.get("status") == "success").sum()) if total else 0
    failed_count = int((rows_df.get("status") == "calibration_failed").sum()) if total else 0
    review_count = int((rows_df.get("status") == "needs_review").sum()) if total else 0
    metric_grid(
        [
            ("总图片数", total, ""),
            ("测量成功", success_count, ""),
            ("校准失败", failed_count, ""),
            ("需要复核", review_count, ""),
        ]
    )

    if not rows_df.empty and total > 0:
        display_df = rows_df.copy()
        display_df["status_badge"] = display_df["status"].map(
            {
                "success": "success",
                "calibration_failed": "calibration_failed",
                "needs_review": "needs_review",
                "exported": "exported",
            }
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        dl_cols = st.columns(2)
        with dl_cols[0]:
            st.download_button(
                "导出批量 CSV",
                dataframe_to_csv_bytes(rows_df),
                file_name="batch_measurements_formal.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl_cols[1]:
            st.download_button(
                "导出批量 Excel",
                dataframe_to_excel_bytes(rows_df),
                file_name="batch_measurements_formal.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
    else:
        st.info("尚未生成批量测量结果。")

with st.container(border=True):
    step_header("3", "逐张复核、修正与导出", "按需加载一张校正图进行拖拽修正，避免同时解码多张大图。")
    review_rows = [
        dict(row)
        for row in st.session_state.get("formal_batch_rows", [])
        if row.get("job_id") and row.get("result_loaded") and row.get("status") == "success"
    ]
    if not review_rows:
        st.info("批量远程测量完成后，可在此选择图片进行人工复核。")
    else:
        label_to_job = {
            f"{row.get('image_name', '')}  [{row.get('measurement_status', '')}]": str(row.get("job_id"))
            for row in review_rows
        }
        selected_label = st.selectbox("选择待复核图片", list(label_to_job))
        selected_job_id = label_to_job[selected_label]
        if st.button("加载所选图片进行复核", use_container_width=True):
            try:
                payload = load_remote_result(selected_job_id, project_root=PROJECT_ROOT, include_warped=True)
                if payload.get("status") != "success" or payload.get("warped_image") is None:
                    raise ValueError("该任务没有可用的校正图像或测量结果")
                result_payload = payload.get("measurement_result", {})
                if not isinstance(result_payload, dict):
                    raise ValueError("该任务的测量结果格式无效")
                calibration = payload.get("calibration_state", {})
                calibration = dict(calibration) if isinstance(calibration, dict) else {}
                calibration["warped_image"] = payload["warped_image"]
                st.session_state.formal_batch_review = {
                    "job_id": selected_job_id,
                    "image_name": payload.get("image_name", ""),
                    "working_image": payload["warped_image"],
                    "calibration_state": calibration,
                    "result": result_payload,
                    "formal_measurement_points": dict(result_payload.get("formal_measurement_points", {})),
                    "formal_measurements": dict(result_payload.get("formal_measurements", {})),
                    "formal_derived_points": dict(result_payload.get("formal_derived_points", {})),
                    "measurement_overrides": {},
                    "last_event_ts": "",
                    "saved": False,
                }
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"加载复核图片失败：{exc}")

        review = st.session_state.get("formal_batch_review", {})
        if review and review.get("job_id") == selected_job_id:
            result_payload = review.get("result", {})
            metadata = result_payload.get("metadata", {}) if isinstance(result_payload, dict) else {}
            calibration = review.get("calibration_state", {})
            mm_per_pixel = float(calibration.get("mm_per_pixel"))
            display, scale, points, lines, polylines, stars = formal_editor_payload(
                review["working_image"],
                review["formal_measurement_points"],
                review["formal_measurements"],
                metadata,
                max_width=1050,
            )
            preview = draw_batch_review_preview(display, points, lines, polylines, stars)
            editor_col, metrics_col = st.columns([0.66, 0.34])
            with editor_col:
                st.markdown("#### 批量复核工作区")
                editor_value = measurement_editor(
                    display,
                    points=points,
                    lines=lines,
                    polylines=polylines,
                    stars=stars,
                    mm_per_pixel=mm_per_pixel,
                    show_labels=False,
                    enable_magnifier=True,
                    image_coord_width=float(review["working_image"].shape[1]),
                    image_coord_height=float(review["working_image"].shape[0]),
                    display_to_image_scale=float(scale),
                    debug=False,
                    key=f"formal_batch_editor_{selected_job_id}",
                )
                st.caption("拖动点位后，在组件内点击“应用修改”即可更新当前批量结果。")

            event_ts = editor_value.get("timestamp") if isinstance(editor_value, dict) else None
            if event_ts and event_ts != review.get("last_event_ts"):
                updated_points, override = update_formal_points_from_editor(
                    review["formal_measurement_points"],
                    editor_value,
                    display_to_original_scale=scale,
                )
                points_ok, point_errors = check_formal_coordinate_consistency(
                    calibration,
                    formal_points=updated_points,
                )
                if not points_ok:
                    st.error("应用修改失败：" + "；".join(point_errors))
                else:
                    measurements, derived = recalculate_formal_measurements(
                        updated_points,
                        metadata,
                        mm_per_pixel=mm_per_pixel,
                    )
                    review["formal_measurement_points"] = updated_points
                    review["formal_measurements"] = measurements
                    review["formal_derived_points"] = derived
                    review["last_event_ts"] = event_ts
                    review["saved"] = False
                    if override:
                        review["measurement_overrides"] = {
                            name: {
                                "xy": updated_points.get(name, []),
                                "modified_by_user": True,
                                "modified_at": override.get("modified_at", ""),
                            }
                            for name in override.get("changed_points", [])
                        }
                    updated_row = simplified_measurement_row(
                        measurements,
                        image_name=str(review.get("image_name") or ""),
                        notes="批量页已应用人工修正，待保存",
                    )
                    updated_row.update(
                        job_id=selected_job_id,
                        result_loaded=True,
                        status="success",
                        calibration_status="success",
                        measurement_status="reviewed",
                    )
                    st.session_state.formal_batch_rows = [
                        updated_row if str(row.get("job_id")) == selected_job_id else row
                        for row in st.session_state.formal_batch_rows
                    ]
                    st.session_state.formal_batch_review = review
                    st.rerun()

            with metrics_col:
                current_row = simplified_measurement_row(
                    review["formal_measurements"],
                    image_name=str(review.get("image_name") or ""),
                    notes="批量页人工复核结果",
                )
                st.markdown("#### 当前测量")
                metric_grid(
                    [
                        ("压拢尾鳍全长 TL", current_row.get("TL_compressed_virtual_mm", ""), "mm"),
                        ("标准长 SL", current_row.get("SL_mm", ""), "mm"),
                        ("叉长 FL", current_row.get("FL_mm", ""), "mm"),
                        ("体高", current_row.get("body_depth_mm", ""), "mm"),
                        ("尾柄高", current_row.get("caudal_peduncle_depth_mm", ""), "mm"),
                    ]
                )
                if st.button("保存该图人工复核结果", type="primary", use_container_width=True):
                    paths = save_batch_review(review, current_row, preview)
                    review["saved"] = True
                    st.session_state.formal_batch_review = review
                    st.success(f"已保存 {len(paths)} 个正式用户输出文件。")
                st.download_button(
                    "导出该图 CSV",
                    dataframe_to_csv_bytes(pd.DataFrame([current_row])),
                    file_name=f"{Path(str(review.get('image_name') or 'image')).stem}_measurement.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                st.download_button(
                    "导出该图标注预览",
                    image_bytes_png(preview),
                    file_name=f"{Path(str(review.get('image_name') or 'image')).stem}_preview.png",
                    mime="image/png",
                    use_container_width=True,
                )

st.caption("批量复核结果只写入 formal_v1_user_outputs，不会覆盖历史 corrected_keypoints 或批量分析结果。")
