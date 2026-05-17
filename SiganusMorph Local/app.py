from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from siganusmorph.annotation import save_keypoints_json
from siganusmorph.aruco_utils import (
    detect_aruco_markers,
    draw_detected_aruco,
    select_board_config_for_markers,
    warp_board_by_aruco,
)
from siganusmorph.axis_utils import AXIS_MODE_AUTO, AXIS_MODE_POLYLINE, AXIS_MODE_SPLINE, axis_mode_label
from siganusmorph.calibration import calculate_scale_from_two_points, estimate_scale_from_ruler_ticks
from siganusmorph.components import image_clicker
from siganusmorph.config import KEYPOINT_DEFS, KeypointDefinition
from siganusmorph.image_utils import image_from_bytes, list_image_files, load_image_file, resize_for_display
from siganusmorph.io_utils import (
    ensure_output_dirs,
    resolve_output_dir,
    sanitize_filename_stem,
    save_annotation_image,
    upsert_result_row,
)
from siganusmorph.measurements import build_result_row, calculate_measurements
from siganusmorph.visualization import draw_keypoints_and_measurements, keypoint_dict_from_list

PROJECT_ROOT = Path(__file__).resolve().parent
TEMPLATE_DIR = PROJECT_ROOT / "assets" / "templates"
TOTAL_KEYPOINTS = len(KEYPOINT_DEFS)

AXIS_MODE_OPTIONS = (
    "自动推荐（可手动覆盖）",
    "折线中轴线（polyline axis）",
    "平滑曲线中轴线（spline axis）",
)
AXIS_MODE_OPTION_VALUES = {
    "自动推荐（可手动覆盖）": AXIS_MODE_AUTO,
    "折线中轴线（polyline axis）": AXIS_MODE_POLYLINE,
    "平滑曲线中轴线（spline axis）": AXIS_MODE_SPLINE,
}


def reset_points(reset_image: bool = False) -> None:
    st.session_state.scale_points = []
    st.session_state.keypoint_points = []
    st.session_state.mm_per_pixel = 0.0
    st.session_state.scale_method_label = ""
    st.session_state.scale_info = {}
    st.session_state.last_saved_signature = ""
    st.session_state.save_message = ""
    if reset_image and "original_image" in st.session_state:
        st.session_state.working_image = st.session_state.original_image.copy()


def ensure_state() -> None:
    defaults: dict[str, Any] = {
        "scale_points": [],
        "keypoint_points": [],
        "mm_per_pixel": 0.0,
        "scale_method_label": "",
        "scale_info": {},
        "image_token": "",
        "last_saved_signature": "",
        "save_message": "",
        "folder_select": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def source_image_selector() -> tuple[str, str, Any] | None:
    source_mode = st.sidebar.radio("图片来源", ("上传单张图片", "本地文件夹"), horizontal=True)
    if source_mode == "上传单张图片":
        uploaded = st.sidebar.file_uploader(
            "选择图片",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
        )
        if uploaded is None:
            return None
        data = uploaded.getvalue()
        digest = hashlib.md5(data).hexdigest()[:12]
        return uploaded.name, f"upload::{uploaded.name}::{digest}", data

    folder_text = st.sidebar.text_input("本地图片文件夹", value=str(PROJECT_ROOT / "data" / "sample_images"))
    image_paths = list_image_files(folder_text)
    if not image_paths:
        st.sidebar.info("这个文件夹里还没有可用图片。")
        return None

    options = [str(path) for path in image_paths]
    selected = st.sidebar.selectbox("选择图片", options, key="folder_select")
    selected_path = Path(selected)

    col_prev, col_next = st.sidebar.columns(2)
    current_index = options.index(selected)
    if col_prev.button("上一张", disabled=current_index == 0):
        st.session_state.folder_select = options[current_index - 1]
        reset_points(reset_image=True)
        st.rerun()
    if col_next.button("下一张", disabled=current_index >= len(options) - 1):
        st.session_state.folder_select = options[current_index + 1]
        reset_points(reset_image=True)
        st.rerun()

    return selected_path.name, f"file::{selected_path.resolve()}::{selected_path.stat().st_mtime_ns}", selected_path


def load_current_image(image_name: str, image_token: str, image_source: Any) -> None:
    if st.session_state.image_token == image_token:
        return

    if isinstance(image_source, bytes):
        image = image_from_bytes(image_source)
    else:
        image = load_image_file(image_source)

    st.session_state.image_token = image_token
    st.session_state.image_name = image_name
    st.session_state.original_image = image
    st.session_state.working_image = image.copy()
    reset_points(reset_image=False)


def current_specimen_id() -> str:
    id_mode = st.sidebar.radio("样本编号", ("1-40 编号区", "手动输入"), horizontal=True)
    if id_mode == "1-40 编号区":
        return str(st.sidebar.selectbox("编号", list(range(1, 41))))
    return st.sidebar.text_input("样本编号", value="").strip()


def current_source_type() -> str:
    label = st.sidebar.radio("样本来源", ("real", "synthetic", "unknown"), horizontal=True)
    return label


def needs_review_from_measurements(measurements: dict[str, Any] | None) -> bool:
    if not measurements:
        return False
    curvature = measurements.get("curvature_index_polyline")
    return curvature is not None and float(curvature) > 1.05


def template_path(filename: str) -> Path:
    return TEMPLATE_DIR / filename


def show_template_image(filename: str, caption: str) -> None:
    path = template_path(filename)
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.caption(f"未找到模板图：{filename}")


def render_annotation_helper(
    definition: KeypointDefinition | None,
    helper_mode: str,
    click_mode: str,
    completed: bool,
) -> None:
    st.subheader("标注辅助区")
    if completed:
        st.success(f"{TOTAL_KEYPOINTS} 个点已完成")
        show_template_image("overview_template.png", f"{TOTAL_KEYPOINTS} 点总览")
        return

    if click_mode != "keypoint" or definition is None:
        st.caption("完成比例校准后，这里会显示当前关键点示范。")
        if helper_mode == "新手模式":
            show_template_image("overview_template.png", f"{TOTAL_KEYPOINTS} 点总览")
        return

    st.markdown(f"**当前点：{definition.code} {definition.label_cn}**")
    st.caption(f"`{definition.name}`")

    if helper_mode == "熟练模式":
        st.write(definition.description)
        return

    show_template_image("overview_template.png", "A. 总览模板鱼")
    show_template_image(f"{definition.code}_template.png", f"B. 当前点高亮：{definition.code}")
    show_template_image(f"{definition.code}_zoom.png", f"局部放大：{definition.code}")
    st.markdown(f"**定义：**{definition.description}")
    st.warning(f"常见错误：{definition.common_errors}")


def calibration_panel(method: str) -> tuple[bool, str]:
    """Render calibration controls and return whether keypoint clicks can start."""
    if method == "手动输入 mm_per_pixel":
        value = st.sidebar.number_input("mm_per_pixel", min_value=0.0, value=st.session_state.mm_per_pixel, step=0.001, format="%.8f")
        st.session_state.mm_per_pixel = float(value)
        st.session_state.scale_method_label = "manual_mm_per_pixel" if value > 0 else ""
        return value > 0, "请输入大于 0 的 mm_per_pixel。"

    if method == "手动点击尺子两点":
        real_distance = st.sidebar.number_input("两点真实距离 mm", min_value=0.01, value=100.0, step=10.0)
        if len(st.session_state.scale_points) == 2:
            st.session_state.mm_per_pixel = calculate_scale_from_two_points(
                st.session_state.scale_points[0],
                st.session_state.scale_points[1],
                float(real_distance),
            )
            st.session_state.scale_method_label = "manual_two_points"
            return True, f"比例已计算：{st.session_state.mm_per_pixel:.8f} mm/pixel"
        st.session_state.mm_per_pixel = 0.0
        st.session_state.scale_method_label = ""
        return False, f"请在尺子上点击两个点：已点击 {len(st.session_state.scale_points)}/2。"

    detection = detect_aruco_markers(st.session_state.original_image)
    markers = detection.get("markers", {})
    st.sidebar.write(f"检测到 ArUco ID：{detection.get('ids', [])}")
    board_config = select_board_config_for_markers(markers)
    st.sidebar.write(f"板型：{board_config.get('name', 'unknown')}")
    required_ids = {int(marker_id) for marker_id in board_config["aruco_corner_ids"].values()}
    if required_ids.issubset(set(markers)):
        if st.sidebar.button("应用 ArUco 校正"):
            warped, _transform, info = warp_board_by_aruco(st.session_state.original_image, markers, board_config)
            ruler_scale = estimate_scale_from_ruler_ticks(warped)
            st.session_state.working_image = warped
            if "location_marker_squares_mm" in board_config:
                st.session_state.mm_per_pixel = float(info["mm_per_pixel"])
                st.session_state.scale_method_label = "aruco_board_coordinate_auto"
                st.session_state.scale_points = []
                st.session_state.scale_info = {
                    "success": True,
                    "board_coordinate_scale": True,
                    "ruler_qc": ruler_scale,
                    **info,
                }
            elif ruler_scale.get("success"):
                st.session_state.mm_per_pixel = float(ruler_scale["mm_per_pixel"])
                st.session_state.scale_method_label = "aruco_ruler_auto"
                st.session_state.scale_points = list(ruler_scale["endpoints"])
                st.session_state.scale_info = {**info, **ruler_scale}
            else:
                st.session_state.mm_per_pixel = float(info["mm_per_pixel"])
                st.session_state.scale_method_label = "aruco_board_width_fallback"
                st.session_state.scale_points = []
                st.session_state.scale_info = {
                    "success": False,
                    "fallback": "board_width",
                    "ruler_message": ruler_scale.get("message", ""),
                    **info,
                }
            st.session_state.keypoint_points = []
            st.session_state.last_saved_signature = ""
            st.rerun()
        if st.session_state.scale_method_label == "aruco_board_coordinate_auto" and st.session_state.mm_per_pixel > 0:
            scale_info = st.session_state.scale_info
            reproj = scale_info.get("reprojection_error_px_mean")
            reproj_text = f"；重投影误差 {reproj:.2f} px" if reproj is not None else ""
            ruler_qc = scale_info.get("ruler_qc", {})
            if ruler_qc.get("success"):
                ruler_text = f"；尺子 QC 跨度 {ruler_qc.get('pixel_span', 0):.1f} px"
            else:
                ruler_text = f"；尺子 QC 未通过：{ruler_qc.get('message', '未检测到尺子')}"
            return True, (
                f"V2 板面坐标比例：{st.session_state.mm_per_pixel:.8f} mm/pixel"
                f"{reproj_text}{ruler_text}。"
            )
        if st.session_state.scale_method_label == "aruco_ruler_auto" and st.session_state.mm_per_pixel > 0:
            scale_info = st.session_state.scale_info
            reproj = scale_info.get("reprojection_error_px_mean")
            reproj_text = f"；重投影误差 {reproj:.2f} px" if reproj is not None else ""
            return True, (
                f"ArUco 已校正；尺子自动比例：{st.session_state.mm_per_pixel:.8f} mm/pixel；"
                f"0-35 cm 跨度 {scale_info.get('pixel_span', 0):.1f} px{reproj_text}。"
            )
        if st.session_state.scale_method_label == "aruco_board_width_fallback" and st.session_state.mm_per_pixel > 0:
            return True, (
                f"ArUco 已校正；尺子识别失败，暂用板宽比例："
                f"{st.session_state.mm_per_pixel:.8f} mm/pixel。建议改用手动尺子两点校准。"
            )
        if "location_marker_squares_mm" in board_config:
            return False, "已检测到 V2 定位 ArUco，点击按钮应用板面坐标校正。"
        return False, "已检测到四角 ArUco，点击按钮应用透视校正并自动识别 0-35 cm 尺子。"

    fallback = st.sidebar.number_input("ArUco 失败时手动 mm_per_pixel", min_value=0.0, value=st.session_state.mm_per_pixel, step=0.001, format="%.8f")
    if fallback > 0:
        st.session_state.mm_per_pixel = float(fallback)
        st.session_state.scale_method_label = "aruco_fallback_manual"
        return True, f"使用手动兜底比例：{fallback:.8f} mm/pixel"
    return False, "没有检测齐 ID 0、1、2、3；请手动输入比例或切换到手动校准。"


def add_clicked_point(click_mode: str, x: float, y: float) -> None:
    point = {"x": float(x), "y": float(y)}
    if click_mode == "scale":
        if len(st.session_state.scale_points) < 2:
            st.session_state.scale_points.append(point)
        return

    if click_mode == "keypoint" and len(st.session_state.keypoint_points) < TOTAL_KEYPOINTS:
        definition = KEYPOINT_DEFS[len(st.session_state.keypoint_points)]
        st.session_state.keypoint_points.append(
            {
                "name": definition.name,
                "code": definition.code,
                "label_cn": definition.label_cn,
                "description": definition.description,
                "x": float(x),
                "y": float(y),
            }
        )


def click_capture(image, scale_factor: float, click_mode: str) -> None:
    if click_mode == "idle":
        st.image(image, use_container_width=False)
        return

    value = image_clicker(
        image,
        key=f"coords-{st.session_state.image_token}-{click_mode}-{len(st.session_state.scale_points)}-{len(st.session_state.keypoint_points)}",
    )
    if value is not None:
        add_clicked_point(click_mode, value["x"] * scale_factor, value["y"] * scale_factor)
        st.rerun()


def save_current_outputs(
    output_dir: Path,
    image_name: str,
    specimen_id: str,
    source_type: str,
    notes: str,
    axis_mode_request: str,
    force: bool = False,
) -> str:
    keypoints = keypoint_dict_from_list(st.session_state.keypoint_points)
    measurements = calculate_measurements(keypoints, st.session_state.mm_per_pixel, axis_mode_request)
    needs_review = needs_review_from_measurements(measurements)
    axis_mode_actual = str(measurements.get("axis_mode_selected", ""))
    signature = repr(
        (
            image_name,
            specimen_id,
            source_type,
            st.session_state.scale_method_label,
            round(st.session_state.mm_per_pixel, 8),
            axis_mode_request,
            axis_mode_actual,
            keypoints,
            notes,
            str(output_dir),
        )
    )
    if not force and signature == st.session_state.last_saved_signature:
        return st.session_state.save_message

    paths = ensure_output_dirs(output_dir)
    row = build_result_row(
        image_name=image_name,
        specimen_id=specimen_id,
        source_type=source_type,
        scale_method=st.session_state.scale_method_label,
        mm_per_pixel=st.session_state.mm_per_pixel,
        keypoints=keypoints,
        measurements=measurements,
        needs_review=needs_review,
        notes=notes,
    )
    result_paths = upsert_result_row(row, output_dir)

    stem_parts = [Path(image_name).stem]
    if specimen_id:
        stem_parts.append(f"id_{specimen_id}")
    safe_stem = sanitize_filename_stem("_".join(stem_parts))
    annotated = draw_keypoints_and_measurements(
        st.session_state.working_image,
        keypoints,
        measurements,
        image_name=image_name,
        specimen_id=specimen_id,
        axis_mode_selected=axis_mode_actual,
    )
    annotation_path = save_annotation_image(annotated, paths["annotations"] / f"{safe_stem}_annotated.png")
    json_path = save_keypoints_json(
        image_name=image_name,
        keypoints=keypoints,
        output_path=paths["keypoints"] / f"{safe_stem}_keypoints.json",
        metadata={
            "specimen_id": specimen_id,
            "source_type": source_type,
            "scale_method": st.session_state.scale_method_label,
            "mm_per_pixel": st.session_state.mm_per_pixel,
            "scale_info": st.session_state.scale_info,
            "axis_mode_selected": axis_mode_actual,
            "axis_mode_request": axis_mode_request,
            "needs_review": needs_review,
            "notes": notes,
        },
        measurements=measurements,
    )

    message = (
        f"已保存：{annotation_path.name}、{json_path.name}、"
        f"{result_paths['csv'].name}、{result_paths['xlsx'].name}"
    )
    st.session_state.last_saved_signature = signature
    st.session_state.save_message = message
    return message


def main() -> None:
    st.set_page_config(page_title="SiganusMorph Local", layout="wide")
    ensure_state()

    st.title("SiganusMorph Local")
    st.caption("V0.1 半自动直线测量 + V0.2 ArUco/板面坐标校准 + 16 个手动点 + P7V 虚拟全长终点")

    selected = source_image_selector()
    output_text = st.sidebar.text_input("输出文件夹", value=str(PROJECT_ROOT / "results"))
    output_dir = resolve_output_dir(output_text, PROJECT_ROOT)
    calibration_method = st.sidebar.radio(
        "比例校准方式",
        ("手动点击尺子两点", "手动输入 mm_per_pixel", "尝试 ArUco 自动识别"),
    )
    specimen_id = current_specimen_id()
    source_type = current_source_type()
    axis_mode_option = st.sidebar.radio("中轴线模式选择", AXIS_MODE_OPTIONS)
    axis_mode_request = AXIS_MODE_OPTION_VALUES[axis_mode_option]
    helper_mode = st.sidebar.radio("标注辅助显示", ("新手模式", "熟练模式"), horizontal=True)
    notes = st.sidebar.text_area("备注", value="", height=90)
    auto_save = st.sidebar.checkbox("关键点完成后自动保存", value=True)
    max_display_width = st.sidebar.slider("图像显示宽度", min_value=600, max_value=1600, value=1100, step=100)

    if selected is None:
        st.info("请先上传一张图片，或在侧边栏选择本地图片文件夹。")
        return

    image_name, image_token, image_source = selected
    load_current_image(image_name, image_token, image_source)

    if calibration_method == "尝试 ArUco 自动识别":
        detection = detect_aruco_markers(st.session_state.original_image)
        with st.expander("ArUco 检测预览", expanded=False):
            preview = draw_detected_aruco(st.session_state.original_image, detection.get("markers", {}))
            preview_image, _ = resize_for_display(preview, max_width=900)
            st.image(preview_image)

    can_place_keypoints, calibration_status = calibration_panel(calibration_method)
    st.sidebar.info(calibration_status)

    col_actions = st.columns([1, 1, 1, 3])
    if col_actions[0].button("撤销上一个点"):
        if st.session_state.keypoint_points:
            st.session_state.keypoint_points.pop()
        elif st.session_state.scale_points:
            st.session_state.scale_points.pop()
        st.session_state.last_saved_signature = ""
        st.rerun()
    if col_actions[1].button("重置当前图片"):
        reset_points(reset_image=True)
        st.rerun()
    if col_actions[2].button("清除透视校正"):
        st.session_state.working_image = st.session_state.original_image.copy()
        reset_points(reset_image=False)
        st.rerun()

    keypoints = keypoint_dict_from_list(st.session_state.keypoint_points)
    measurements = None
    needs_review = False
    completed = len(st.session_state.keypoint_points) == TOTAL_KEYPOINTS and st.session_state.mm_per_pixel > 0
    if completed:
        measurements = calculate_measurements(keypoints, st.session_state.mm_per_pixel, axis_mode_request)
        needs_review = needs_review_from_measurements(measurements)

    current_definition: KeypointDefinition | None = None
    if calibration_method == "手动点击尺子两点" and len(st.session_state.scale_points) < 2:
        click_mode = "scale"
        prompt = f"当前步骤：点击尺子校准点 S{len(st.session_state.scale_points) + 1}"
        instruction = "说明：请点击尺子上两个已知真实距离的端点，用于计算 mm_per_pixel。"
    elif can_place_keypoints and len(st.session_state.keypoint_points) < TOTAL_KEYPOINTS:
        click_mode = "keypoint"
        definition = KEYPOINT_DEFS[len(st.session_state.keypoint_points)]
        current_definition = definition
        prompt = f"当前点：{definition.code} {definition.label_cn}"
        instruction = f"说明：{definition.description}"
    else:
        click_mode = "idle"
        prompt = "当前步骤：等待比例校准或结果保存" if not completed else "关键点已完成"
        instruction = f"说明：请先完成比例校准，再按提示依次标注 {TOTAL_KEYPOINTS} 个点。" if not completed else f"说明：已完成 {TOTAL_KEYPOINTS} 个点，可保存结果或进入下一张。"

    image_col, helper_col = st.columns([4, 1.35])
    with image_col:
        st.subheader(prompt)
        st.info(instruction)
        st.write(
            f"关键点进度：{len(st.session_state.keypoint_points)}/{TOTAL_KEYPOINTS}；"
            f"mm_per_pixel：{st.session_state.mm_per_pixel:.8f}" if st.session_state.mm_per_pixel > 0 else
            f"关键点进度：{len(st.session_state.keypoint_points)}/{TOTAL_KEYPOINTS}；mm_per_pixel：未设置"
        )
        if measurements:
            axis_actual = str(measurements.get("axis_mode_selected", ""))
            curvature_polyline = measurements.get("curvature_index_polyline")
            curvature_text = f"；折线弯曲指数 {float(curvature_polyline):.4f}" if curvature_polyline is not None else ""
            st.info(f"中轴线模式：{axis_mode_label(axis_actual)}{curvature_text}")
            if needs_review:
                st.warning("鱼体弯曲较明显，建议人工复核。")

        overlay = draw_keypoints_and_measurements(
            st.session_state.working_image,
            keypoints,
            measurements,
            image_name=image_name,
            specimen_id=specimen_id,
            scale_points=[] if st.session_state.scale_method_label == "aruco_board_coordinate_auto" else st.session_state.scale_points,
            axis_mode_selected=str((measurements or {}).get("axis_mode_selected", "")),
        )
        display_image, scale_factor = resize_for_display(overlay, max_width=max_display_width)
        click_capture(display_image, scale_factor, click_mode)

    with helper_col:
        render_annotation_helper(current_definition, helper_mode, click_mode, completed)

    if completed:
        row = build_result_row(
            image_name=image_name,
            specimen_id=specimen_id,
            source_type=source_type,
            scale_method=st.session_state.scale_method_label,
            mm_per_pixel=st.session_state.mm_per_pixel,
            keypoints=keypoints,
            measurements=measurements or {},
            needs_review=needs_review,
            notes=notes,
        )
        st.subheader("计算结果")
        metric_columns = [
            "TL_straight_axis_mm",
            "TL_axis_polyline_mm",
            "TL_axis_spline_mm",
            "TL_final_mm",
            "caudal_tip_selected",
            "caudal_extension_axis_mm",
            "SL_straight_mm",
            "SL_axis_polyline_mm",
            "SL_axis_spline_mm",
            "SL_final_mm",
            "curvature_index_polyline",
            "curvature_index_spline",
            "axis_mode_selected",
            "body_depth_mm",
            "head_length_straight_mm",
            "snout_length_straight_mm",
            "caudal_peduncle_length_straight_mm",
            "caudal_peduncle_length_axis_mm",
            "caudal_peduncle_depth_mm",
        ]
        st.dataframe(pd.DataFrame([{column: row[column] for column in metric_columns}]), use_container_width=True)

        if auto_save:
            message = save_current_outputs(output_dir, image_name, specimen_id, source_type, notes, axis_mode_request)
            st.success(message)

        if st.button("保存/覆盖当前结果"):
            message = save_current_outputs(output_dir, image_name, specimen_id, source_type, notes, axis_mode_request, force=True)
            st.success(message)

        csv_path = output_dir / "results.csv"
        xlsx_path = output_dir / "results.xlsx"
        if csv_path.exists() or xlsx_path.exists():
            dl_csv, dl_xlsx = st.columns(2)
            if csv_path.exists():
                dl_csv.download_button("下载 CSV", data=csv_path.read_bytes(), file_name="results.csv", mime="text/csv")
            if xlsx_path.exists():
                dl_xlsx.download_button(
                    "下载 Excel",
                    data=xlsx_path.read_bytes(),
                    file_name="results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )


if __name__ == "__main__":
    main()
