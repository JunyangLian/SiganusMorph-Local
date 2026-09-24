from __future__ import annotations

from pathlib import Path

import streamlit as st

from siganusmorph.ui_styles import apply_formal_theme, feature_card, page_header, render_sidebar_brand


PAGES_DIR = Path(__file__).resolve().parent / "pages"


def formal_page_path(local_name: str, server_name: str) -> str:
    local_path = PAGES_DIR / local_name
    selected = local_path if local_path.exists() else PAGES_DIR / server_name
    return selected.relative_to(PAGES_DIR.parent).as_posix()


st.set_page_config(
    page_title="蓝子鱼形态参数智能测量与标注系统 V1.0",
    page_icon="",
    layout="wide",
)


def render_home() -> None:
    """Formal V1.0 homepage for normal users and copyright screenshots."""
    apply_formal_theme()
    page_header(
        "蓝子鱼形态参数智能测量与标注系统 V1.0",
        "面向标准化单鱼侧位图像的智能形态测量与标注工具，支持图像校准、自动测量、人工复核、结果导出和质量控制。",
        eyebrow="SiganusMorph Local",
        media=True,
    )

    action_cols = st.columns([1, 1, 1, 2])
    with action_cols[0]:
        st.page_link(formal_page_path("1_单鱼测量.py", "1_single_fish.py"), label="开始单鱼测量", use_container_width=True)
    with action_cols[1]:
        st.page_link(formal_page_path("2_批量测量.py", "2_batch_measurement.py"), label="批量测量", use_container_width=True)
    with action_cols[2]:
        st.page_link(formal_page_path("3_结果导出.py", "3_results_export.py"), label="查看结果导出", use_container_width=True)
    with action_cols[3]:
        st.markdown(
            '<div class="sm-help">推荐流程：导入图片 → 校准 → 自动测量 → 人工修正 → 导出结果。</div>',
            unsafe_allow_html=True,
        )

    st.markdown("### 核心功能")
    cards = [
        ("图像校准", "检测校准板标记，生成用于测量的透视校正图像和比例尺。"),
        ("AI 预标注", "自动识别鱼体关键点，生成推荐测量点和测量轴。"),
        ("人工复核", "在正式测量界面中拖拽修正点位，结果即时更新。"),
        ("形态参数导出", "导出 CSV、Excel、预览图和标注 JSON，用于后续分析。"),
        ("高级质量控制", "对异常姿态、尾鳍展开、遮挡和反光图像提供复核入口。"),
    ]
    st.markdown(
        '<div class="sm-feature-grid">'
        + "".join(feature_card(title, body) for title, body in cards)
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### 推荐使用流程")
    flow_steps = [
        ("1", "导入图片", "上传标准化侧位照片。"),
        ("2", "校准", "检测校准板并生成校正图。"),
        ("3", "自动测量", "计算关键点、测量轴和形态参数。"),
        ("4", "人工修正", "必要时拖拽修正测量点。"),
        ("5", "导出结果", "保存表格、预览图和 JSON。"),
    ]
    flow_html = "".join(
        (
            '<div class="sm-flow-card">'
            f'<div class="sm-flow-number">0{number}</div>'
            f'<div class="sm-flow-title">{title}</div>'
            f'<div class="sm-flow-body">{desc}</div>'
            "</div>"
        )
        for number, title, desc in flow_steps
    )
    st.markdown(f'<div class="sm-flow-grid">{flow_html}</div>', unsafe_allow_html=True)

    st.markdown("### 适用图像条件")
    condition_cols = st.columns([1.2, 1])
    with condition_cols[0]:
        st.markdown(
            """
            - 鱼体侧面平放，轮廓完整；
            - 校准板或比例尺清晰可见；
            - 鱼体边界、吻端、鳃盖和尾鳍上下叶可辨认；
            - 图像清晰，无明显运动模糊；
            - 遮挡、强反光和极端弯曲较少。
            """
        )
    with condition_cols[1]:
        st.warning("异常姿态、鱼鳍明显展开、遮挡或反光严重时，需要人工复核关键点和测量结果。")


def main() -> None:
    """Use hidden routing plus a deterministic Chinese public sidebar."""
    apply_formal_theme()
    home_page = st.Page(render_home, title="首页", icon=":material/home:", default=True)
    single_page = st.Page(
        formal_page_path("1_单鱼测量.py", "1_single_fish.py"),
        title="单鱼测量",
        icon=":material/photo_camera:",
    )
    batch_page = st.Page(
        formal_page_path("2_批量测量.py", "2_batch_measurement.py"),
        title="批量测量",
        icon=":material/folder_open:",
    )
    export_page = st.Page(
        formal_page_path("3_结果导出.py", "3_results_export.py"),
        title="结果导出",
        icon=":material/download:",
    )
    contact_page = st.Page(
        formal_page_path("4_联系我们.py", "4_contact.py"),
        title="联系我们",
        icon=":material/contact_support:",
    )
    public_pages = [home_page, single_page, batch_page, export_page, contact_page]
    navigation = st.navigation(public_pages, position="hidden")

    render_sidebar_brand()
    st.sidebar.page_link(home_page, label="首页", icon=":material/home:")
    st.sidebar.page_link(
        single_page,
        label="单鱼测量",
        icon=":material/photo_camera:",
    )
    st.sidebar.page_link(
        batch_page,
        label="批量测量",
        icon=":material/folder_open:",
    )
    st.sidebar.page_link(
        export_page,
        label="结果导出",
        icon=":material/download:",
    )
    st.sidebar.page_link(
        contact_page,
        label="联系我们",
        icon=":material/contact_support:",
    )
    navigation.run()


if __name__ == "__main__":
    main()
