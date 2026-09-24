from __future__ import annotations

import streamlit as st

from siganusmorph.ui_styles import apply_formal_theme, feature_card, page_header, step_header


apply_formal_theme()
page_header(
    "高级工具",
    "此区域用于开发者复核、模型调试和历史 Review Queue，不作为普通用户默认测量入口。",
    eyebrow="Developer and Review Tools",
)

st.warning("高级工具包含开发和复核功能。请不要将未人工确认的自动预标注直接作为训练标签。")

with st.container(border=True):
    step_header("1", "复核与开发入口", "保留旧工具，但从正式测量流程中折叠到高级区域。")
    tool_cols = st.columns(2)
    with tool_cols[0]:
        st.markdown(
            feature_card(
                "Review Queue / Advanced Overlay",
                "用于真实图片逐张复核、显示高级 overlay、检查 QC warning 和调试几何建议点。普通用户无需进入该页面。",
            ),
            unsafe_allow_html=True,
        )
        st.code("streamlit run pages/3_realworld_review.py", language="powershell")
    with tool_cols[1]:
        st.markdown(
            feature_card(
                "旧开发后台",
                "原 app.py 开发后台已迁移保留，包含历史标注、模型对比、调试入口和开发用结果检查。",
            ),
            unsafe_allow_html=True,
        )
        st.code("streamlit run developer_tools/legacy_app.py", language="powershell")

with st.container(border=True):
    step_header("2", "模型评估与审计资料", "供开发者追踪历史实验、评估和质量控制结果。")
    info_cols = st.columns(4)
    cards = [
        ("Review Queue", "人工复核、修正日志和 corrected labels 检查。"),
        ("Advanced Overlay", "热图点、几何建议点、QC warning 和调试图层。"),
        ("模型评估", "关键点误差、hard cases、版本比较和选择报告。"),
        ("历史结果检查", "批量测量、压拢尾鳍 TL 审计和导出字段核对。"),
    ]
    for col, (title, body) in zip(info_cols, cards):
        with col:
            st.markdown(feature_card(title, body), unsafe_allow_html=True)

with st.container(border=True):
    step_header("3", "常用文档", "正式说明书、方法概览、数据结构和问题排查入口。")
    doc_cols = st.columns(3)
    with doc_cols[0]:
        st.markdown("- `docs/USER_GUIDE.md`\n- `docs/METHOD_OVERVIEW.md`")
    with doc_cols[1]:
        st.markdown("- `docs/DATA_STRUCTURE.md`\n- `docs/TROUBLESHOOTING.md`")
    with doc_cols[2]:
        st.markdown("- `docs/software_copyright_ui_notes.md`\n- `docs/user_manual_v1.0_draft.md`")

st.info(
    "正式 V1.0 首页、单鱼测量、批量测量和结果导出是普通用户推荐入口；高级工具仅用于开发者和高级复核。"
)
