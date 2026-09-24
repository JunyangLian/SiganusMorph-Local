from __future__ import annotations

from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def rel(path: str) -> Path:
    return PROJECT_ROOT / path


def status_badge(ok: bool) -> str:
    return "可用" if ok else "未找到"


def render_entry_table() -> None:
    entries = [
        ("旧开发后台", "developer_tools/legacy_app.py", "streamlit run developer_tools/legacy_app.py"),
        ("Review Queue", "pages/3_realworld_review.py", "streamlit run pages/3_realworld_review.py"),
        ("物种/个体编号整理", "pages/1_specimen_assignment.py", "streamlit run pages/1_specimen_assignment.py"),
        ("图像筛选", "pages/2_image_curation.py", "streamlit run pages/2_image_curation.py"),
        ("V2 后端", "siganusmorph_v2/backend/app.py", "python -m uvicorn siganusmorph_v2.backend.app:app --host 127.0.0.1 --port 8000 --reload"),
    ]
    rows = []
    for name, path, command in entries:
        rows.append(
            {
                "功能入口": name,
                "相对路径": path,
                "状态": status_badge(rel(path).exists()),
                "启动命令": command,
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


st.set_page_config(
    page_title="SiganusMorph V2.0 Admin Console",
    layout="wide",
)

st.title("SiganusMorph V2.0 Admin Console")
st.caption("管理者端入口：保留 Streamlit 全功能后台，用于复核、调试、数据整理和后续优化。")

st.warning("本页面面向管理者/开发者，不作为普通用户测量入口。请勿将未人工确认的自动预标注直接作为训练标签。")

summary_cols = st.columns(4)
summary_cols[0].metric("用户端输出", "results/v2_user_outputs/")
summary_cols[1].metric("管理端输出", "results/v2_admin_outputs/")
summary_cols[2].metric("V1.0 历史结果", "只读保护")
summary_cols[3].metric("模型训练", "本入口不执行")

st.subheader("保留的管理功能")
feature_cols = st.columns(3)
features = [
    ("样本与个体整理", "specimen assignment、image curation、source batch 追踪。"),
    ("人工复核", "single image review、realworld review、Advanced overlay。"),
    ("质量控制", "compressed-tail TL audit、measurement QC、异常样本复核。"),
    ("模型评估", "历史模型评估结果查看和后续评估脚本入口。"),
    ("数据导出", "dataset export、training dataset planning。"),
    ("开发者后台", "保留旧 Streamlit 后台，不作为普通用户入口。"),
]
for col, (title, body) in zip(feature_cols * 2, features):
    with col:
        st.markdown(f"**{title}**")
        st.caption(body)

st.subheader("入口状态与启动命令")
render_entry_table()

st.subheader("安全边界")
st.markdown(
    """
    - V2.0 用户端输出写入 `results/v2_user_outputs/`。
    - 管理者端后续输出建议写入 `results/v2_admin_outputs/`。
    - 本入口不训练模型，不修改 `corrected_keypoints`，不覆盖历史 batch/final analysis。
    - 旧 Review Queue、Advanced overlay 和开发后台保留为管理者工具。
    """
)

st.subheader("常用启动命令")
cmd_cols = st.columns(2)
with cmd_cols[0]:
    st.markdown("**V2 FastAPI Backend**")
    st.code("python -m uvicorn siganusmorph_v2.backend.app:app --host 127.0.0.1 --port 8000 --reload", language="powershell")
    st.markdown("**V2 Backend Smoke Test**")
    st.code("python siganusmorph_v2/backend/smoke_test_api.py", language="powershell")
with cmd_cols[1]:
    st.markdown("**旧开发后台**")
    st.code("streamlit run developer_tools/legacy_app.py", language="powershell")
    st.markdown("**Review Queue**")
    st.code("streamlit run pages/3_realworld_review.py", language="powershell")

st.info("普通用户请使用 React/Tauri 用户端；本 Admin Console 仅用于管理者复核和研发维护。")
