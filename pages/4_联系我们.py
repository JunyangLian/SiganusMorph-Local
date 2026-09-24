from __future__ import annotations

import os

import streamlit as st

from siganusmorph.ui_styles import apply_formal_theme, feature_card, page_header


apply_formal_theme()
page_header(
    "联系我们",
    "获取使用支持、反馈测量问题或咨询软件部署与数据导出。",
    eyebrow="Support & Contact",
)

contact_name = os.getenv("SIGANUSMORPH_CONTACT_NAME", "SiganusMorph 项目维护者").strip()
contact_org = os.getenv("SIGANUSMORPH_CONTACT_ORG", "").strip()
contact_email = os.getenv("SIGANUSMORPH_CONTACT_EMAIL", "").strip()
contact_note = os.getenv("SIGANUSMORPH_CONTACT_NOTE", "").strip()

left, right = st.columns([1.05, 1])
with left:
    st.markdown("### 联系信息")
    st.markdown(feature_card("软件维护", contact_name), unsafe_allow_html=True)
    if contact_org:
        st.markdown(feature_card("单位", contact_org), unsafe_allow_html=True)
    if contact_email:
        st.markdown(f"**联系邮箱**  \n[{contact_email}](mailto:{contact_email})")
    else:
        st.info("公开联系邮箱尚未配置，请联系软件提供者。")
    if contact_note:
        st.caption(contact_note)

with right:
    st.markdown("### 反馈问题时请提供")
    st.markdown(
        """
        - 图片文件名和样本编号；
        - 页面显示的处理状态；
        - 校准是否成功；
        - 需要复核的点位或测量项；
        - 问题截图和大致发生时间。
        """
    )
    st.warning("请勿在公开反馈中包含账号密码、服务器 Token 或其他敏感信息。")

st.markdown("### 使用建议")
st.markdown(
    "对校准失败、异常姿态、明显遮挡、强反光或鱼鳍过度展开的图像，"
    "建议重新拍摄或在测量页进行人工复核，不建议直接将自动结果用于正式分析。"
)
