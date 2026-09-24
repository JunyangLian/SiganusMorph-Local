from __future__ import annotations

from pathlib import Path

import streamlit as st

from siganusmorph.formal_ui import (
    available_measurement_tables,
    dataframe_to_csv_bytes,
    dataframe_to_excel_bytes,
    read_measurement_table,
    simplified_measurement_table,
)
from siganusmorph.ui_styles import apply_formal_theme, metric_grid, page_header, step_header


PROJECT_ROOT = Path(__file__).resolve().parents[1]

apply_formal_theme()
page_header(
    "结果导出",
    "集中查看正式测量结果，导出普通用户常用字段或完整内部字段。",
    eyebrow="Export Center",
)

tables = available_measurement_tables(PROJECT_ROOT)
if not tables:
    with st.container(border=True):
        st.warning("尚未找到可展示的测量结果表。请先完成单鱼或批量测量。")
    st.stop()

with st.container(border=True):
    step_header("1", "选择结果数据", "选择已有测量表，查看简化结果并导出。")
    label = st.selectbox("结果数据", list(tables))
    path = tables[label]
    df = read_measurement_table(path)
    simple_df = simplified_measurement_table(df)
    metric_grid(
        [
            ("记录数", len(simple_df), ""),
            ("字段数", len(simple_df.columns), ""),
            ("数据源", label, ""),
            ("导出格式", "CSV / Excel", ""),
        ]
    )

with st.container(border=True):
    step_header("2", "简化测量表", "默认展示普通测量和软著截图所需字段。")
    st.dataframe(simple_df, use_container_width=True, hide_index=True)

with st.container(border=True):
    step_header("3", "导出文件", "简化导出面向普通用户，完整导出保留全部内部字段。")
    dl_cols = st.columns(4)
    with dl_cols[0]:
        st.download_button(
            "导出简化 CSV",
            dataframe_to_csv_bytes(simple_df),
            file_name="measurement_export_simple.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl_cols[1]:
        st.download_button(
            "导出简化 Excel",
            dataframe_to_excel_bytes(simple_df),
            file_name="measurement_export_simple.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with dl_cols[2]:
        st.download_button(
            "导出完整 CSV",
            dataframe_to_csv_bytes(df),
            file_name="measurement_export_full.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl_cols[3]:
        st.download_button(
            "导出完整 Excel",
            dataframe_to_excel_bytes(df),
            file_name="measurement_export_full.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    with st.expander("字段说明", expanded=False):
        st.markdown(
            """
            - `image_name`：图像文件名。
            - `specimen_id`：样本编号。
            - `weight_g`：可选样品重量，单位为 g；未填写时保持为空。
            - `TL_compressed_virtual_mm`：虚拟压拢尾鳍后的候选总长。
            - `TL_open_projection_mm`：历史投影法总长口径。
            - `SL_mm`：标准长。
            - `FL_mm`：叉长字段；若当前数据源未生成该字段，则保持为空。
            - `measurement_status`：测量状态或复核建议。
            """
        )

st.caption("本页面只读取结果表用于展示和下载，不修改历史测量文件。")
