from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph.real_dataset import (  # noqa: E402
    check_specimen_groups,
    load_manifest,
    real_manifest_path,
    relpath,
    save_manifest,
)


st.set_page_config(page_title="Specimen assignment", layout="wide")
st.title("真实图片 specimen_id 手动分组")

manifest_path = real_manifest_path(PROJECT_ROOT)
df = load_manifest(PROJECT_ROOT)
if df.empty:
    st.info("还没有找到 data/real_images_manifest.csv。请先运行 `python scripts/import_real_heic_images.py`。")
    st.stop()

work = df.copy()
work["thumbnail"] = work["imported_path"].apply(lambda value: str((PROJECT_ROOT / str(value)).resolve()))
editable_columns = [
    "thumbnail",
    "image_name",
    "original_file_name",
    "specimen_id",
    "view_id",
    "notes",
    "width",
    "height",
]
for column in editable_columns:
    if column not in work.columns:
        work[column] = ""

st.caption(f"Manifest: {relpath(manifest_path, PROJECT_ROOT)}")
st.write("给同一条鱼的多张照片填写相同 `specimen_id`，例如 `fish_001`。`view_id` 可以写 `view_1`、`view_2`。")

edited = st.data_editor(
    work.loc[:, editable_columns],
    use_container_width=True,
    height=720,
    num_rows="fixed",
    column_config={
        "thumbnail": st.column_config.ImageColumn("缩略图", width="small"),
        "specimen_id": st.column_config.TextColumn("specimen_id", help="同一条鱼使用相同编号，例如 fish_001"),
        "view_id": st.column_config.TextColumn("view_id", help="同一条鱼不同照片视角，例如 view_1"),
        "notes": st.column_config.TextColumn("notes"),
    },
    disabled=["thumbnail", "image_name", "original_file_name", "width", "height"],
)

col_save, col_check = st.columns(2)
if col_save.button("保存 specimen_id / view_id 到 manifest", type="primary"):
    updated = df.copy()
    for column in ("specimen_id", "view_id", "notes"):
        updated[column] = edited[column].fillna("").astype(str)
    save_manifest(PROJECT_ROOT, updated)
    st.success("已保存到 data/real_images_manifest.csv")

if col_check.button("检查 specimen_id"):
    result = check_specimen_groups(PROJECT_ROOT)
    st.write(
        {
            "未填写 specimen_id 的图片数": result["missing_count"],
            "唯一 specimen_id 数": result["num_specimens"],
            "summary": relpath(result["summary_path"], PROJECT_ROOT),
        }
    )
    summary = result["summary"]
    if isinstance(summary, pd.DataFrame) and not summary.empty:
        st.dataframe(summary, use_container_width=True)
    if result["num_specimens"] != 35:
        st.warning("当前唯一 specimen_id 数不等于 35。可以继续编辑，重点是确保同一条鱼不要被拆到多个编号。")
