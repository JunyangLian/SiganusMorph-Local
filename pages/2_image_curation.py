from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph.real_dataset import (  # noqa: E402
    CURATION_COLUMNS,
    load_manifest,
    real_manifest_path,
    relpath,
    save_manifest,
)


def to_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def from_bool(value: object) -> str:
    return "true" if bool(value) else "false"


st.set_page_config(page_title="Image curation", layout="wide")
st.title("真实图片筛选 / 去重")

manifest_path = real_manifest_path(PROJECT_ROOT)
df = load_manifest(PROJECT_ROOT)
if df.empty:
    st.info("还没有找到 data/real_images_manifest.csv。请先导入真实图片。")
    st.stop()

st.caption(f"Manifest: {relpath(manifest_path, PROJECT_ROOT)}")
st.write("这里只修改 manifest 标记，不会物理删除任何图片。训练导出默认只使用 `keep_for_training=true` 的真实图片。")

specimen_ids = ["全部"] + sorted([item for item in df["specimen_id"].astype(str).unique().tolist() if item])
selected_specimen = st.selectbox("按 specimen_id 查看", specimen_ids)
view_df = df.copy()
if selected_specimen != "全部":
    view_df = view_df[view_df["specimen_id"].astype(str) == selected_specimen].copy()

show_only_candidates = st.checkbox("只显示 keep_for_training=true 且 warp_success=true", value=False)
if show_only_candidates:
    view_df = view_df[
        view_df["keep_for_training"].astype(str).str.lower().eq("true")
        & view_df["warp_success"].astype(str).str.lower().eq("true")
    ].copy()

work = view_df.copy()
work["thumbnail"] = work["imported_path"].apply(lambda value: str((PROJECT_ROOT / str(value)).resolve()))
work["keep_for_annotation"] = work["keep_for_annotation"].apply(to_bool)
work["keep_for_training"] = work["keep_for_training"].apply(to_bool)
for column in ("duplicate_group", "exclude_reason", "quality_score", "curation_notes"):
    work[column] = work[column].fillna("").astype(str)
work["quality_score"] = pd.to_numeric(work["quality_score"], errors="coerce")

editable_columns = [
    "thumbnail",
    "image_name",
    "original_file_name",
    "specimen_id",
    "warp_success",
    "keep_for_annotation",
    "keep_for_training",
    "duplicate_group",
    "exclude_reason",
    "quality_score",
    "curation_notes",
]

edited = st.data_editor(
    work.loc[:, editable_columns],
    use_container_width=True,
    height=760,
    num_rows="fixed",
    column_config={
        "thumbnail": st.column_config.ImageColumn("缩略图", width="small"),
        "keep_for_annotation": st.column_config.CheckboxColumn("keep_for_annotation"),
        "keep_for_training": st.column_config.CheckboxColumn("keep_for_training"),
        "duplicate_group": st.column_config.TextColumn("duplicate_group", help="近重复组，例如 dup_001"),
        "exclude_reason": st.column_config.SelectboxColumn(
            "exclude_reason",
            options=["", "near_duplicate", "blurred", "tail_unclear", "warp_failed", "body_occluded", "other"],
        ),
        "quality_score": st.column_config.NumberColumn("quality_score", min_value=0, max_value=5, step=1),
        "curation_notes": st.column_config.TextColumn("curation_notes"),
    },
    disabled=["thumbnail", "image_name", "original_file_name", "specimen_id", "warp_success"],
)

summary = (
    df.groupby("specimen_id", dropna=False)
    .agg(
        num_images=("image_name", "count"),
        keep_for_training=("keep_for_training", lambda s: sum(str(v).lower() == "true" for v in s)),
        warp_success=("warp_success", lambda s: sum(str(v).lower() == "true" for v in s)),
    )
    .reset_index()
)
with st.expander("按 specimen_id 汇总", expanded=False):
    st.dataframe(summary, use_container_width=True)

if st.button("保存筛选/去重标记", type="primary"):
    updated = df.copy()
    edited_by_name = edited.set_index("image_name")
    for image_name, row in edited_by_name.iterrows():
        mask = updated["image_name"].astype(str) == str(image_name)
        if not mask.any():
            continue
        updated.loc[mask, "keep_for_annotation"] = from_bool(row["keep_for_annotation"])
        updated.loc[mask, "keep_for_training"] = from_bool(row["keep_for_training"])
        for column in ("duplicate_group", "exclude_reason", "quality_score", "curation_notes"):
            value = row.get(column, "")
            if column == "quality_score" and pd.isna(value):
                value = ""
            updated.loc[mask, column] = str(value).strip()
    save_manifest(PROJECT_ROOT, updated)
    st.success("已保存筛选字段到 data/real_images_manifest.csv。")

st.info(
    "建议：同一条鱼的近重复连拍可以写同一个 duplicate_group；"
    "保留最清晰、关键点最完整的图片为 keep_for_training=true，其余设为 false 并填写 exclude_reason。"
)
