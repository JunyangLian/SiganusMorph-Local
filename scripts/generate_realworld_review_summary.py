"""Generate the v0.4.1 real-world review summary report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "results" / "realworld_review_v0.4.1" / "post_review_evaluation"
REPORT_PATH = ROOT / "realworld_review_summary_report.md"
JSON_PATH = ROOT / "realworld_review_summary.json"


def _num(value: object, default: float = np.nan) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except Exception:
        return default


def _int_num(value: object, default: int = 0) -> int:
    try:
        if value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _fmt(value: object, digits: int = 2) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _pct(value: object) -> str:
    return f"{float(value) * 100:.1f}%"


def _table(df: pd.DataFrame, columns: list[str], headers: list[str] | None = None, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    headers = headers or columns
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = _fmt(value, 3)
            values.append(str(value).replace("|", "/"))
        lines.append("|" + "|".join(values) + "|")
    return "\n".join(lines)


def main() -> None:
    efficiency = pd.read_csv(ROOT / "review_efficiency_summary.csv", dtype=str, keep_default_na=False)
    point = pd.read_csv(ROOT / "manual_correction_by_point.csv")
    image = pd.read_csv(ROOT / "manual_correction_by_image.csv")
    qc = pd.read_csv(ROOT / "qc_warning_accuracy.csv", dtype=str, keep_default_na=False)
    excluded = pd.read_csv(ROOT / "excluded_cases.csv", dtype=str, keep_default_na=False)

    row = efficiency.iloc[0].to_dict() if not efficiency.empty else {}
    total_sample_count = _int_num(row.get("total_sample_count", 30))
    reviewed_count = _int_num(row.get("reviewed_count", row.get("num_images_reviewed", 0)))
    excluded_count = _int_num(row.get("excluded_count", 0))
    pending_count = _int_num(row.get("pending_count", max(0, total_sample_count - reviewed_count - excluded_count)))
    completion = str(row.get("completion_rate_excluding_excluded", ""))
    mean_review_time_sec = _num(row.get("mean_review_time_sec"))
    median_review_time_sec = _num(row.get("median_review_time_sec"))
    mean_num_points_moved = _num(row.get("mean_num_points_moved"))
    median_num_points_moved = _num(row.get("median_num_points_moved"))
    mean_displacement_mm = _num(row.get("mean_displacement_mm"))
    median_displacement_mm = _num(row.get("median_displacement_mm"))
    most_freq = [item for item in str(row.get("most_frequently_moved_keypoints", "")).split(";") if item]

    point = point.copy()
    point["was_moved_count"] = (point["num_images"] * point["moved_rate"]).round().astype(int)
    point_sorted_moved = point.sort_values(["moved_rate", "mean_displacement_mm"], ascending=[False, False])
    point_sorted_mean = point.sort_values(["mean_displacement_mm", "moved_rate"], ascending=[False, False])
    stable = point[
        (point["moved_rate"] <= 0.10)
        & (point["median_displacement_mm"] == 0)
        & (point["max_displacement_mm"] <= 5.0)
    ].sort_values(["moved_rate", "mean_displacement_mm"])

    qc_summary = pd.DataFrame()
    if not qc.empty:
        qc_work = qc.copy()
        qc_work["has_large_correction_bool"] = qc_work["has_large_correction"].astype(str).str.lower().eq("true")
        qc_work["num_points_moved_num"] = pd.to_numeric(qc_work["num_points_moved"], errors="coerce")
        qc_summary = (
            qc_work.groupby("qc_warning_type")
            .agg(
                count=("image_name", "count"),
                large_correction_rate=("has_large_correction_bool", "mean"),
                mean_num_points_moved=("num_points_moved_num", "mean"),
                example_images=("image_name", lambda s: ";".join(list(dict.fromkeys(s.astype(str)))[:5])),
            )
            .reset_index()
            .sort_values(["count", "large_correction_rate"], ascending=[False, False])
        )
        qc_summary["large_correction_rate_pct"] = qc_summary["large_correction_rate"].map(_pct)

    image = image.copy()
    image["potential_failure_flag"] = (
        (image["max_displacement_mm"] > 10.0)
        | (image["num_large_corrections"] >= 3)
        | (image["num_points_moved"] >= 7)
    )
    potential_failures = image[image["potential_failure_flag"]].sort_values(
        ["max_displacement_mm", "num_large_corrections"], ascending=[False, False]
    )
    most_modified_images = image.sort_values(["num_points_moved", "mean_displacement_mm"], ascending=[False, False]).head(5)
    largest_image_displacements = image.sort_values(
        ["max_displacement_mm", "mean_displacement_mm"], ascending=[False, False]
    ).head(5)

    excluded_lines = []
    for _, excluded_row in excluded.iterrows():
        excluded_lines.append(
            f"- `{excluded_row.get('image_name', '')}` / `{excluded_row.get('specimen_id', '')}`: "
            f"`{excluded_row.get('exclude_reason', '')}`. {excluded_row.get('suggested_next_action', '')}"
        )

    top_moved_names = point_sorted_moved.head(5)["keypoint_name"].tolist()
    top_displacement_names = point_sorted_mean.head(5)["keypoint_name"].tolist()
    stable_names = stable["keypoint_name"].tolist()
    recommend_train_v05 = bool(
        mean_num_points_moved >= 3.0
        or point_sorted_moved.head(3)["moved_rate"].max() >= 0.5
        or point_sorted_mean.head(3)["mean_displacement_mm"].max() >= 3.0
    )
    recommendation_text = (
        "建议准备 v0.5 数据集和训练计划，但先不要立即训练；继续用 v0.4.1 作为默认预标注流程，"
        "同时把这 29 张 confirmed labels 作为后续 v0.5 候选增量数据。"
        if recommend_train_v05
        else "建议暂不训练 v0.5，继续使用 v0.4.1，并积累更多真实 review 样本。"
    )

    report = f"""# v0.4.1 Real-world Review Summary Report

## 一、样本概况

- 总 review 样本数：**{total_sample_count}**
- confirmed reviewed images：**{reviewed_count}**
- excluded images：**{excluded_count}**
- pending images：**{pending_count}**
- excluding excluded 后完成率：**{completion}**
- 本轮性质：真实未训练图泛化测试。review sample 来自真实鱼照片，统计结果只基于 29 张人工确认后的 confirmed images；AI 合成图未参与本轮结论。

Excluded case:
{chr(10).join(excluded_lines) if excluded_lines else "- 无"}

## 二、人工确认效率

- 平均确认耗时：**{_fmt(mean_review_time_sec, 1)} 秒/图**
- 中位确认耗时：**{_fmt(median_review_time_sec, 1)} 秒/图**
- 平均每图需要移动点数：**{_fmt(mean_num_points_moved, 2)} / 16 点**
- 中位每图需要移动点数：**{_fmt(median_num_points_moved, 1)} / 16 点**
- 平均点位移动距离：**{_fmt(mean_displacement_mm, 3)} mm**
- 中位点位移动距离：**{_fmt(median_displacement_mm, 3)} mm**
- 高频修正点：`{"; ".join(most_freq)}`

解释：中位移动距离为 0，说明大多数点在多数图片中无需调整；但平均每图仍有约 4.3 个点需要人工动一下，主要集中在少数结构点。

## 三、每个点的修正情况

### 最常被改的 5 个点

{_table(point_sorted_moved.assign(moved_rate_pct=point_sorted_moved["moved_rate"].map(_pct)), ["keypoint_name", "was_moved_count", "moved_rate_pct", "mean_displacement_mm", "median_displacement_mm", "max_displacement_mm"], ["keypoint", "moved_count", "moved_rate", "mean_mm", "median_mm", "max_mm"], max_rows=5)}

### 平均移动距离最大的 5 个点

{_table(point_sorted_mean, ["keypoint_name", "was_moved_count", "moved_rate", "mean_displacement_mm", "median_displacement_mm", "max_displacement_mm"], ["keypoint", "moved_count", "moved_rate", "mean_mm", "median_mm", "max_mm"], max_rows=5)}

### 基本稳定、很少被改的点

{_table(stable, ["keypoint_name", "was_moved_count", "moved_rate", "mean_displacement_mm", "median_displacement_mm", "max_displacement_mm"], ["keypoint", "moved_count", "moved_rate", "mean_mm", "median_mm", "max_mm"]) if not stable.empty else "无"}

补充：`P3_operculum_posterior`、`P7U_caudal_fin_upper_tip`、`P7L_caudal_fin_lower_tip` 的 moved_rate 不高，但存在少数大离群，仍建议复核时看一眼。

## 四、按图片的修正情况

### 修改点数最多的图片

{_table(most_modified_images, ["image_name", "specimen_id", "num_points_moved", "mean_displacement_mm", "max_displacement_mm", "num_large_corrections", "large_correction_keypoints"], ["image", "specimen", "moved_points", "mean_mm", "max_mm", "large_count", "large_keypoints"])}

### 最大单点偏移最高的图片

{_table(largest_image_displacements, ["image_name", "specimen_id", "num_points_moved", "mean_displacement_mm", "max_displacement_mm", "num_large_corrections", "large_correction_keypoints"], ["image", "specimen", "moved_points", "mean_mm", "max_mm", "large_count", "large_keypoints"])}

潜在需要复盘的 confirmed 图：`{"; ".join(potential_failures["image_name"].astype(str).tolist()) if not potential_failures.empty else "无"}`。
这些图不是失败样本，但有较大的人工修正，适合后续作为 v0.5 hard cases 或 QC 调参样本。

## 五、QC warning 有效性

{_table(qc_summary, ["qc_warning_type", "count", "large_correction_rate_pct", "mean_num_points_moved", "example_images"], ["warning", "count", "large_correction_rate", "mean_moved_points", "examples"]) if not qc_summary.empty else "本轮无 QC warning。"}

结论：

- `P9_body_depth_ventral_heatmap_unreliable` 最常出现，且大修正命中率约 83%，是有效 warning。
- `heatmap_axis_qc_failed` 出现 9 次，大修正命中率约 89%，说明中轴点 QC 有实际价值。
- `P7U/P7L heatmap_mask_disagreement` 出现次数少，但命中率高，适合保留。
- curvature warning 在本轮样本中出现少，但都伴随大修正，建议继续保留。
- `C2_trunk_axis_point_heatmap_axis_unreliable` 只出现 1 次且没有 large correction，可能是误报或轻度提示，暂不急着改。

## 六、v0.4.1 真实泛化表现结论

v0.4.1 适合继续作为默认自动预标注流程。理由是：

- 29 张 confirmed 图片中，完成率 excluding excluded 达到 100%。
- 中位点位移动距离为 0 mm，多数点在多数图上无需调整。
- 平均确认耗时约 85.5 秒/图，已经进入可用的人工确认工作流。
- `P1_snout_tip`、`P2_eye_front`、`P5_caudal_base_midpoint`、`P6_caudal_fork_midpoint` 在本轮几乎不需要修正。

仍需重点复核：

- `P4_peduncle_start_midpoint`：27/29 被移动，平均位移 4.49 mm。
- `C2_trunk_axis_point`：23/29 被移动，平均位移 4.18 mm，并有多个大离群。
- `P9_body_depth_ventral`：20/29 被移动，平均位移 2.86 mm。
- `P8_body_depth_dorsal`、`P11_peduncle_depth_ventral`、`C1/C3/C4`：多数情况下可用，但仍常需要轻微确认。
- `P3` 和 `P7U/P7L`：不是高频错误，但一旦错可能偏得较大。

`real_037.png / fish_18` 的 failed_preannotation 目前更像个别异常或某类困难样本提示，不应影响 v0.4.1 作为默认流程，但应进入 error cases，后续用于分析“识别完全失败”的触发条件。

## 七、是否建议训练 v0.5

{recommendation_text}

判断依据：

- 优点：v0.4.1 已经显著减少从零标注工作量，很多点无需动。
- 训练信号：P4、C2、P9 被频繁修正，说明新增 29 张 corrected labels 对 v0.5 有价值。
- 风险：这批样本只有 29 张，建议先导出 v0.5 candidate dataset、做 hard-case 清单和训练计划，再决定是否训练。

建议下一步：

1. 继续使用 v0.4.1 做默认预标注。
2. 把这 29 张 confirmed labels 纳入 v0.5 候选数据池。
3. 单独标记 P4/C2/P9 和 P3/P7U 离群样本作为 hard cases。
4. 先生成 v0.5 dataset export + split plan，确认无泄漏后再训练。
"""

    REPORT_PATH.write_text(report, encoding="utf-8-sig")
    summary = {
        "total_sample_count": total_sample_count,
        "reviewed_count": reviewed_count,
        "excluded_count": excluded_count,
        "mean_review_time_sec": mean_review_time_sec,
        "median_review_time_sec": median_review_time_sec,
        "mean_num_points_moved": mean_num_points_moved,
        "median_num_points_moved": median_num_points_moved,
        "mean_displacement_mm": mean_displacement_mm,
        "median_displacement_mm": median_displacement_mm,
        "top_moved_keypoints": top_moved_names,
        "top_displacement_keypoints": top_displacement_names,
        "stable_keypoints": stable_names,
        "recommend_train_v05": recommend_train_v05,
        "recommendation_text": recommendation_text,
    }
    JSON_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {JSON_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
