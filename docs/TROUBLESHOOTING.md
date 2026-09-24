# 故障排查

## 页面打不开

确认 Streamlit 是否启动：

```bash
streamlit run app.py
```

访问：

```text
http://localhost:8501
```

## 找不到 warped image

检查：

```text
data/real_images_warped/
data/real_images_manifest.csv
```

如果图片尚未透视校正，需要先运行导入和 warp 流程。

## 自动预标注失败

常见原因：

- 鱼体不在放置区。
- 反光严重。
- 尾部或眼睛不可见。
- fish mask segmentation failed。
- 模型文件路径不存在。

建议：

1. 检查 warped image 是否正常。
2. 检查 fish mask preview。
3. 在页面中手动标注或修正。
4. 将失败图标记为 review/excluded，不要直接进入训练。

## 点位点击后偏移

当前页面使用固定 display-to-original 坐标转换。若出现偏移：

- 检查是否处于 zoom/crop 视图。
- 查看 Coordinate debug 面板。
- 确认点击后 pending preview 中蓝色/红色十字是否重合。

## P7V invalid

优先检查：

- P5 `caudal_base_midpoint`
- P6 `caudal_fork_midpoint`
- P7U `caudal_fin_upper_tip`
- P7L `caudal_fin_lower_tip`
- C4 `peduncle_axis_point`

P7V invalid 时，TL 相关字段会标记为需要复核。

## P6 geometry QC warning

如果页面提示：

```text
P6 可能不可靠，请检查尾鳍分叉点。
```

含义是 P6 `P6_caudal_fork_midpoint` 没有通过尾部几何检查。系统会检查 P6 是否位于 P5 之后、尾鳍末端之前，是否靠近 P7U/P7L 之间的分叉凹陷区域，以及是否过度偏向某一个尾叶。

处理建议：

1. 打开尾部区域，人工确认 P6 是否落在尾鳍上下叶之间的分叉点。
2. 如果 Advanced settings 中显示 fallback suggestion，可将它作为参考，但不要盲目接受。
3. 已保存的 `corrected_keypoints` 不会被 P6 fallback 自动覆盖。
4. 如果是未确认预标注图，系统可能用 `mask_fork_rule_fallback` 初始化 P6，但仍会提示人工复核。

## body_midline_axis 与 C1-C4 不一致

这是预期行为。`body_midline_axis` 是几何测量轴，不会覆盖 C1-C4 corrected labels。

如果 `dual_axis_disagreement = true`，需要人工复核曲线测量。

## geometric P8/P9 与默认体高点不一致

`geometric P8/P9` 是基于 `body_midline_axis` 和局部法线扫描得到的实验性体高建议点。v0.6.5 评估显示它目前不如 v0.6 默认 P8/P9 稳定，因此：

- 普通推荐模式仍使用 v0.6 keypoint-wise selector 的 P8/P9。
- geometric P8/P9 只在 Advanced settings 中作为蓝色建议点和最大体高截面参考显示。
- 它不会写入 `corrected_keypoints`，也不会替换 `body_depth_mm`。
- 如果体高点看起来异常，可以同时参考默认 P8/P9、geometric P8/P9 和鱼体轮廓，再人工决定是否修正。

## outlier 不一定是错误

`measurement_outlier_cases.csv` 用于提示人工复核。当前 14 张 outlier preview 已人工查看，未见明显点位错误，因此保留进入 `final_analysis_dataset`，但保留 outlier flag 方便追踪。
