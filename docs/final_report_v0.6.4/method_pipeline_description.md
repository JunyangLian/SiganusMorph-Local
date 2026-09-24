# v0.6.4 方法流程描述

## 输入

输入为标准化拍摄的蓝子鱼侧面照片。图像经 A3 校准板检测和透视校正后进入标注流程。

## 预标注

v0.6 推荐流程结合：

- fish mask segmentation
- heatmap-U-Net v0.1
- heatmap-U-Net v0.5
- v0.4.1 hybrid geometry
- keypoint-wise source selection
- P7V derivation and QC

## 人工确认

自动预标注不会直接成为训练标签或正式测量标签。用户需要检查并保存为 `corrected_keypoints`。

## 测量

测量同时计算两套轴：

- `model_axis`
- `body_midline_axis`

`auto_qc_gated` 根据 QC 选择用于 selected output 的测量轴。

## 输出

输出包括：

- corrected keypoint JSON
- annotated preview
- measurement CSV/Excel
- QC summary
- outlier review files
- final analysis dataset

