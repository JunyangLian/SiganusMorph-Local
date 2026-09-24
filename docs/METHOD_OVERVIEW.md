# 方法概览

## 总体流程

```text
real image / warped image
-> fish mask segmentation
-> heatmap keypoint prediction
-> mask/geometric rules
-> v0.6 keypoint-wise selector
-> P7V derivation
-> human correction
-> corrected_keypoints
-> dual-axis measurement
-> CSV/Excel export + QC
```

## 鱼体分割

当前校准板为蓝色背景，因此第一版分割使用 HSV/颜色阈值、形态学处理和最大连通域提取。fish mask 用于：

- fish bbox / crop
- P1 mask-left-boundary 修正
- P6/P7U/P7L 尾部几何辅助
- body_midline_axis
- local-normal 体高/尾柄高 QC

## 关键点模型

当前保留两代 heatmap-U-Net：

- v0.1：median error 更低，部分稳定点表现好。
- v0.5：mean error 更低，hard cases 改善明显。

v0.6 keypoint-wise selector 会按关键点选择更合适的来源，而不是整体替换模型。

## 几何规则

主要规则：

- P1：fish mask 最左边界。
- P7U/P7L：尾部 mask 与 heatmap 互相 QC。
- P6：tail gap / fork notch QC。
- P7V：由 P5、P7U、P7L 和 tail axis 派生。
- P8/P9、P10/P11：local-normal suggestion 仅作为 QC。

## 双轴测量

`model_axis` 使用 confirmed C1-C4。

`body_midline_axis` 使用去鳍后的鱼身主体中线：

- C1-C3 几何中线点仅用于测量轴。
- C4 使用 midpoint(P4, P5)。
- 不覆盖人工确认的 C1-C4。

`auto_qc_gated` 根据 QC 自动选择测量轴。

## 质量控制

QC 包括：

- P7V valid
- segmentation success
- dual-axis disagreement
- curvature level
- local-normal measurement warning
- outlier ratio warning

