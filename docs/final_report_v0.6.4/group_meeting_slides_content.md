# v0.6.4 组会幻灯片内容草稿

## Slide 1: SiganusMorph Local

蓝子鱼照片半自动形态测量工具：AI-assisted keypoints + human confirmation + geometric measurement QC。

## Slide 2: 为什么需要半自动流程

- 传统手工测量耗时。
- 尾鳍末端、尾柄、体高等点位容易主观偏移。
- 需要保留人工复核和可追踪日志。

## Slide 3: 当前关键点体系

- 16 个手动关键点。
- P7U/P7L 替代单个 P7。
- P7V 作为全长虚拟终点。

## Slide 4: 当前推荐工作流

```text
image -> fish mask -> heatmap models -> v0.6 selector
-> P7V/QC -> user correction -> corrected_keypoints
-> dual-axis measurement -> CSV/Excel
```

## Slide 5: v0.6 selector

- 不整体替换模型。
- 按关键点选择 v0.1、v0.5、v0.4.1 或几何规则。
- P4/P9 等 hard points 得到改善。

## Slide 6: 双轴测量

- model_axis：兼容人工 C1-C4。
- body_midline_axis：更平滑的几何测量轴。
- auto_qc_gated：根据 QC 推荐 selected axis。

## Slide 7: real-world review

- 30 张 review sample。
- 29 张 confirmed。
- 1 张 failed_preannotation excluded。
- 平均确认耗时约 85.5 秒/图。

## Slide 8: stable batch measurement

- 96 张纳入 final analysis dataset。
- P7V valid 96/96。
- 14 张 outlier 已人工 preview 复核并保留。

## Slide 9: 当前限制

- 强弯曲鱼尚未开发 curved-fish mode。
- outlier 需要结合生物学解释和人工测量验证。
- C1-C4 是否未来移出模型训练目标仍需进一步评估。

## Slide 10: 下一步

- 人工测量一致性验证。
- MAE / MAPE / ICC / Bland-Altman。
- 准备论文方法图和流程图。

