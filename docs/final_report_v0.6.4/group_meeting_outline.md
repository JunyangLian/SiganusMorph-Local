# v0.6.4 组会汇报提纲

## 1. 研究目标

- 建立蓝子鱼照片半自动形态测量流程。
- 降低人工关键点标注工作量。
- 输出可复核、可追踪、可批量分析的测量数据。

## 2. 数据与拍摄

- 真实蓝子鱼照片。
- A3 校准板与 warped image。
- 16 个人工确认关键点 + P7V 派生点。

## 3. 方法演进

- v0.3：16 点体系与 P7V。
- v0.4：heatmap + geometry hybrid。
- v0.5：扩增 corrected labels，训练 heatmap-U-Net v0.5。
- v0.6：keypoint-wise hybrid selector。
- v0.6.4：双轴测量和 stable batch measurement。

## 4. 当前稳定工作流

- v0.6 AI-assisted preannotation。
- auto_qc_gated measurement axis。
- 人工确认后保存 corrected_keypoints。
- 批量测量与 outlier review。

## 5. 核心结果

- real-world review：29 confirmed / 1 excluded。
- 平均确认耗时约 85.5 秒/图。
- batch measurement：96 张，P7V valid 96/96。
- outlier 14 张已人工 preview 复核，暂不排除。

## 6. 下一步

- 人工测量一致性验证。
- 自动测量 vs 手工测量统计检验。
- 准备正式方法描述和图表。

