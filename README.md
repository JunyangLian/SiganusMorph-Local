# SiganusMorph Local

## 项目简介

SiganusMorph Local 是一个本地运行的蓝子鱼照片半自动形态测量工具。它面向标准化拍摄的鱼体侧面照片，支持校准板透视校正、AI-assisted 关键点预标注、人工确认修正、P7V 虚拟全长终点派生、几何测量轴、批量测量导出和 QC/异常值复核。

项目的核心原则是：自动模型只负责减少人工点击工作量，最终用于训练和正式测量的数据仍然来自人工确认后的 `corrected_keypoints`。

## 当前稳定版本

当前稳定版本：`v0.6.4 stable`

当前推荐工作流已加入 `v0.6.5_tail_and_body_depth_geometry_fix` 的清理项：P6 尾鳍分叉点几何 QC 会在普通模式中提示明显异常；P8/P9 的几何体高扫描仅保留为 Advanced QC suggestion，不作为默认点位来源。

推荐工作流：

- 自动预标注：`v0.6 keypoint-wise hybrid selector`
- 测量轴：`auto_qc_gated recommended`
- 普通页面默认隐藏复杂设置，只显示推荐流程
- `Advanced settings / Developer mode` 中保留 `v0.4.1`、`model_axis`、`body_midline_axis` 等开发选项

`v0.6.4 stable` 不会用几何中轴点覆盖人工确认的 C1-C4。`body_midline_axis` 只作为测量轴候选，用于曲线长度和弯曲 QC。

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

启动 Streamlit：

```bash
streamlit run app.py
```

Windows 也可以直接运行：

```bat
run_app.bat
```

打开本地页面：

```text
http://localhost:8501
```

## 推荐使用流程

1. 选择已透视校正的真实图片，或导入新图片。
2. 点击 `Run AI-assisted preannotation`。
3. 检查 16 个关键点、P7V、尾鳍上下叶和 QC warning。
4. 必要时人工修正点位。
5. 保存 `corrected annotation`。
6. 批量导出 measurement CSV/Excel。
7. 查看 QC、比例指标和 outlier 标记。

## 关键点体系

当前手动关键点为 16 个：

| 编号 | key | 含义 |
| --- | --- | --- |
| P1 | `P1_snout_tip` | 吻端 |
| P2 | `P2_eye_front` | 眼眶前缘 |
| P3 | `P3_operculum_posterior` | 鳃盖骨后缘 |
| P4 | `P4_peduncle_start_midpoint` | 尾柄前端中轴点 |
| P5 | `P5_caudal_base_midpoint` | 尾鳍基部中点 |
| P6 | `P6_caudal_fork_midpoint` | 尾鳍分叉点 |
| P7U | `P7U_caudal_fin_upper_tip` | 尾鳍上叶末端 |
| P7L | `P7L_caudal_fin_lower_tip` | 尾鳍下叶末端 |
| P8 | `P8_body_depth_dorsal` | 最大体高上点 |
| P9 | `P9_body_depth_ventral` | 最大体高下点 |
| P10 | `P10_peduncle_depth_dorsal` | 尾柄最窄处上点 |
| P11 | `P11_peduncle_depth_ventral` | 尾柄最窄处下点 |
| C1 | `C1_head_axis_point` | 头后部中轴点 |
| C2 | `C2_trunk_axis_point` | 躯干中部中轴点 |
| C3 | `C3_posterior_trunk_axis_point` | 躯干后部中轴点 |
| C4 | `C4_peduncle_axis_point` | 尾柄中轴点 |

派生点：

- `P7V_caudal_fin_posterior_endpoint`：由 P5、P7U、P7L 和 tail axis 计算得到的虚拟全长终点，不需要手动点击。

重要约定：

- `corrected_keypoints` 是最终人工确认标签。
- `body_midline_axis` 不覆盖 C1-C4。
- P7V 用于全长计算，避免把吻端到尾鳍单侧末端的斜向距离误当成全长。

## 测量指标

主要输出指标包括：

- 全长 `TL_final_mm`
- 标准长 `SL_final_mm`
- 体高 `body_depth_mm`
- 头长 `head_length_mm`
- 吻长 `snout_length_mm`
- 尾柄长 `caudal_peduncle_length_mm`
- 尾柄高 `caudal_peduncle_depth_mm`
- 弯曲指数 `curvature_index`
- 推荐测量轴 `selected_measurement_axis`
- 双轴字段：`model_axis` 与 `body_midline_axis` 对应的 TL/SL/curvature/smoothness

从 `v0.6.4` 起，CSV/Excel 同时保留 `model_axis` 和 `body_midline_axis` 的曲线测量值，并输出 `auto_qc_gated` 选择的 `selected` 测量值。

## 当前模型与算法

当前推荐预标注流程由多部分组成：

- 鱼体蓝色校准板背景分割与 fish mask
- heatmap-U-Net v0.1 / v0.5 关键点模型
- `v0.6 keypoint-wise hybrid selector`
- P1、P6、P7U/P7L 等 mask/geometric rules
- P7V 几何派生
- local-normal 体高/尾柄高 suggestion 和 QC
- `body_midline_axis` 几何测量轴
- `auto_qc_gated` 测量轴选择
- `v0.6.5` P6 geometry QC：检查 P6 是否位于 P5 与尾鳍末端之间、是否靠近上下尾叶之间的分叉区域，并在异常时提示人工复核
- `v0.6.5` geometric P8/P9：基于 `body_midline_axis` 的局部法线最大体高扫描，仅作为 Advanced 参考，不替代 v0.6 默认 P8/P9

当前模型权重：

```text
models/siganusmorph_heatmap_unet_v0.1/preannotation_candidate.pt
models/siganusmorph_heatmap_unet_v0.5/preannotation_candidate.pt
```

## 当前验证结果

关键验证结果：

- real-world review：29 张 confirmed，1 张 excluded
- 平均人工确认耗时约 85.5 秒/图
- 平均每图移动 4.28 / 16 个点
- v0.6 same-set comparison：mean error 降低，P4/P9 明显改善
- v0.6.3 dual-axis comparison：`body_midline_axis` 更平滑
- v0.6.4 end-to-end regression：pass
- stable batch measurement：96 张纳入正式测量，P7V valid 96/96
- final analysis dataset：96 rows，outlier 已人工 preview 复核并全部保留
- v0.6.5 tail/body-depth geometry check：96 张评估中 P6 geometry QC 标记 13 张需要复核；geometric P8/P9 误差高于 v0.6 默认点，因此仅作为 Advanced suggestion

## 输出目录

重要结果目录：

```text
results/batch_measurement_v0.6.4_stable/
results/batch_measurement_v0.6.4_stable/final_analysis_dataset.csv
results/e2e_regression_v0.6.4/
results/model_eval/
results/realworld_review_v0.4.1/
```

常用结果文件：

```text
results/batch_measurement_v0.6.4_stable/batch_measurements.csv
results/batch_measurement_v0.6.4_stable/batch_measurements.xlsx
results/batch_measurement_v0.6.4_stable/final_analysis_dataset.csv
results/batch_measurement_v0.6.4_stable/final_analysis_dataset.xlsx
results/batch_measurement_v0.6.4_stable/measurement_outlier_cases.csv
```

## 文档入口

- [用户指南](docs/USER_GUIDE.md)
- [方法概览](docs/METHOD_OVERVIEW.md)
- [版本历史](docs/VERSION_HISTORY.md)
- [数据结构](docs/DATA_STRUCTURE.md)
- [故障排查](docs/TROUBLESHOOTING.md)
- [v0.6.4 组会材料](docs/final_report_v0.6.4/)

## 后续计划

- 人工测量一致性验证
- 自动测量 vs 手工测量：MAE、MAPE、ICC、Bland-Altman
- 检查 outlier 是否来自真实个体比例差异、姿态差异或拍摄质量
- 如未来需要处理强弯曲鱼，再开发 curved-fish mode

## 旧 README

旧版 README 已备份为：

```text
README_legacy_v0.1.md
```
