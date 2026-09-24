# SiganusMorph Local 用户指南

## 1. 启动

```bash
pip install -r requirements.txt
streamlit run app.py
```

Windows:

```bat
run_app.bat
```

页面地址：

```text
http://localhost:8501
```

## 2. 推荐页面

普通页面默认使用推荐工作流：

- 预标注：`v0.6 keypoint-wise hybrid selector`
- 测量轴：`auto_qc_gated recommended`
- 显示：关键点、P7V、selected measurement axis、body_midline_axis、QC warning

复杂选项默认隐藏在 `Advanced settings / Developer mode` 中。

## 3. 标注流程

1. 选择图片来源。
2. 对真实 warped image，比例默认使用板面坐标 `0.1 mm/pixel`。
3. 点击 `Run AI-assisted preannotation`。
4. 检查关键点、P7V 和 warning。
5. 选择需要修正的点，在图上点击新位置。
6. 点击 Apply 更新点位。
7. 保存 `corrected annotation`。

## 4. 点位编辑说明

- 左键点击图像：记录 pending coordinate。
- Apply：将 pending coordinate 应用到当前选择的关键点。
- Undo：撤销上一次编辑。
- 保存前不会写入正式 corrected JSON。

## 5. Measurement axis mode

普通模式默认 `auto_qc_gated`。

可选轴：

- `model_axis`：使用 C1-C4 等 confirmed keypoints。
- `body_midline_axis`：使用去鳍后的鱼身主体中线和 P4-P5 midpoint。
- `auto_qc_gated`：当 body_midline_axis QC 通过且双轴差异不大时使用 body_midline_axis，否则保留 model_axis。

切换测量轴不会覆盖 `corrected_keypoints` 或 C1-C4，只影响曲线测量和导出字段。

## 6. 保存结果

保存 corrected annotation 后会输出：

- keypoints JSON
- annotated preview
- measurements CSV/Excel
- correction summary

真实数据当前主要输出目录：

```text
results/real_annotation_5_15_v0.3_corrected/
results/realworld_review_v0.4.1/
```

## 7. 批量测量

稳定批量测量结果位于：

```text
results/batch_measurement_v0.6.4_stable/
```

正式分析建议优先使用：

```text
results/batch_measurement_v0.6.4_stable/final_analysis_dataset.csv
results/batch_measurement_v0.6.4_stable/final_analysis_dataset.xlsx
```

