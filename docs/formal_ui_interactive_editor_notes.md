# 正式 V1.0 前台交互式测量编辑说明

本文档说明正式前台“单鱼测量”页面的精简 overlay 和拖拽编辑逻辑。正式 V1.0 单鱼测量流程已简化为“导入与自动测量 → 复核、修正与导出”。该逻辑只服务普通用户展示和测量流程，不影响 Review Queue、Advanced overlay 或开发者工具。

## 两步式正式流程

正式单鱼测量页只显示两个主步骤：

1. `Step 1 导入与自动测量`：上传单张图片，填写 `specimen_id` 和 `weight_g`，点击“开始自动测量”。系统内部依次完成自动校准、生成 `warped_image`、鱼体识别、自动预标注和形态测量。校准仍是硬门槛，失败时不会继续测量，也不会使用默认 `mm_per_pixel = 0.1`。
2. `Step 2 复核、修正与导出`：在 `warped_image` 坐标中显示正式测量图像、可拖拽测量点、`body_midline_axis`、体高线、尾柄高线和 P7V，并提供测量表、重置、保存和导出功能。

校准状态、marker 数量、`mm_per_pixel`、原始图 marker overlay 和 `warped_image` 放在 Step 1 的“校准详情”折叠区中，默认收起。

## 正式界面显示的点

正式 overlay 只显示最终用于测量的点和线：

- `P1_snout_tip`
- `P2_eye_anterior`
- `P3_operculum_posterior`
- `P4_peduncle_anterior`
- `P5_caudal_base`
- `P6_caudal_fork`
- `P7U_upper_lobe_tip`
- `P7L_lower_lobe_tip`
- `body_depth_upper`
- `body_depth_lower`
- `peduncle_depth_upper`
- `peduncle_depth_lower`
- 派生点 `P7V_virtual_tail_tip`

其中 `body_depth_upper/lower` 是正式体高测量端点，`peduncle_depth_upper/lower` 是正式尾柄高测量端点。它们不必等同于开发流程中的原始 P8/P9/P10/P11 点。

## 正式界面隐藏的内容

正式页面默认隐藏：

- C1、C2、C3、C4 点；
- C1-C4 构成的模型中轴线；
- 原始红色 P8/P9/P10/P11 调试点；
- v0.1/v0.5 heatmap 点；
- v0.4.1 / v0.6 对比点；
- legacy axis、debug contours 和 audit 字段。

这些内容仍保留在高级工具中。

## 测量轴

正式界面只显示 `body_midline_axis`，即基于鱼体轮廓中心线得到的平滑测量轴。P7V 的正式显示方向优先使用 body_midline_axis 在 P5 附近或尾柄末端的局部切线方向。

## 拖拽编辑

用户可以直接在图像上拖动可见测量点：

1. 鼠标按下并选中最近点；
2. 拖动点位；
3. 拖动过程完全在前端 canvas 内完成；
4. 松开鼠标后仍不会立即触发 Streamlit 整页刷新；
5. 组件内部临时测量面板实时更新；
6. 用户点击组件内“应用修改”后，点位才同步回 Python；
7. 当前会话中的 `formal_measurement_points` 更新；
8. 页面测量表格刷新。

拖动结果记录在 `measurement_overrides` 中。除非用户在 Step 2 点击“保存人工确认结果”，否则不会写出文件。

## 为什么采用前端本地拖拽

Streamlit 的页面状态更新会触发脚本重跑。如果在 `mousemove` 或 `mouseup` 中频繁向 Python 发送坐标，页面会出现闪烁或短暂白屏。正式前台的测量编辑器因此采用前端本地状态：

- 拖动过程中不调用 `Streamlit.setComponentValue`；
- 松开鼠标时不自动触发整页 rerun；
- 前端 canvas 内部实时重绘点位、测量线、P7V 和临时测量值；
- 只有点击“应用修改”时才把最终点位传回 Python。

这样可以保持拖拽连续、稳定，减少页面重绘对人工对点的干扰。

## 点位显示半径与命中半径

正式编辑器将“看得见的点”和“可选中的范围”分开：

- 普通关键点可见半径约 4 px；
- 体高和尾柄高测量端点可见半径约 5 px；
- P7V 星形点约 6 px；
- hover 或选中时点位会放大到约 7-8 px；
- 实际鼠标命中范围约 14-18 px，但不显示出来。

这样既能精确观察鱼体边界，又不至于难以选中点位。

## 精确微调

选中点位后可使用键盘方向键微调：

- 方向键：每次移动 1 px；
- Shift + 方向键：每次移动 5 px。

键盘微调同样只更新前端临时状态，不会立即触发 Streamlit rerun。点击“应用修改”后，表格才会同步更新。

## 放大镜

正式编辑器提供局部放大镜开关。启用后，鼠标附近会显示局部放大窗口，帮助查看鱼嘴、鳃盖边缘、体高边界、尾柄边界和尾鳍端点。

放大镜只用于显示，不改变图像坐标，也不会写入任何结果。

## 标签显示

默认情况下点位标签隐藏，避免遮挡鱼体结构。鼠标 hover 或选中点位时会显示当前点名和坐标。用户也可以打开“显示点位标签”开关，让所有点位显示简短标签。

## 即时更新的测量

拖动点位后会重算：

- `TL_compressed_virtual_mm`
- `TL_open_projection_mm`
- `SL_mm`
- `FL_mm`
- `body_depth_mm`
- `head_length_mm`
- `snout_length_mm`
- `caudal_peduncle_length_mm`
- `caudal_peduncle_depth_mm`

其中：

- 拖动 P1 会影响 TL、SL、FL、头长、吻长；
- 拖动 P5 会影响 SL、尾柄长、P7V 和 TL；
- 拖动 P7U/P7L 会影响 P7V、虚拟压拢尾鳍总长和尾鳍角度；
- 拖动 `body_depth_upper/lower` 会影响体高；
- 拖动 `peduncle_depth_upper/lower` 会影响尾柄高。

## 保存和历史数据保护

正式前台的“保存人工确认结果”只写入：

```text
results/formal_v1_user_outputs/
```

单鱼测量页的 `specimen_id` 和 `weight_g` 属于正式前台会话元数据。`weight_g` 单位为 g，是可选字段，未称重时可以保持为空或 0.0；填写后会进入当前测量表、CSV、Excel、测量 JSON、formal points JSON 和 preview metadata。该字段只用于导出记录和后续长度-重量关系、肥满度或生长记录分析，不会改变关键点坐标，也不会写入历史 corrected_keypoints。

它不会覆盖：

- 历史 `corrected_keypoints`；
- `batch_measurements.csv`；
- `final_analysis_dataset.csv`；
- Review Queue 中的人工标注。

## 与高级工具的区别

正式前台强调成熟测量流程和简洁展示；高级工具用于开发者复核、模型对比、Advanced overlay、QC 细节和历史评估结果追溯。

## 校准与坐标系统

正式前台严格区分三套坐标：

- `raw_image`：上传的原始照片，只用于校准板检测和人工检查校准点；
- `warped_image`：校准成功后得到的透视校正图，是正式测量唯一背景图；
- `display canvas`：浏览器中缩放显示的 canvas，用于交互拖拽。

自动预标注、分割、正式测量点、body_midline_axis、P7V 和导出预览图均使用 `warped_image` 坐标。显示 canvas 只负责交互展示，拖动后的显示坐标会按缩放比例转换回 `warped_image` 像素坐标后再用于计算。

校准失败时，正式页面不会使用默认 `mm_per_pixel` 继续测量，也不会在 raw image 上叠加 warped coordinates。
