# SiganusMorph Local

本项目是一个本地运行的蓝子鱼/篮子鱼照片半自动形态测量工具。当前版本目标是先完成 V0.1 可用链路：上传或选择图片、比例校准、人工点击 16 个关键点、自动派生 P7V 虚拟全长终点、计算直线和中轴线形态指标，并保存结果表、关键点 JSON 和带标注复核图。V0.2 已预留 ArUco 检测、透视校正和板面坐标校准。

## 快速开始

```bash
pip install -r requirements.txt
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

项目内置了一个轻量 Streamlit 图片点击组件，安装 `requirements.txt` 后即可直接在图上点击关键点。

## 当前功能

- 上传单张图片，或从本地文件夹选择图片。
- 支持三种比例方式：
  - 点击尺子两点并输入真实距离。
  - 直接输入 `mm_per_pixel`。
  - 尝试识别 ArUco。旧板使用尺子自动比例；A3 V2 板使用已知板面坐标作为主比例，尺子只作为 QC。
- 按固定顺序点击 16 个关键点，其中 P7U/P7L 分别记录尾鳍上叶和下叶末端，程序自动生成 P7V 全长虚拟终点。
- 支持中轴线模式选择：自动推荐、折线中轴线、平滑曲线中轴线。
- 标注界面右侧提供模板鱼辅助区，支持新手模式和熟练模式。
- 计算 TL、SL、体高、头长、吻长、尾柄长、尾柄高。
- 保存：
  - `results/results.csv`
  - `results/results.xlsx`
  - `results/annotations/*_annotated.png`
  - `results/keypoints/*_keypoints.json`

## 关键点顺序

1. P1 `snout_tip`：吻端
2. P2 `eye_front`：眼眶前缘
3. P3 `operculum_posterior`：鳃盖骨后缘
4. P4 `peduncle_start_midpoint`：尾柄前端中轴点
5. P5 `caudal_base_midpoint`：尾鳍基部中点
6. P6 `caudal_fork_midpoint`：尾鳍分叉点
7. P7U `caudal_fin_upper_tip`：尾鳍上叶末端
8. P7L `caudal_fin_lower_tip`：尾鳍下叶末端
9. P8 `body_depth_dorsal`：最大体高上点
10. P9 `body_depth_ventral`：最大体高下点
11. P10 `peduncle_depth_dorsal`：尾柄最窄处上点
12. P11 `peduncle_depth_ventral`：尾柄最窄处下点
13. C1 `head_axis_point`：头后部中轴点
14. C2 `trunk_axis_point`：躯干中部中轴点
15. C3 `posterior_trunk_axis_point`：躯干后部中轴点
16. C4 `peduncle_axis_point`：尾柄中轴点

主中轴线固定按 `P1 → C1 → C2 → C3 → P4 → C4 → P5` 连接。尾部轴向优先使用 `P5 → P6`，程序比较 P7U/P7L 在该方向上的投影，生成虚拟点 P7V；曲线全长为 `SL_curve_mm + caudal_extension_axis_mm`。

## 输出字段

结果表每张图一行，包含图片名、样本编号、样本来源、比例方式、`mm_per_pixel`、16 个手动关键点坐标、P7V 派生点坐标、直线测量指标、中轴线长度字段，以及 V2.0 兼容字段 `SL_curve_mm`、`TL_curve_mm`、`curvature_index`。

新增中轴线字段：

```text
axis_mode_selected
P7V_caudal_fin_posterior_endpoint_x
P7V_caudal_fin_posterior_endpoint_y
caudal_tip_selected
caudal_extension_axis_mm
TL_straight_axis_mm
TL_axis_polyline_mm
TL_axis_spline_mm
TL_final_mm
SL_axis_polyline_mm
SL_axis_spline_mm
curvature_index_polyline
curvature_index_spline
SL_final_mm
caudal_peduncle_length_axis_mm
```

自动推荐规则：`curvature_index_polyline <= 1.02` 推荐折线中轴线；大于 `1.02` 推荐平滑曲线中轴线；大于 `1.05` 时标记 `needs_review=True`。

## 标注模板素材

模板鱼示意图位于：

```text
assets/templates/
```

当前模板底图为 `assets/templates/template_fish_source.png`。包括 `overview_template.png`、`P1_template.png` 到 `P11_template.png`、`P7U_template.png`、`P7L_template.png`、`C1_template.png` 到 `C4_template.png`，以及对应的局部放大图。重新生成模板素材：

```bash
python scripts/generate_template_assets.py
```

## 校准板

项目内已放入当前 A3 打印版校准板：

```text
data/calibration_board/bluefish_calibration_board_A3_v3b_print.pdf
```

新版 A3 V2 ChArUco + Plumb-line 校准板：

```text
data/calibration_board/v2_charuco_plumb/siganusmorph_a3_v2_charuco_plumb_print.pdf
```

重新生成新版板子：

```bash
python scripts/generate_a3_charuco_plumb_board.py
```

打印时请选择实际大小 100%，不要缩放。正式拍摄前建议先拍一张空板图测试 ArUco 和比例尺识别。

## 后续路线

- V0.2：完善 ArUco 标记真实坐标配置、自动比例尺换算、编号区辅助记录。
- V0.2 当前已加入 A3 V2 多 ArUco 板面坐标校准；0-35 cm 尺子降级为质量检查，后续会继续增强镜头去畸变和质控评分。
- V0.3：加入蓝色背景鱼体分割和 mask 预览。
- V1.0：基于人工关键点 JSON 训练关键点模型。
- V2.0：加入中轴线曲线长度、弯曲指数和弯曲样本复核标记。
