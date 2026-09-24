# 版本历史

## v0.1-v0.2

- 建立本地 Streamlit 工具。
- 支持图像加载、比例校准、人工关键点点击和基础测量。
- 加入 ArUco / ChArUco 校准板与 warped image 流程。

## v0.3

- 关键点体系更新为 16 个手动点。
- 单个 P7 尾鳍末端改为 P7U/P7L。
- 新增 P7V 虚拟全长终点。
- 第一次人工标注迁移为 `corrected_keypoints`。

## v0.4

- 训练 heatmap-U-Net v0.1。
- 建立 v0.4 hybrid heatmap + geometry 预标注流程。
- 加入 real-world review workflow。

## v0.4.1

- 加入 curvature-aware measurement。
- 加入 local-normal body depth / peduncle depth suggestion。
- 完成 30 张 real-world review，其中 29 张 confirmed，1 张 excluded。

## v0.5

- 整合原始 corrected labels 与 real-world review confirmed labels。
- 构建 heatmap-U-Net v0.5 candidate dataset。
- v0.5 对 hard cases 有改善，但不适合整体替换 v0.1。

## v0.6

- 开发 keypoint-wise hybrid selector。
- 对不同关键点选择 v0.1、v0.5、v0.4.1 hybrid 或几何规则。
- 接入为 experimental mode。

## v0.6.2-v0.6.3

- 实验 body contour midline。
- 发现其不适合替代 C1/C3/C4 keypoints，但适合作为更平滑的测量轴。
- 建立 dual-axis measurement comparison。

## v0.6.4 stable

- 接入 `measurement_axis_mode`：
  - `model_axis`
  - `body_midline_axis`
  - `auto_qc_gated`
- 完成 end-to-end regression test。
- 完成 stable batch measurement。
- 生成 final analysis dataset。
- 普通页面默认使用推荐流程：v0.6 + auto_qc_gated。

