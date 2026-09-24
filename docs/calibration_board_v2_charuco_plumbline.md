# A3 V2 ChArUco + Plumb-line 校准板

新版 A3 板用于提高透视校正、镜头畸变评估和校正质量检查的稳定性。

## 文件位置

```text
data/calibration_board/v2_charuco_plumb/siganusmorph_a3_v2_charuco_plumb_print.pdf
data/calibration_board/v2_charuco_plumb/siganusmorph_a3_v2_charuco_plumb_300dpi.png
data/calibration_board/v2_charuco_plumb/siganusmorph_a3_v2_charuco_plumb.svg
data/calibration_board/v2_charuco_plumb/siganusmorph_a3_v2_charuco_plumb_metadata.json
```

## 主要变化

1. 定位 ArUco 从 4 个增加到 8 个：
   - ID 0-3：四角。
   - ID 4-7：上、右、下、左四边中点。
2. 定位 ArUco 使用 `DICT_4X4_50`，便于兼容旧流程。
3. 新增 ChArUco 标定区，使用 `DICT_6X6_250`。
4. 新增 plumb-line 直线质控元素：
   - 外框线。
   - 鱼体放置框。
   - 水平/垂直参考线。
5. 鱼体放置区继续保持大面积蓝色背景，方便后续鱼体分割。

## 推荐使用流程

正式拍鱼前，先拍 3-5 张空板图：

1. 使用固定焦段，优先手机 1x 或 2x，不要用超广角。
2. 相机尽量垂直向下。
3. 板面压平，避免塑封或亚克力翘曲。
4. 软件检测 8 个 ArUco，并用多点 homography 校正到真实板面坐标。
5. V2 板主比例来自板面坐标：`4200 px = 420 mm`，即 `mm_per_pixel = 0.1`。
6. 软件检测 0-35 cm 尺子、ChArUco 面板和 plumb-line 参考线，输出质控结果。

## 质控建议

如果出现以下情况，应重新拍摄或标记为需要复核：

1. 8 个定位 ArUco 未全部识别。
2. 多点 homography 重投影误差偏大。
3. 0-35 cm 尺子识别失败或尺子刻度间距变异系数偏大。
5. 校正后直线参考线仍明显弯曲或倾斜。
6. 右侧或边缘反光遮挡 ArUco、尺子或基准线。

## 注意

Plumb-line 直线法当前主要用于质控和后续镜头畸变校正，不建议直接使用非线性“橡皮布变形”强行拉直鱼体图像。更安全的顺序是：

```text
镜头去畸变
→ 多 ArUco / ChArUco 平面校正
→ 板面坐标比例校准
→ 尺子比例 QC
→ 直线质控
→ 鱼体测量
```
