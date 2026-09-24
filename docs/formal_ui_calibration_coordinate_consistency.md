# 正式 V1.0 前台校准与坐标一致性说明

本文档说明正式“单鱼测量”页面中的校准门槛和坐标流。该逻辑只作用于正式前台，不影响 Review Queue、Advanced overlay 或开发者工具。

## 坐标流

正式测量采用以下坐标流程：

```text
raw_image
→ calibration detection
→ homography H_raw_to_warped
→ warped_image / corrected_image
→ segmentation / preannotation / measurement points
→ display canvas
```

其中：

- `raw_image` 是上传原图，只用于校准检查；
- `H_raw_to_warped` 是从原图到校正图的透视变换；
- `warped_image` 是自动测量、人工拖拽和导出预览唯一使用的图像背景；
- `formal_measurement_points` 全部保存为 warped image 像素坐标；
- display canvas 只是缩放显示，不能作为测量坐标保存。

## 校准硬门槛

正式前台新增 `formal_calibration_state`：

```python
{
    "status": "not_run | success | failed | manual_required",
    "homography_raw_to_warped": ...,
    "mm_per_pixel": ...,
    "mm_per_pixel_source": ...,
    "warped_image": ...,
    "detected_markers": ...,
    "detected_board_corners_raw": ...,
    "calibration_message": ...
}
```

只有 `status == "success"` 时，Step 3 自动测量按钮才可用。

如果校准失败：

- 不显示默认比例尺作为成功结果；
- 不允许自动测量；
- 不在原图上叠加 warped 坐标；
- 提示重新拍摄、重新校准或手动修正校准点。

## Step 2 显示内容

校准步骤同时显示：

1. 原始图和检测到的 marker / board corner overlay；
2. 校正后 `warped_image`；
3. 检测 marker 数量；
4. warped image 尺寸；
5. 由校准得到的 `mm_per_pixel`。

## 手动校准点修正

当自动校准失败时，正式页面提供最小可用的手动校准入口：

1. 查看原始图；
2. 在表格中输入或调整板面四角点；
3. 点击“根据手动角点重新计算校准”；
4. 系统根据四角点计算 homography；
5. 生成 `warped_image` 和 `mm_per_pixel`；
6. 成功后才允许进入 Step 3。

## 坐标一致性检查

正式自动测量前会运行 `check_formal_coordinate_consistency()`，检查：

- 校准状态是否成功；
- 是否存在 `warped_image`；
- 是否存在由校准产生的 `mm_per_pixel`；
- 是否误用 fallback 默认比例尺；
- formal measurement points 是否在 warped image 边界内。

如检查失败，系统不会运行自动测量。

## 历史数据保护

该校准与坐标一致性逻辑不会修改：

- Review Queue；
- Advanced overlay；
- 历史 `corrected_keypoints`；
- `batch_measurements.csv`；
- `final_analysis_dataset.csv`。
