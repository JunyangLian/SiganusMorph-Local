# 数据结构

## 输入数据

```text
data/real_images_raw_png/
data/real_images_warped/
data/real_images_manifest.csv
```

真实训练和测量主要基于 warped image。

## 标注结果

原始人工标注：

```text
results/real_annotation_5_15/
```

迁移后的 corrected labels：

```text
results/real_annotation_5_15_v0.3_corrected/keypoints/
```

real-world review corrected labels：

```text
results/realworld_review_v0.4.1/corrected_keypoints/
```

## corrected JSON 主要字段

常见字段：

```text
annotation_status
annotation_mode
image_name
specimen_id
corrected_keypoints
model_keypoints_raw
heatmap_keypoints
v034_keypoints
v06_keypoints
mask_suggestions
geometric_suggestions
derived_points
measurement_axes
keypoint_edit_log
qc_results
```

注意：

- `corrected_keypoints` 是最终标签。
- `model_keypoints_raw`、`heatmap_keypoints`、`mask_suggestions` 只用于参考和对比。
- P7V 在 `derived_points` 中，不进入 16 点训练标签。

## 测量结果

稳定批量测量：

```text
results/batch_measurement_v0.6.4_stable/batch_measurements.csv
results/batch_measurement_v0.6.4_stable/batch_measurements.xlsx
```

正式分析数据集：

```text
results/batch_measurement_v0.6.4_stable/final_analysis_dataset.csv
results/batch_measurement_v0.6.4_stable/final_analysis_dataset.xlsx
```

## 数据集目录

heatmap-U-Net 数据集：

```text
datasets/siganusmorph_heatmap_unet_v0.1/
datasets/siganusmorph_heatmap_unet_v0.5_candidate/
```

YOLO-pose 历史数据集：

```text
datasets/siganusmorph_real_5_15_yolopose_maskcrop_v0.3_corrected/
```

