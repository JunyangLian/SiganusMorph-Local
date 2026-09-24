# External Landmark Model Benchmark

This benchmark path is for comparing SiganusMorph YOLO-pose enhanced preannotation
with heatmap-style landmark models such as DeepLabCut, SLEAP, and U-Net heatmaps.

The exports are prepared from corrected manual labels:

```text
results/real_annotation_5_15_v0.3_corrected/keypoints/
```

Prepared datasets:

```text
datasets/siganusmorph_dlc_v0.1/
datasets/siganusmorph_sleap_v0.1/
datasets/siganusmorph_heatmap_unet_v0.1/
```

The split is inherited from:

```text
datasets/siganusmorph_real_5_15_yolopose_maskcrop_v0.3_corrected/split_manifest.csv
```

## Evaluate Predictions

Use `scripts/evaluate_landmark_predictions.py` after exporting predictions from an
external model.

Example for crop-coordinate predictions:

```powershell
python scripts\evaluate_landmark_predictions.py `
  --predictions path\to\predictions.json `
  --model-name deeplabcut_resnet50_v0.1 `
  --input-type maskcrop `
  --coordinate-space crop
```

Example for warped-image-coordinate predictions:

```powershell
python scripts\evaluate_landmark_predictions.py `
  --predictions path\to\predictions.csv `
  --model-name sleap_unet_v0.1 `
  --input-type warped `
  --coordinate-space warped
```

Outputs are written to:

```text
results/model_eval/external_benchmark/
```

The evaluator reports point-wise and image-wise error, plus comparison columns for
the current bottleneck landmarks P2, P3, P5, P6, P7U, and P7L.
