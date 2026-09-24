# 下一步计划

## 1. 人工测量一致性验证

建议选取 representative subset，由人工重复测量或与传统测量结果对比。

统计指标：

- MAE
- MAPE
- ICC
- Bland-Altman

## 2. outlier 二次复核

当前 14 张 outlier preview 未见明显点位错误，暂时保留。后续应结合：

- 原始照片
- 同一 specimen 其他视角
- 生物学比例范围

判断是否为真实极端值。

## 3. 方法图整理

建议绘制：

- 图像校正流程图
- keypoint schema 图
- v0.6 hybrid selector 流程图
- dual-axis measurement 示意图

## 4. curved-fish mode

当前 v0.6.4 主要适合直鱼和轻微弯曲鱼。如后续需要处理强弯曲样本，应开发 curved-fish axis/mask 模式。

## 5. 训练目标调整

可继续评估是否将 C1-C4 从未来模型训练目标中移除，改为几何派生或只保留为人工复核点。

