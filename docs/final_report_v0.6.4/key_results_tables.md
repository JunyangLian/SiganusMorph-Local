# v0.6.4 核心结果表

## real-world review

| 项目 | 数值 |
| --- | --- |
| review sample | 30 |
| confirmed | 29 |
| excluded | 1 |
| excluded case | real_037.png / fish_18 / failed_preannotation |
| mean review time | ~85.5 sec/image |
| mean moved points | 4.28 / 16 |

## v0.6 same-set comparison

| 模型/流程 | 结论 |
| --- | --- |
| v0.1 heatmap | median 低，但 mean error 受 outlier 影响 |
| v0.5 heatmap | hard cases 改善明显 |
| v0.4.1 automatic | 稳定基线 |
| v0.6 selector | mean error 和 large-error rate 改善 |

## dual-axis comparison

| 指标 | 结果 |
| --- | --- |
| model_axis smoothness | 18.881 |
| body_midline_axis smoothness | 4.297 |
| TL_curve diff mean / median / max | 1.042 / 0.578 / 4.940 mm |
| dual-axis disagreement | 0 |
| P7V valid rate | 1.000 |

## stable batch measurement

| 项目 | 数值 |
| --- | --- |
| included images | 96 |
| P7V valid | 96 / 96 |
| outlier cases | 14 |
| final analysis rows | 96 |

