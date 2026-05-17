# 关键点定义

当前版本使用固定顺序人工点击 16 个关键点。P1-P11、P7U、P7L 是核心测量点，C1-C4 是主中轴线辅助点。程序会根据 P7U/P7L 自动生成虚拟全长终点 P7V。

## 点击顺序

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

## 尾鳍虚拟终点

P7V `caudal_fin_posterior_endpoint` 是程序自动计算的虚拟点，不需要用户手动点击。计算步骤：

```text
tail_axis = unit_vector(P5 caudal_base_midpoint → P6 caudal_fork_midpoint)
```

如果 P5 和 P6 重合或 P6 无效，则回退为：

```text
tail_axis = unit_vector(C4 peduncle_axis_point → P5 caudal_base_midpoint)
```

然后以 P5 为参考点，比较：

```text
proj_upper = dot(P7U - P5, tail_axis)
proj_lower = dot(P7L - P5, tail_axis)
```

投影更大的尾叶被记录到 `caudal_tip_selected`，并生成：

```text
P7V = P5 + max(proj_upper, proj_lower) * tail_axis
```

## 核心测量点

### P1 `snout_tip`：吻端

鱼吻部最前端，也就是鱼头最左侧点。用于全长、标准长、头长、吻长的起点，也是主中轴线起点。

### P2 `eye_front`：眼眶前缘

眼眶最靠近吻端的一侧边缘，也就是眼眶最左侧点。用于吻长终点。

### P3 `operculum_posterior`：鳃盖骨后缘

鳃盖骨后缘最明显的位置，也就是鱼头盖骨最右侧边界。用于头长终点。

### P4 `peduncle_start_midpoint`：尾柄前端中轴点

尾柄开始处在鱼体中轴线上的点，不是鱼体外轮廓点。用于尾柄长起点，也是主中轴线中的一个节点。

### P5 `caudal_base_midpoint`：尾鳍基部中点

鱼身和尾鳍交界处的中点，位于中轴线上。用于标准长终点、尾柄长终点、主中轴线终点，也是尾鳍轴向参考起点。

### P6 `caudal_fork_midpoint`：尾鳍分叉点

尾鳍上下叶分叉位置的中间点。用于辅助确定尾部轴向方向，不再作为全长路径的必经点。

### P7U `caudal_fin_upper_tip`：尾鳍上叶末端

尾鳍上叶最远离吻端的末端点。该点不一定直接用于全长，程序会根据尾部轴向方向判断它是否是轴向最远端。

### P7L `caudal_fin_lower_tip`：尾鳍下叶末端

尾鳍下叶最远离吻端的末端点。程序会比较上叶和下叶在尾部轴向上的投影距离，并自动生成全长终点 P7V。

### P8 `body_depth_dorsal`：最大体高上点

躯干部最大体高位置的背侧边界点，不包括背鳍。P8 和 P9 应尽量位于垂直于局部中轴线方向的最大体高两端。

### P9 `body_depth_ventral`：最大体高下点

与 P8 对应的腹侧边界点，不包括腹鳍、臀鳍。

### P10 `peduncle_depth_dorsal`：尾柄最窄处上点

尾柄最窄位置的背侧边界点，不包括尾鳍。

### P11 `peduncle_depth_ventral`：尾柄最窄处下点

与 P10 对应的尾柄腹侧边界点，不包括尾鳍。P10 和 P11 应尽量位于垂直于尾柄局部中轴线方向的最窄处两端。

## 中轴线辅助点 C1-C4

C1 `head_axis_point`：头盖骨/鳃盖附近的中轴点，位于头后部上下边界之间的中心位置。

C2 `trunk_axis_point`：鱼体躯干中部的中心点。

C3 `posterior_trunk_axis_point`：躯干后部、接近尾柄前方的中心点。

C4 `peduncle_axis_point`：尾柄中部的中心点，位于 P4 和 P5 之间。

## 轴线顺序

主中轴线 `body_axis_points_order`：

```text
P1 snout_tip
→ C1 head_axis_point
→ C2 trunk_axis_point
→ C3 posterior_trunk_axis_point
→ P4 peduncle_start_midpoint
→ C4 peduncle_axis_point
→ P5 caudal_base_midpoint
```

尾鳍末端候选点 `caudal_tip_candidates`：

```text
P7U caudal_fin_upper_tip
P7L caudal_fin_lower_tip
```

## 当前测量

```text
TL_straight_axis = distance(P1, P7V)
TL_curve = curve_length(P1-C1-C2-C3-P4-C4-P5) + caudal_extension_axis

SL_straight = distance(P1, P5)
SL_curve = curve_length(P1-C1-C2-C3-P4-C4-P5)

head_length_straight = distance(P1, P3)
snout_length_straight = distance(P1, P2)

caudal_peduncle_length_straight = distance(P4, P5)
caudal_peduncle_length_axis = curve_length(P4-C4-P5)

body_depth = distance(P8, P9)
caudal_peduncle_depth = distance(P10, P11)
```
