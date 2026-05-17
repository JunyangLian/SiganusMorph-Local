"""Shared configuration for SiganusMorph Local."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KeypointDefinition:
    code: str
    name: str
    label_cn: str
    description: str
    common_errors: str = ""


KEYPOINT_DEFS: tuple[KeypointDefinition, ...] = (
    KeypointDefinition("P1", "snout_tip", "吻端", "请点击鱼吻部最前端，也就是鱼头最左侧点。该点也是主中轴线起点。", "不要点到张开的嘴唇阴影，也不要点到鼻孔或额部轮廓。"),
    KeypointDefinition("P2", "eye_front", "眼眶前缘", "请点击眼眶最靠近吻端的一侧边缘，也就是眼眶最左侧点。", "不要点到瞳孔中心、眼球反光点或眼眶后缘。"),
    KeypointDefinition("P3", "operculum_posterior", "鳃盖骨后缘", "请点击鳃盖骨后缘最明显的位置，也就是鱼头盖骨最右侧边界。", "不要点到眼后缘，不要点到鳃盖内侧阴影。"),
    KeypointDefinition("P4", "peduncle_start_midpoint", "尾柄前端中轴点", "请点击尾柄开始处在鱼体中轴线上的点。该点不是鱼体外轮廓点，而是尾柄起点处的中线位置。", "不要点到背侧或腹侧外轮廓，也不要点到臀鳍基部后端。"),
    KeypointDefinition("P5", "caudal_base_midpoint", "尾鳍基部中点", "请点击鱼身与尾鳍交界处的中点。该点是标准长终点，也是尾鳍延长线起点。", "不要点到尾鳍叶尖，也不要选尾柄上缘或下缘。"),
    KeypointDefinition("P6", "caudal_fork_midpoint", "尾鳍分叉点", "请点击尾鳍上下叶分叉位置的中间点，用于辅助确定尾鳍延长方向。", "不要点到尾鳍叶尖或尾鳍基部，应点分叉凹陷附近的中间位置。"),
    KeypointDefinition("P7U", "caudal_fin_upper_tip", "尾鳍上叶末端", "请点击尾鳍上叶最远离吻端的末端点。该点不一定直接用于全长，程序会根据中轴线方向判断它是否是轴向最远端。", "不要点到尾鳍分叉凹陷或上叶边缘中段，应点上叶可见最远端。"),
    KeypointDefinition("P7L", "caudal_fin_lower_tip", "尾鳍下叶末端", "请点击尾鳍下叶最远离吻端的末端点。程序会比较上叶和下叶在尾部轴向上的投影距离，并自动生成全长终点 P7V。", "不要点到尾鳍分叉凹陷或下叶边缘中段，应点下叶可见最远端。"),
    KeypointDefinition("P8", "body_depth_dorsal", "最大体高上点", "请点击鱼体躯干部最大体高位置的背侧边界点，不包含背鳍。", "不要包含背鳍高度，也不要点到背鳍鳍条尖端。"),
    KeypointDefinition("P9", "body_depth_ventral", "最大体高下点", "请点击与 P8 对应的腹侧边界点，不包含腹鳍、臀鳍。", "不要包含腹鳍或臀鳍，P8-P9 应尽量垂直于局部中轴线。"),
    KeypointDefinition("P10", "peduncle_depth_dorsal", "尾柄最窄处上点", "请点击尾柄最窄位置的背侧边界点，不包含尾鳍。", "不要点到尾鳍基部展开处，应选尾柄最窄截面的上边界。"),
    KeypointDefinition("P11", "peduncle_depth_ventral", "尾柄最窄处下点", "请点击与 P10 对应的尾柄腹侧边界点，不包含尾鳍。", "不要点到尾鳍或臀鳍残端，P10-P11 应尽量垂直于尾柄局部中轴线。"),
    KeypointDefinition("C1", "head_axis_point", "头后部中轴点", "请点击头盖骨/鳃盖附近的中轴点，位于头后部上下边界之间的中心位置。", "不要贴近背侧或腹侧轮廓，应位于鱼体厚度中心。"),
    KeypointDefinition("C2", "trunk_axis_point", "躯干中部中轴点", "请点击鱼体躯干中部、位于背腹边界之间的中心点，用于拟合中轴线。", "不要跟随体表斑纹或阴影，应按背腹边界中间估计。"),
    KeypointDefinition("C3", "posterior_trunk_axis_point", "躯干后部中轴点", "请点击躯干后部、接近尾柄前方的中心点，用于拟合中轴线。", "不要点到背鳍或臀鳍基部，应落在鱼体中心线上。"),
    KeypointDefinition("C4", "peduncle_axis_point", "尾柄中轴点", "请点击尾柄中部的中心点，位于 P4 和 P5 之间。", "不要偏向尾柄上缘或下缘，应与 P10/P11 的中间方向一致。"),
)

KEYPOINT_NAMES = tuple(kp.name for kp in KEYPOINT_DEFS)
KEYPOINT_BY_NAME = {kp.name: kp for kp in KEYPOINT_DEFS}
KEYPOINT_BY_CODE = {kp.code: kp for kp in KEYPOINT_DEFS}
CORE_KEYPOINT_NAMES = tuple(kp.name for kp in KEYPOINT_DEFS if kp.code.startswith("P"))
AXIS_AUXILIARY_KEYPOINT_NAMES = tuple(kp.name for kp in KEYPOINT_DEFS if kp.code.startswith("C"))
BODY_AXIS_POINT_ORDER = (
    "snout_tip",
    "head_axis_point",
    "trunk_axis_point",
    "posterior_trunk_axis_point",
    "peduncle_start_midpoint",
    "peduncle_axis_point",
    "caudal_base_midpoint",
)
CAUDAL_FIN_AXIS_POINT_ORDER = (
    "caudal_base_midpoint",
    "caudal_fork_midpoint",
)
CAUDAL_TIP_CANDIDATE_NAMES = (
    "caudal_fin_upper_tip",
    "caudal_fin_lower_tip",
)
DERIVED_POINT_DEFS = (
    KeypointDefinition("P7V", "caudal_fin_posterior_endpoint", "尾鳍轴向最远端虚拟点", "程序根据 P7U 和 P7L 在尾部轴向上的投影自动生成的全长终点。", ""),
)
AXIS_POINT_ORDER = BODY_AXIS_POINT_ORDER
BODY_AXIS_POINT_KEYS = tuple(f"{KEYPOINT_BY_NAME[name].code}_{name}" for name in BODY_AXIS_POINT_ORDER)
CAUDAL_FIN_AXIS_POINT_KEYS = tuple(f"{KEYPOINT_BY_NAME[name].code}_{name}" for name in CAUDAL_FIN_AXIS_POINT_ORDER)
CAUDAL_TIP_CANDIDATE_KEYS = tuple(f"{KEYPOINT_BY_NAME[name].code}_{name}" for name in CAUDAL_TIP_CANDIDATE_NAMES)
DERIVED_POINT_KEYS = tuple(f"{definition.code}_{definition.name}" for definition in DERIVED_POINT_DEFS)
AXIS_POINT_KEYS = BODY_AXIS_POINT_KEYS

MEASUREMENT_DEFS = (
    ("SL_straight_mm", "SL straight", "snout_tip", "caudal_base_midpoint"),
    ("body_depth_mm", "Body depth", "body_depth_dorsal", "body_depth_ventral"),
    ("head_length_straight_mm", "Head length", "snout_tip", "operculum_posterior"),
    ("snout_length_straight_mm", "Snout length", "snout_tip", "eye_front"),
    ("caudal_peduncle_length_straight_mm", "Peduncle length", "peduncle_start_midpoint", "caudal_base_midpoint"),
    ("caudal_peduncle_depth_mm", "Peduncle depth", "peduncle_depth_dorsal", "peduncle_depth_ventral"),
)

RESULT_COLUMNS = (
    "image_name",
    "specimen_id",
    "source_type",
    "axis_mode_selected",
    "needs_review",
    "notes",
    "scale_method",
    "mm_per_pixel",
    *tuple(f"{definition.code}_{definition.name}_{axis}" for definition in KEYPOINT_DEFS for axis in ("x", "y")),
    *tuple(f"{definition.code}_{definition.name}_{axis}" for definition in DERIVED_POINT_DEFS for axis in ("x", "y")),
    "caudal_tip_selected",
    "caudal_extension_axis_mm",
    "TL_straight_axis_mm",
    "TL_axis_polyline_mm",
    "TL_axis_spline_mm",
    "TL_final_mm",
    "SL_straight_mm",
    "SL_axis_polyline_mm",
    "SL_axis_spline_mm",
    "SL_final_mm",
    "body_depth_mm",
    "body_depth_axis_corrected_mm",
    "head_length_straight_mm",
    "head_length_axis_mm",
    "snout_length_straight_mm",
    "snout_length_axis_mm",
    "caudal_peduncle_length_straight_mm",
    "caudal_peduncle_length_axis_mm",
    "caudal_peduncle_depth_mm",
    "caudal_peduncle_depth_axis_corrected_mm",
    "SL_curve_mm",
    "TL_curve_mm",
    "curvature_index",
    "curvature_index_polyline",
    "curvature_index_spline",
)

SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

DEFAULT_BOARD_CONFIG = {
    "name": "a3_v1_four_corner_aruco",
    "board_width_mm": 420.0,
    "board_height_mm": 297.0,
    "output_px_per_mm": 10.0,
    "dictionary_name": "DICT_4X4_50",
    "aruco_corner_ids": {
        "top_left": 0,
        "top_right": 1,
        "bottom_left": 2,
        "bottom_right": 3,
    },
}

A3_V2_CHARUCO_PLUMB_CONFIG = {
    "name": "a3_v2_charuco_plumb",
    "board_width_mm": 420.0,
    "board_height_mm": 297.0,
    "output_px_per_mm": 10.0,
    "dictionary_name": "DICT_4X4_50",
    "aruco_corner_ids": {
        "top_left": 0,
        "top_right": 1,
        "bottom_left": 2,
        "bottom_right": 3,
    },
    "aruco_midpoint_ids": {
        "top_mid": 4,
        "right_mid": 5,
        "bottom_mid": 6,
        "left_mid": 7,
    },
    # Coordinates are the black ArUco marker square, not the white quiet zone panel.
    # The order of corners is top-left, top-right, bottom-right, bottom-left.
    "location_marker_squares_mm": {
        0: {"x": 11.0, "y": 11.0, "size": 16.0},
        1: {"x": 393.0, "y": 11.0, "size": 16.0},
        2: {"x": 11.0, "y": 270.0, "size": 16.0},
        3: {"x": 393.0, "y": 270.0, "size": 16.0},
        4: {"x": 202.0, "y": 11.0, "size": 16.0},
        5: {"x": 393.0, "y": 140.5, "size": 16.0},
        6: {"x": 202.0, "y": 270.0, "size": 16.0},
        7: {"x": 11.0, "y": 140.5, "size": 16.0},
    },
    "charuco": {
        "dictionary_name": "DICT_6X6_250",
        "squares_x": 18,
        "squares_y": 4,
        "square_length_mm": 7.0,
        "marker_length_mm": 5.0,
        "origin_x_mm": 60.0,
        "origin_y_mm": 247.0,
    },
    "ruler": {
        "start_x_mm": 35.0,
        "y_mm": 50.0,
        "length_mm": 350.0,
        "tick_count": 351,
    },
    "plumb_lines": {
        "outer_frame_mm": [6.0, 6.0, 414.0, 291.0],
        "fish_box_mm": [45.0, 86.0, 375.0, 223.0],
        "top_ruler_baseline_y_mm": 72.0,
        "bottom_reference_y_mm": 236.0,
        "left_reference_x_mm": 45.0,
        "right_reference_x_mm": 375.0,
    },
}

BOARD_CONFIGS = {
    DEFAULT_BOARD_CONFIG["name"]: DEFAULT_BOARD_CONFIG,
    A3_V2_CHARUCO_PLUMB_CONFIG["name"]: A3_V2_CHARUCO_PLUMB_CONFIG,
}
