"""Generate synthetic fish-on-board images for SiganusMorph workflow tests.

The images are intended for UI and export-flow testing only. They are kept in
``data/synthetic_test_images`` and marked as ``source_type=synthetic`` in the
manifest so they can be filtered away from real training data.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
BOARD_PATH = ROOT / "data" / "calibration_board" / "v2_charuco_plumb" / "siganusmorph_a3_v2_charuco_plumb_300dpi.png"
FISH_SOURCE_PATH = ROOT / "assets" / "templates" / "template_fish_source.png"
OUT_DIR = ROOT / "data" / "synthetic_test_images"
MANIFEST_PATH = OUT_DIR / "synthetic_manifest.csv"

BOARD_W_MM = 420.0
BOARD_H_MM = 297.0
FISH_BOX_MM = (45.0, 86.0, 375.0, 223.0)


@dataclass(frozen=True)
class Scenario:
    index: int
    scenario_type: str
    description: str
    difficulty: str
    notes: str
    center_mm: tuple[float, float] = (210.0, 154.5)
    length_mm: float = 205.0
    rotation_deg: float = 0.0
    curve_kind: str = "straight"
    curve_amplitude_mm: float = 0.0
    tail_variant: str = "normal"
    vertical_scale: float = 1.0


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(1, "straight_standard", "标准直鱼居中放置，尾鳍展开良好", "easy", "用于测试正常标注流程"),
    Scenario(2, "straight_standard", "鱼体略偏上，整体保持笔直", "easy", "用于测试正常标注流程", center_mm=(210.0, 145.0)),
    Scenario(3, "straight_standard", "鱼体略偏下，整体保持笔直", "easy", "用于测试正常标注流程", center_mm=(210.0, 164.0)),
    Scenario(4, "straight_standard", "鱼体略偏左，整体保持笔直", "easy", "用于测试正常标注流程", center_mm=(196.0, 154.5)),
    Scenario(5, "straight_standard", "鱼体略偏右，整体保持笔直", "easy", "用于测试正常标注流程", center_mm=(224.0, 154.5)),
    Scenario(6, "straight_standard", "鱼体稍微大一点，尾鳍展开良好", "easy", "用于测试正常标注流程", length_mm=228.0),
    Scenario(7, "mildly_curved", "鱼体轻微向上弯曲", "medium", "用于测试中轴线曲线拟合", curve_kind="up", curve_amplitude_mm=6.5),
    Scenario(8, "mildly_curved", "鱼体轻微向下弯曲", "medium", "用于测试中轴线曲线拟合", curve_kind="down", curve_amplitude_mm=6.5),
    Scenario(9, "mildly_curved", "尾柄略微上翘", "medium", "用于测试中轴线曲线拟合", curve_kind="tail_up", curve_amplitude_mm=8.0),
    Scenario(10, "mildly_curved", "尾柄略微下弯", "medium", "用于测试中轴线曲线拟合", curve_kind="tail_down", curve_amplitude_mm=8.0),
    Scenario(11, "mildly_curved", "头部与躯干轻微不在一条直线上", "medium", "用于测试中轴线曲线拟合", curve_kind="head_offset", curve_amplitude_mm=6.0),
    Scenario(12, "mildly_curved", "整体形成很轻微 S 形", "medium", "用于测试中轴线曲线拟合", curve_kind="s", curve_amplitude_mm=6.5),
    Scenario(13, "caudal_fin_variant", "上尾叶略长", "medium", "用于测试P7U/P7L/P7V逻辑", tail_variant="upper_long"),
    Scenario(14, "caudal_fin_variant", "下尾叶略长", "medium", "用于测试P7U/P7L/P7V逻辑", tail_variant="lower_long"),
    Scenario(15, "caudal_fin_variant", "尾鳍略微收拢", "medium", "用于测试P7U/P7L/P7V逻辑", tail_variant="closed"),
    Scenario(16, "caudal_fin_variant", "尾鳍展开较宽，分叉点明显", "medium", "用于测试P7U/P7L/P7V逻辑", tail_variant="wide"),
    Scenario(17, "position_pose_variant", "鱼体整体略微顺时针旋转", "medium", "用于测试姿态变化鲁棒性", rotation_deg=-4.0, center_mm=(210.0, 156.0)),
    Scenario(18, "position_pose_variant", "鱼体整体略微逆时针旋转", "medium", "用于测试姿态变化鲁棒性", rotation_deg=4.0, center_mm=(210.0, 154.0)),
    Scenario(19, "position_pose_variant", "鱼体偏左且略小", "medium", "用于测试姿态变化鲁棒性", center_mm=(190.0, 154.0), length_mm=180.0),
    Scenario(20, "position_pose_variant", "鱼体偏右且略大", "medium", "用于测试姿态变化鲁棒性", center_mm=(228.0, 154.0), length_mm=222.0),
)


def px_per_mm(board: Image.Image) -> tuple[float, float]:
    return board.width / BOARD_W_MM, board.height / BOARD_H_MM


def mm_to_px(x_mm: float, y_mm: float, board: Image.Image) -> tuple[int, int]:
    sx, sy = px_per_mm(board)
    return int(round(x_mm * sx)), int(round(y_mm * sy))


def fish_box_px(board: Image.Image) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = FISH_BOX_MM
    left, top = mm_to_px(x0, y0, board)
    right, bottom = mm_to_px(x1, y1, board)
    return left, top, right, bottom


def extract_fish_cutout(source_path: Path) -> Image.Image:
    source = Image.open(source_path).convert("RGB")
    rgb = np.array(source)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # Blue/cyan board background. The largest non-blue component is the fish.
    blue_background = (h >= 82) & (h <= 112) & (s >= 55) & (v >= 70)
    foreground = (~blue_background).astype(np.uint8) * 255
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))

    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError(f"Could not isolate fish from {source_path}")

    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros(foreground.shape, dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    mask = cv2.GaussianBlur(mask, (17, 17), 0)

    ys, xs = np.where(mask > 8)
    left = max(0, int(xs.min()) - 24)
    right = min(mask.shape[1], int(xs.max()) + 25)
    top = max(0, int(ys.min()) - 24)
    bottom = min(mask.shape[0], int(ys.max()) + 25)

    rgba = np.dstack([rgb, mask])
    return Image.fromarray(rgba, "RGBA").crop((left, top, right, bottom))


def transparent_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    alpha = np.array(image.getchannel("A"))
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0:
        return 0, 0, image.width, image.height
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop_transparent(image: Image.Image, margin: int = 16) -> Image.Image:
    left, top, right, bottom = transparent_bbox(image)
    return image.crop(
        (
            max(0, left - margin),
            max(0, top - margin),
            min(image.width, right + margin),
            min(image.height, bottom + margin),
        )
    )


def resize_to_length(cutout: Image.Image, length_mm: float, board: Image.Image, vertical_scale: float) -> Image.Image:
    sx, _ = px_per_mm(board)
    target_width = int(round(length_mm * sx))
    scale = target_width / cutout.width
    target_height = int(round(cutout.height * scale * vertical_scale))
    return cutout.resize((target_width, target_height), Image.Resampling.LANCZOS)


def remap_rgba(image: Image.Image, map_x: np.ndarray, map_y: np.ndarray) -> Image.Image:
    arr = np.array(image)
    remapped = cv2.remap(
        arr,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return Image.fromarray(remapped, "RGBA")


def curve_warp(image: Image.Image, kind: str, amplitude_px: float) -> Image.Image:
    if kind == "straight" or abs(amplitude_px) < 1:
        return image

    pad_y = max(80, int(abs(amplitude_px) * 2.4))
    padded = Image.new("RGBA", (image.width, image.height + pad_y * 2), (0, 0, 0, 0))
    padded.alpha_composite(image, (0, pad_y))
    width, height = padded.size
    x_grid, y_grid = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    t = x_grid / max(1, width - 1)

    if kind == "up":
        offset = -amplitude_px * np.sin(np.pi * t)
    elif kind == "down":
        offset = amplitude_px * np.sin(np.pi * t)
    elif kind == "tail_up":
        ramp = np.clip((t - 0.62) / 0.38, 0, 1)
        offset = -amplitude_px * (ramp * ramp * (3 - 2 * ramp))
    elif kind == "tail_down":
        ramp = np.clip((t - 0.62) / 0.38, 0, 1)
        offset = amplitude_px * (ramp * ramp * (3 - 2 * ramp))
    elif kind == "head_offset":
        ramp = np.clip((0.40 - t) / 0.40, 0, 1)
        offset = -amplitude_px * (ramp * ramp * (3 - 2 * ramp))
    elif kind == "s":
        offset = amplitude_px * 0.75 * np.sin(2 * np.pi * (t - 0.08))
    else:
        offset = np.zeros_like(t)

    return crop_transparent(remap_rgba(padded, x_grid, y_grid - offset), margin=20)


def tail_variant_warp(image: Image.Image, variant: str) -> Image.Image:
    if variant == "normal":
        return image

    pad_x = 100
    pad_y = 80
    padded = Image.new("RGBA", (image.width + pad_x * 2, image.height + pad_y * 2), (0, 0, 0, 0))
    padded.alpha_composite(image, (pad_x, pad_y))
    width, height = padded.size
    x_grid, y_grid = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    t = x_grid / max(1, width - 1)
    mid = height * 0.50
    tail = np.clip((t - 0.72) / 0.28, 0, 1)
    tail = tail * tail * (3 - 2 * tail)

    map_x = x_grid.copy()
    map_y = y_grid.copy()
    upper = (y_grid < mid).astype(np.float32)
    lower = 1.0 - upper

    if variant == "upper_long":
        shift = 70.0 * tail * upper + 15.0 * tail * lower
        map_x = x_grid - shift
    elif variant == "lower_long":
        shift = 15.0 * tail * upper + 70.0 * tail * lower
        map_x = x_grid - shift
    elif variant == "closed":
        scale = 1.0 - 0.28 * tail
        map_y = mid + (y_grid - mid) / np.maximum(scale, 0.4)
    elif variant == "wide":
        scale = 1.0 + 0.22 * tail
        map_y = mid + (y_grid - mid) / scale

    return crop_transparent(remap_rgba(padded, map_x, map_y), margin=20)


def rotate_rgba(image: Image.Image, angle_deg: float) -> Image.Image:
    if abs(angle_deg) < 0.1:
        return image
    return crop_transparent(
        image.rotate(angle_deg, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(0, 0, 0, 0)),
        margin=20,
    )


def draw_shadow(base: Image.Image, fish: Image.Image, paste_xy: tuple[int, int]) -> None:
    alpha = fish.getchannel("A").filter(ImageFilter.GaussianBlur(radius=24))
    shadow = Image.new("RGBA", fish.size, (0, 0, 0, 68))
    shadow.putalpha(alpha.point(lambda value: int(value * 0.22)))
    base.alpha_composite(shadow, (paste_xy[0] + 18, paste_xy[1] + 24))


def compose(board: Image.Image, fish: Image.Image, scenario: Scenario) -> tuple[Image.Image, tuple[int, int, int, int]]:
    base = board.convert("RGBA")
    center_x, center_y = mm_to_px(*scenario.center_mm, board)
    paste_x = int(round(center_x - fish.width / 2))
    paste_y = int(round(center_y - fish.height / 2))

    box_left, box_top, box_right, box_bottom = fish_box_px(board)
    # Keep all synthetic fish safely inside the clean fish placement box.
    paste_x = min(max(paste_x, box_left + 20), box_right - fish.width - 20)
    paste_y = min(max(paste_y, box_top + 20), box_bottom - fish.height - 20)

    draw_shadow(base, fish, (paste_x, paste_y))
    base.alpha_composite(fish, (paste_x, paste_y))
    bbox = transparent_bbox(fish)
    pasted_bbox = (
        paste_x + bbox[0],
        paste_y + bbox[1],
        paste_x + bbox[2],
        paste_y + bbox[3],
    )
    return base.convert("RGB"), pasted_bbox


def build_synthetic_image(board: Image.Image, cutout: Image.Image, scenario: Scenario) -> tuple[Image.Image, tuple[int, int, int, int]]:
    sx, _ = px_per_mm(board)
    fish = resize_to_length(cutout, scenario.length_mm, board, scenario.vertical_scale)
    fish = curve_warp(fish, scenario.curve_kind, scenario.curve_amplitude_mm * sx)
    fish = tail_variant_warp(fish, scenario.tail_variant)
    fish = rotate_rgba(fish, scenario.rotation_deg)
    return compose(board, fish, scenario)


def check_generated_image(path: Path, board: Image.Image, pasted_bbox: tuple[int, int, int, int]) -> None:
    with Image.open(path) as image:
        if image.size != board.size:
            raise RuntimeError(f"{path.name}: image size {image.size} does not match board size {board.size}")

    box_left, box_top, box_right, box_bottom = fish_box_px(board)
    left, top, right, bottom = pasted_bbox
    if left < box_left or top < box_top or right > box_right or bottom > box_bottom:
        raise RuntimeError(f"{path.name}: fish bbox {pasted_bbox} exceeds fish placement box")
    if right <= left or bottom <= top:
        raise RuntimeError(f"{path.name}: invalid fish bbox {pasted_bbox}")


def write_manifest(rows: list[dict[str, str]]) -> None:
    fields = ["image_name", "source_type", "scenario_type", "description", "expected_difficulty", "notes"]
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not BOARD_PATH.exists():
        raise FileNotFoundError(f"Board image not found: {BOARD_PATH}")
    if not FISH_SOURCE_PATH.exists():
        raise FileNotFoundError(f"Fish source image not found: {FISH_SOURCE_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    board = Image.open(BOARD_PATH).convert("RGB")
    cutout = extract_fish_cutout(FISH_SOURCE_PATH)
    rows: list[dict[str, str]] = []
    counts: dict[str, int] = {}

    for scenario in SCENARIOS:
        image_name = f"synthetic_{scenario.index:03d}.png"
        image, bbox = build_synthetic_image(board, cutout, scenario)
        output_path = OUT_DIR / image_name
        image.save(output_path)
        check_generated_image(output_path, board, bbox)
        rows.append(
            {
                "image_name": image_name,
                "source_type": "synthetic",
                "scenario_type": scenario.scenario_type,
                "description": scenario.description,
                "expected_difficulty": scenario.difficulty,
                "notes": scenario.notes,
            }
        )
        counts[scenario.scenario_type] = counts.get(scenario.scenario_type, 0) + 1

    write_manifest(rows)
    if len(rows) != 20:
        raise RuntimeError(f"Expected 20 synthetic images, generated {len(rows)}")

    print("Generated 20 synthetic fish test images.")
    print("Output directory: data/synthetic_test_images/")
    print("Manifest: data/synthetic_test_images/synthetic_manifest.csv")
    for name in ("straight_standard", "mildly_curved", "caudal_fin_variant", "position_pose_variant"):
        print(f"{name}: {counts.get(name, 0)}")


if __name__ == "__main__":
    main()
