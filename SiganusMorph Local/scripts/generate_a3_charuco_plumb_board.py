from __future__ import annotations

import base64
import json
import sys
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from siganusmorph.aruco_utils import ARUCO_DICTIONARIES
from siganusmorph.config import A3_V2_CHARUCO_PLUMB_CONFIG


DPI = 300
PX_PER_MM = DPI / 25.4
BOARD_W_MM = 420.0
BOARD_H_MM = 297.0
OUT_DIR = Path("data/calibration_board/v2_charuco_plumb")
STEM = "siganusmorph_a3_v2_charuco_plumb"

BLUE = (0, 150, 216)
DEEP_BLUE = (0, 109, 178)
WHITE = (245, 250, 252)
BLACK = (10, 15, 20)
LIGHT = (220, 242, 250)
GRID = (80, 105, 118)


def px(value_mm: float) -> int:
    return int(round(value_mm * PX_PER_MM))


def font(size_px: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/arial.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size_px)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_centered_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fnt, fill=WHITE) -> None:
    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=fnt)
    draw.text((x - (bbox[2] - bbox[0]) / 2, y - (bbox[3] - bbox[1]) / 2), text, font=fnt, fill=fill)


def marker_image(dictionary_name: str, marker_id: int, size_px: int) -> Image.Image:
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[dictionary_name])
    marker = cv2.aruco.generateImageMarker(dictionary, int(marker_id), int(size_px))
    return Image.fromarray(marker).convert("RGB")


def paste_location_marker(
    canvas_image: Image.Image,
    draw: ImageDraw.ImageDraw,
    marker_id: int,
    square: dict[str, float],
    label: str,
) -> None:
    marker_size = px(float(square["size"]))
    marker_x = px(float(square["x"]))
    marker_y = px(float(square["y"]))
    quiet = px(3.0)
    panel = [marker_x - quiet, marker_y - quiet, marker_x + marker_size + quiet, marker_y + marker_size + quiet]
    draw.rectangle(panel, fill=WHITE)
    canvas_image.paste(marker_image("DICT_4X4_50", marker_id, marker_size), (marker_x, marker_y))


def paste_charuco_panel(canvas_image: Image.Image, draw: ImageDraw.ImageDraw) -> None:
    cfg = A3_V2_CHARUCO_PLUMB_CONFIG["charuco"]
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[cfg["dictionary_name"]])
    board = cv2.aruco.CharucoBoard(
        (int(cfg["squares_x"]), int(cfg["squares_y"])),
        float(cfg["square_length_mm"]),
        float(cfg["marker_length_mm"]),
        dictionary,
    )
    width_px = px(float(cfg["squares_x"]) * float(cfg["square_length_mm"]))
    height_px = px(float(cfg["squares_y"]) * float(cfg["square_length_mm"]))
    panel = board.generateImage((width_px, height_px), marginSize=0, borderBits=1)
    panel_image = Image.fromarray(panel).convert("RGB")
    x = px(float(cfg["origin_x_mm"]))
    y = px(float(cfg["origin_y_mm"]))
    pad = px(2.0)
    draw.rectangle([x - pad, y - pad, x + width_px + pad, y + height_px + pad], fill=WHITE)
    canvas_image.paste(panel_image, (x, y))


def draw_ruler(draw: ImageDraw.ImageDraw) -> None:
    ruler = A3_V2_CHARUCO_PLUMB_CONFIG["ruler"]
    x0 = float(ruler["start_x_mm"])
    y0 = float(ruler["y_mm"])
    length = float(ruler["length_mm"])
    draw.rounded_rectangle(
        [px(x0 - 4), px(y0 - 9), px(x0 + length + 4), px(y0 + 27)],
        radius=px(2.2),
        fill=WHITE,
    )
    checker_h = 7.0
    for cm_index in range(35):
        x = x0 + cm_index * 10.0
        fill = BLACK if cm_index % 2 == 0 else WHITE
        draw.rectangle([px(x), px(y0 - 2), px(x + 10), px(y0 + checker_h)], fill=fill, outline=BLACK)
    number_font = font(px(2.5), bold=True)
    small_font = font(px(2.4))
    for cm_index in range(36):
        x = x0 + cm_index * 10.0
        tick_len = 13.5
        draw.line([px(x), px(y0 + checker_h), px(x), px(y0 + checker_h + tick_len)], fill=BLACK, width=max(1, px(0.28)))
        if cm_index % 5 == 0:
            draw_centered_text(draw, (px(x), px(y0 - 6.0)), str(cm_index), number_font, fill=BLACK)
        elif 0 < cm_index < 35:
            draw_centered_text(draw, (px(x), px(y0 - 6.0)), str(cm_index), small_font, fill=BLACK)
    for mm_index in range(351):
        x = x0 + mm_index
        if mm_index % 10 == 0:
            continue
        tick_len = 9.0 if mm_index % 5 == 0 else 5.5
        draw.line(
            [px(x), px(y0 + checker_h), px(x), px(y0 + checker_h + tick_len)],
            fill=BLACK,
            width=max(1, px(0.18)),
        )
    draw_centered_text(
        draw,
        (px(x0 + length / 2), px(y0 + 22)),
        "0-35 cm scale (1 mm ticks)",
        font(px(2.8), bold=True),
        fill=BLACK,
    )


def draw_sample_id_grid(draw: ImageDraw.ImageDraw) -> None:
    cell_w = 8.5
    cell_h = 6.4
    x0 = (BOARD_W_MM - cell_w * 20) / 2
    y0 = 228.0
    num_font = font(px(2.5), bold=True)
    for row in range(2):
        for col in range(20):
            n = row * 20 + col + 1
            x = x0 + col * cell_w
            y = y0 + row * cell_h
            draw.rectangle([px(x), px(y), px(x + cell_w), px(y + cell_h)], fill=WHITE, outline=GRID, width=max(1, px(0.18)))
            draw_centered_text(draw, (px(x + cell_w / 2), px(y + cell_h / 2)), str(n), num_font, fill=BLACK)


def draw_plumb_lines(draw: ImageDraw.ImageDraw) -> None:
    plumb = A3_V2_CHARUCO_PLUMB_CONFIG["plumb_lines"]
    outer = plumb["outer_frame_mm"]
    fish = plumb["fish_box_mm"]
    thin = max(1, px(0.18))
    medium = max(1, px(0.35))
    draw.rectangle([px(outer[0]), px(outer[1]), px(outer[2]), px(outer[3])], outline=WHITE, width=medium)
    draw.rectangle([px(fish[0]), px(fish[1]), px(fish[2]), px(fish[3])], outline=WHITE, width=medium)
    draw.line([px(fish[0]), px(plumb["bottom_reference_y_mm"]), px(fish[2]), px(plumb["bottom_reference_y_mm"])], fill=LIGHT, width=thin)
    draw.line([px(plumb["left_reference_x_mm"]), px(fish[1]), px(plumb["left_reference_x_mm"]), px(fish[3])], fill=LIGHT, width=thin)
    draw.line([px(plumb["right_reference_x_mm"]), px(fish[1]), px(plumb["right_reference_x_mm"]), px(fish[3])], fill=LIGHT, width=thin)
    for offset in (0, 10, 20, 30):
        draw.line([px(35), px(78 + offset), px(385), px(78 + offset)], fill=(28, 166, 220), width=thin)


def draw_direction_icon(draw: ImageDraw.ImageDraw) -> None:
    x = px(26)
    y = px(165)
    line_w = max(1, px(0.7))
    subtle = (90, 195, 228)
    draw.line([x + px(12), y, x - px(10), y], fill=subtle, width=line_w)
    draw.line([x - px(10), y, x - px(3), y - px(5)], fill=subtle, width=line_w)
    draw.line([x - px(10), y, x - px(3), y + px(5)], fill=subtle, width=line_w)


def build_board_image() -> Image.Image:
    image = Image.new("RGB", (px(BOARD_W_MM), px(BOARD_H_MM)), BLUE)
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, image.width, px(36)], fill=DEEP_BLUE)
    draw_centered_text(draw, (px(122), px(14.0)), "蓝子鱼拍照校准板 A3 V2", font(px(4.8), bold=True), fill=WHITE)
    draw_centered_text(draw, (px(300), px(14.0)), "ChArUco + Plumb-line", font(px(4.8), bold=True), fill=WHITE)

    draw_plumb_lines(draw)
    draw_ruler(draw)
    draw_sample_id_grid(draw)
    draw_direction_icon(draw)

    labels = {
        0: "ID 0 top-left",
        1: "ID 1 top-right",
        2: "ID 2 bottom-left",
        3: "ID 3 bottom-right",
        4: "ID 4 top-mid",
        5: "ID 5 right-mid",
        6: "ID 6 bottom-mid",
        7: "ID 7 left-mid",
    }
    for marker_id, square in A3_V2_CHARUCO_PLUMB_CONFIG["location_marker_squares_mm"].items():
        paste_location_marker(image, draw, int(marker_id), square, labels[int(marker_id)])
    paste_charuco_panel(image, draw)

    return image


def save_pdf(image: Image.Image, output_pdf: Path) -> None:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    c = canvas.Canvas(str(output_pdf), pagesize=(BOARD_W_MM * mm, BOARD_H_MM * mm))
    c.drawImage(ImageReader(buffer), 0, 0, width=BOARD_W_MM * mm, height=BOARD_H_MM * mm)
    c.showPage()
    c.save()


def save_svg_embedded_png(png_path: Path, output_svg: Path) -> None:
    encoded = base64.b64encode(png_path.read_bytes()).decode("ascii")
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{BOARD_W_MM}mm" height="{BOARD_H_MM}mm" viewBox="0 0 {BOARD_W_MM} {BOARD_H_MM}">
  <image href="data:image/png;base64,{encoded}" x="0" y="0" width="{BOARD_W_MM}" height="{BOARD_H_MM}" preserveAspectRatio="none"/>
</svg>
'''
    output_svg.write_text(svg, encoding="utf-8")


def save_metadata(output_json: Path) -> None:
    payload = {
        "name": A3_V2_CHARUCO_PLUMB_CONFIG["name"],
        "units": "mm",
        "paper": {"size": "A3 landscape", "width_mm": BOARD_W_MM, "height_mm": BOARD_H_MM},
        "dpi": DPI,
        "location_marker_dictionary": "DICT_4X4_50",
        "location_marker_squares_mm": A3_V2_CHARUCO_PLUMB_CONFIG["location_marker_squares_mm"],
        "charuco": A3_V2_CHARUCO_PLUMB_CONFIG["charuco"],
        "ruler": A3_V2_CHARUCO_PLUMB_CONFIG["ruler"],
        "plumb_lines": A3_V2_CHARUCO_PLUMB_CONFIG["plumb_lines"],
        "notes": [
            "Location ArUco markers use IDs 0-7 in DICT_4X4_50.",
            "The ChArUco panel uses DICT_6X6_250 and is intended for empty-board camera calibration/QC.",
            "Fish placement area is kept blue and visually clean for segmentation.",
        ],
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image = build_board_image()
    png_path = OUT_DIR / f"{STEM}_300dpi.png"
    pdf_path = OUT_DIR / f"{STEM}_print.pdf"
    svg_path = OUT_DIR / f"{STEM}.svg"
    json_path = OUT_DIR / f"{STEM}_metadata.json"
    preview_path = OUT_DIR / f"{STEM}_preview.jpg"

    image.save(png_path, dpi=(DPI, DPI))
    image.resize((1400, 990), Image.Resampling.LANCZOS).save(preview_path, quality=92)
    try:
        save_pdf(image, pdf_path)
    except PermissionError:
        pdf_path = OUT_DIR / f"{STEM}_print_updated.pdf"
        save_pdf(image, pdf_path)
    save_svg_embedded_png(png_path, svg_path)
    save_metadata(json_path)

    print(png_path)
    print(pdf_path)
    print(svg_path)
    print(json_path)
    print(preview_path)


if __name__ == "__main__":
    main()
