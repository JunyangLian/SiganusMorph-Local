"""Generate fish keypoint template PNGs for the Streamlit helper panel."""

from __future__ import annotations

from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from siganusmorph.config import BODY_AXIS_POINT_ORDER, CAUDAL_FIN_AXIS_POINT_ORDER, KEYPOINT_DEFS  # noqa: E402

OUT_DIR = ROOT / "assets" / "templates"
SOURCE_IMAGE = OUT_DIR / "template_fish_source.png"
CANVAS_SIZE = (900, 450)

POINTS = {
    "snout_tip": (100, 236),
    "eye_front": (170, 181),
    "operculum_posterior": (265, 246),
    "peduncle_start_midpoint": (645, 250),
    "caudal_base_midpoint": (760, 250),
    "caudal_fork_midpoint": (824, 250),
    "caudal_fin_upper_tip": (870, 162),
    "caudal_fin_lower_tip": (858, 330),
    "caudal_fin_posterior_endpoint": (870, 250),
    "body_depth_dorsal": (405, 112),
    "body_depth_ventral": (405, 343),
    "peduncle_depth_dorsal": (710, 218),
    "peduncle_depth_ventral": (710, 282),
    "head_axis_point": (225, 236),
    "trunk_axis_point": (360, 232),
    "posterior_trunk_axis_point": (535, 238),
    "peduncle_axis_point": (703, 250),
}

AXIS_NAMES = BODY_AXIS_POINT_ORDER
CAUDAL_AXIS_NAMES = CAUDAL_FIN_AXIS_POINT_ORDER

LABEL_OFFSETS = {
    "P1": (12, 12),
    "P2": (14, -22),
    "P3": (-42, 18),
    "P4": (18, -18),
    "P5": (18, -18),
    "P6": (-32, 18),
    "P7U": (-54, -34),
    "P7L": (-54, 18),
    "P7V": (-32, -42),
    "C4": (-34, -34),
    "P8": (14, -22),
    "P9": (14, 4),
    "P10": (18, -28),
    "P11": (18, 2),
}


def font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "arial.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_fish(draw: ImageDraw.ImageDraw) -> None:
    axis = [POINTS[name] for name in AXIS_NAMES]
    for a, b in zip(axis, axis[1:]):
        draw.line((a[0], a[1], b[0], b[1]), fill=(35, 110, 210), width=4)
    caudal_axis = [POINTS[name] for name in CAUDAL_AXIS_NAMES]
    for a, b in zip(caudal_axis, caudal_axis[1:]):
        draw.line((a[0], a[1], b[0], b[1]), fill=(150, 70, 190), width=4)
    draw.line(
        (
            POINTS["caudal_base_midpoint"][0],
            POINTS["caudal_base_midpoint"][1],
            POINTS["caudal_fin_posterior_endpoint"][0],
            POINTS["caudal_fin_posterior_endpoint"][1],
        ),
        fill=(245, 130, 32),
        width=4,
    )


def draw_point(
    draw: ImageDraw.ImageDraw,
    code: str,
    xy: tuple[int, int],
    is_axis_auxiliary: bool,
    highlight: bool = False,
    dim: bool = False,
) -> None:
    if dim:
        fill = (178, 188, 190)
        outline = (255, 255, 255)
    else:
        fill = (46, 204, 113) if is_axis_auxiliary else (230, 57, 70)
        outline = (20, 28, 32)
    radius = 17 if highlight else 11
    width = 5 if highlight else 3
    x, y = xy
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=width)
    text_fill = fill if not dim else (90, 98, 100)
    dx, dy = LABEL_OFFSETS.get(code, (radius + 4, -radius - 2))
    draw.text((x + dx, y + dy), code, font=font(24 if highlight else 19), fill=text_fill)


def draw_star(draw: ImageDraw.ImageDraw, code: str, xy: tuple[int, int]) -> None:
    x, y = xy
    radius = 13
    points = [
        (x, y - radius),
        (x + 4, y - 4),
        (x + radius, y - 4),
        (x + 6, y + 3),
        (x + 8, y + radius),
        (x, y + 7),
        (x - 8, y + radius),
        (x - 6, y + 3),
        (x - radius, y - 4),
        (x - 4, y - 4),
    ]
    draw.polygon(points, fill=(255, 220, 40), outline=(120, 70, 160))
    dx, dy = LABEL_OFFSETS.get(code, (18, -18))
    draw.text((x + dx, y + dy), code, font=font(19), fill=(120, 70, 160))


def base_image() -> Image.Image:
    if SOURCE_IMAGE.exists():
        image = Image.open(SOURCE_IMAGE).convert("RGB").resize(CANVAS_SIZE, Image.Resampling.LANCZOS)
    else:
        image = Image.new("RGB", CANVAS_SIZE, (246, 248, 247))
    draw = ImageDraw.Draw(image)
    draw_fish(draw)
    return image


def draw_template(highlight_code: str | None = None) -> Image.Image:
    image = base_image()
    draw = ImageDraw.Draw(image)
    for definition in KEYPOINT_DEFS:
        xy = POINTS[definition.name]
        highlight = definition.code == highlight_code
        dim = highlight_code is not None and not highlight
        draw_point(draw, definition.code, xy, definition.code.startswith("C"), highlight=highlight, dim=dim)
    if highlight_code is None:
        draw_star(draw, "P7V", POINTS["caudal_fin_posterior_endpoint"])
    title = "16-point overview" if highlight_code is None else f"Current point: {highlight_code}"
    draw.rectangle((16, 14, 330, 52), fill=(255, 255, 255), outline=(190, 198, 196), width=1)
    draw.text((28, 20), title, font=font(24), fill=(38, 48, 52))
    return image


def draw_zoom(definition_code: str) -> Image.Image:
    definition = next(item for item in KEYPOINT_DEFS if item.code == definition_code)
    image = draw_template(definition_code)
    x, y = POINTS[definition.name]
    margin_x = 170
    margin_y = 115
    left = max(0, x - margin_x)
    top = max(0, y - margin_y)
    right = min(CANVAS_SIZE[0], x + margin_x)
    bottom = min(CANVAS_SIZE[1], y + margin_y)
    crop = image.crop((left, top, right, bottom)).resize((520, 320), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(crop)
    label = f"{definition.code} {definition.name}"
    draw.rectangle((12, 12, 506, 48), fill=(255, 255, 255), outline=(190, 198, 196), width=1)
    draw.text((22, 18), label, font=font(20), fill=(38, 48, 52))
    return crop


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    draw_template().save(OUT_DIR / "overview_template.png")
    for definition in KEYPOINT_DEFS:
        draw_template(definition.code).save(OUT_DIR / f"{definition.code}_template.png")
        draw_zoom(definition.code).save(OUT_DIR / f"{definition.code}_zoom.png")
    print(f"Generated template assets in {OUT_DIR}")


if __name__ == "__main__":
    main()
