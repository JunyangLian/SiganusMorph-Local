"""Image loading and display helpers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .config import SUPPORTED_IMAGE_EXTENSIONS

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:
    pass


def load_image_file(path: str | Path) -> np.ndarray:
    """Load an image file as an RGB numpy array."""
    with Image.open(path) as image:
        return pil_to_rgb_array(image)


def image_from_bytes(data: bytes) -> np.ndarray:
    """Load uploaded image bytes as an RGB numpy array."""
    with Image.open(BytesIO(data)) as image:
        return pil_to_rgb_array(image)


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    """Apply EXIF orientation and return RGB array."""
    image = ImageOps.exif_transpose(image)
    return np.asarray(image.convert("RGB"))


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Return a 3-channel RGB array."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        return np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[2] == 4:
        return arr[:, :, :3]
    return arr


def resize_for_display(
    image: np.ndarray | Image.Image,
    max_width: int = 1100,
) -> tuple[Image.Image, float]:
    """Resize an image for the web UI.

    Returns the display image and a scale factor that converts display
    coordinates back to original image coordinates.
    """
    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        pil_image = Image.fromarray(ensure_rgb(image))

    width, height = pil_image.size
    if width <= max_width:
        return pil_image, 1.0

    display_width = int(max_width)
    display_height = max(1, round(height * display_width / width))
    display_image = pil_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
    return display_image, width / display_width


def prepare_zoom_display(
    image: np.ndarray | Image.Image,
    max_width: int = 1100,
    zoom_mode: bool = False,
    zoom_center: tuple[float, float] | None = None,
    zoom_factor: float = 3.0,
) -> tuple[Image.Image, dict[str, float | bool]]:
    """Prepare an interactive full-image or zoomed crop display.

    The returned transform maps display coordinates back to original image
    coordinates:

    ``original_x = offset_x + display_x * scale_x``
    ``original_y = offset_y + display_y * scale_y``
    """
    full_display, full_scale = resize_for_display(image, max_width=max_width)
    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        pil_image = Image.fromarray(ensure_rgb(image))

    width, height = pil_image.size
    display_width, display_height = full_display.size
    transform: dict[str, float | bool] = {
        "offset_x": 0.0,
        "offset_y": 0.0,
        "scale_x": float(full_scale),
        "scale_y": float(full_scale),
        "display_width": float(display_width),
        "display_height": float(display_height),
        "zoom_mode": False,
        "zoom_factor": 1.0,
    }
    if not zoom_mode or zoom_center is None:
        return full_display, transform

    factor = max(1.0, float(zoom_factor))
    crop_width = min(float(width), display_width * full_scale / factor)
    crop_height = min(float(height), display_height * full_scale / factor)
    center_x = min(max(float(zoom_center[0]), crop_width / 2), float(width) - crop_width / 2)
    center_y = min(max(float(zoom_center[1]), crop_height / 2), float(height) - crop_height / 2)
    left = max(0.0, min(float(width) - crop_width, center_x - crop_width / 2))
    top = max(0.0, min(float(height) - crop_height, center_y - crop_height / 2))
    right = min(float(width), left + crop_width)
    bottom = min(float(height), top + crop_height)

    crop = pil_image.crop((round(left), round(top), round(right), round(bottom)))
    display = crop.resize((display_width, display_height), Image.Resampling.LANCZOS)
    transform.update(
        {
            "offset_x": float(round(left)),
            "offset_y": float(round(top)),
            "scale_x": float((round(right) - round(left)) / display_width),
            "scale_y": float((round(bottom) - round(top)) / display_height),
            "zoom_mode": True,
            "zoom_factor": factor,
        }
    )
    return display, transform


def display_to_original(
    point: dict[str, float],
    transform: dict[str, float | bool],
) -> dict[str, float]:
    """Convert a click on the current display image to original coordinates."""
    return {
        "x": float(transform["offset_x"]) + float(point["x"]) * float(transform["scale_x"]),
        "y": float(transform["offset_y"]) + float(point["y"]) * float(transform["scale_y"]),
    }


def list_image_files(folder: str | Path) -> list[Path]:
    """List supported image files in a folder."""
    root = Path(folder).expanduser()
    if not root.exists() or not root.is_dir():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
