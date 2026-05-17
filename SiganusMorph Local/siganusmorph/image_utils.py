"""Image loading and display helpers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .config import SUPPORTED_IMAGE_EXTENSIONS


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
