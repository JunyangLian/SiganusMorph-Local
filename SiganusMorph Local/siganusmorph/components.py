"""Small Streamlit components bundled with the project."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
import streamlit.components.v1 as components


_COMPONENT_DIR = Path(__file__).resolve().parent / "web_components" / "image_clicker"
_image_clicker = components.declare_component("siganus_image_clicker", path=str(_COMPONENT_DIR))


def image_clicker(image: Image.Image, key: str) -> dict[str, Any] | None:
    """Display an image and return clicked display coordinates."""
    rgb_image = image.convert("RGB")
    buffer = BytesIO()
    rgb_image.save(buffer, format="PNG")
    image_data = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    return _image_clicker(
        image_data=image_data,
        width=rgb_image.width,
        height=rgb_image.height,
        key=key,
        default=None,
    )
