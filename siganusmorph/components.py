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
_MEASUREMENT_EDITOR_DIR = Path(__file__).resolve().parent / "web_components" / "measurement_editor"
_measurement_editor = components.declare_component("siganus_measurement_editor", path=str(_MEASUREMENT_EDITOR_DIR))


def image_clicker(image: Image.Image, key: str) -> dict[str, Any] | None:
    """Display an image and return clicked display coordinates.

    The returned payload includes ``button`` with ``left`` or ``right``.
    """
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


def measurement_editor(
    image: Image.Image,
    *,
    points: list[dict[str, Any]],
    lines: list[dict[str, Any]] | None = None,
    polylines: list[dict[str, Any]] | None = None,
    stars: list[dict[str, Any]] | None = None,
    mm_per_pixel: float = 0.1,
    show_labels: bool = False,
    enable_magnifier: bool = True,
    image_coord_width: float | None = None,
    image_coord_height: float | None = None,
    display_to_image_scale: float = 1.0,
    debug: bool = False,
    key: str,
) -> dict[str, Any] | None:
    """Display a draggable formal measurement overlay and return edited points."""
    rgb_image = image.convert("RGB")
    buffer = BytesIO()
    rgb_image.save(buffer, format="PNG")
    image_data = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    return _measurement_editor(
        image_data=image_data,
        width=rgb_image.width,
        height=rgb_image.height,
        points=points,
        lines=lines or [],
        polylines=polylines or [],
        stars=stars or [],
        mm_per_pixel=float(mm_per_pixel),
        show_labels=bool(show_labels),
        enable_magnifier=bool(enable_magnifier),
        image_coord_width=float(image_coord_width if image_coord_width is not None else rgb_image.width),
        image_coord_height=float(image_coord_height if image_coord_height is not None else rgb_image.height),
        display_to_image_scale=float(display_to_image_scale),
        debug=bool(debug),
        key=key,
        default=None,
    )
