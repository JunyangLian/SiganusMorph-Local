"""JSON keypoint persistence."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .config import BODY_AXIS_POINT_KEYS, CAUDAL_TIP_CANDIDATE_KEYS, KEYPOINT_BY_NAME, KEYPOINT_DEFS


def save_keypoints_json(
    image_name: str,
    keypoints: Mapping[str, Mapping[str, float]],
    output_path: str | Path,
    metadata: Mapping[str, Any] | None = None,
    measurements: Mapping[str, Any] | None = None,
) -> Path:
    """Save keypoints and metadata to a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_dict = dict(metadata or {})
    formatted_keypoints: dict[str, list[float]] = {}
    for definition in KEYPOINT_DEFS:
        point = keypoints.get(definition.name)
        if point is None:
            continue
        formatted_keypoints[f"{definition.code}_{definition.name}"] = [
            float(point["x"]),
            float(point["y"]),
        ]
    for name, point in keypoints.items():
        if name in KEYPOINT_BY_NAME:
            continue
        formatted_keypoints[name] = [float(point["x"]), float(point["y"])]
    measurements_dict = dict(measurements or {})
    derived_payload: dict[str, list[float]] = {}
    derived_points = measurements_dict.get("derived_points", {})
    if isinstance(derived_points, Mapping):
        for name, point in derived_points.items():
            if isinstance(point, Mapping):
                derived_payload[str(name)] = [float(point["x"]), float(point["y"])]

    payload = {
        "image_name": image_name,
        "specimen_id": metadata_dict.get("specimen_id", ""),
        "source_type": metadata_dict.get("source_type", ""),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metadata": metadata_dict,
        "keypoints": formatted_keypoints,
        "derived_points": derived_payload,
        "body_axis_points_order": list(BODY_AXIS_POINT_KEYS),
        "caudal_tip_candidates": list(CAUDAL_TIP_CANDIDATE_KEYS),
        "caudal_tip_selected": measurements_dict.get("caudal_tip_selected", ""),
        "axis_mode_selected": metadata_dict.get("axis_mode_selected", ""),
        "needs_review": bool(metadata_dict.get("needs_review", False)),
        "notes": str(metadata_dict.get("notes", "")),
        "measurements": measurements_dict,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_keypoints_json(json_path: str | Path) -> dict[str, Any]:
    """Load a saved keypoint JSON file."""
    return json.loads(Path(json_path).read_text(encoding="utf-8"))
