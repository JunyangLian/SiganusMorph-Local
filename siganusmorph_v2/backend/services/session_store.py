from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
V2_OUTPUT_ROOT = PROJECT_ROOT / "results" / "v2_user_outputs"
SESSION_ROOT = V2_OUTPUT_ROOT / "sessions"
EXPORT_ROOT = V2_OUTPUT_ROOT / "exports"

SESSION_ID_PATTERN = re.compile(r"^[0-9a-fA-F-]{36}$")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def jsonable(value: Any) -> Any:
    """Convert numpy/scalar/path objects into JSON-safe values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def session_dir(session_id: str) -> Path:
    validate_session_id(session_id)
    return SESSION_ROOT / session_id


def asset_url(session_id: str, filename: str) -> str:
    validate_session_id(session_id)
    return f"/api/assets/{session_id}/{filename}"


def validate_session_id(session_id: str) -> str:
    """Accept only canonical UUID session ids to keep V2 paths inside SESSION_ROOT."""
    if not SESSION_ID_PATTERN.match(str(session_id)):
        raise ValueError("Invalid V2 session_id.")
    try:
        parsed = UUID(str(session_id))
    except ValueError as exc:
        raise ValueError("Invalid V2 session_id.") from exc
    canonical = str(parsed)
    if canonical.lower() != str(session_id).lower():
        raise ValueError("Invalid V2 session_id.")
    return canonical


def create_session(image_name: str, *, specimen_id: str = "", weight_g: float | None = None) -> dict[str, Any]:
    session_id = str(uuid4())
    directory = session_dir(session_id)
    directory.mkdir(parents=True, exist_ok=True)
    session = {
        "version": "2.0.0",
        "session_id": session_id,
        "image_name": image_name,
        "specimen_id": specimen_id or "",
        "weight_g": weight_g,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "status": "uploaded",
        "coordinate_space": "warped_image",
        "raw_image": {},
        "calibration_state": {"status": "not_run"},
        "formal_points": {},
        "derived_points": {},
        "measurements": {},
        "qc": {},
        "measurement_overrides": {},
        "exports": {},
    }
    save_session(session)
    return session


def save_session(session: dict[str, Any]) -> None:
    session["updated_at"] = now_iso()
    directory = session_dir(str(session["session_id"]))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "session.json").write_text(
        json.dumps(jsonable(session), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_session(session_id: str) -> dict[str, Any]:
    path = session_dir(session_id) / "session.json"
    if not path.exists():
        raise FileNotFoundError(f"Unknown V2 session: {session_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def update_session(session_id: str, patch: dict[str, Any]) -> dict[str, Any]:
    session = load_session(session_id)
    session.update(patch)
    save_session(session)
    return session
