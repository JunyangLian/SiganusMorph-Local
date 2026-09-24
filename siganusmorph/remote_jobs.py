"""File-backed SQLite queue for remote SiganusMorph measurement workers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import sqlite3
import threading
from typing import Any
from uuid import uuid4
from zipfile import ZipFile

from .image_utils import image_from_bytes


ACTIVE_STATUSES = {"queued", "claimed", "processing"}
TERMINAL_STATUSES = {"completed", "failed"}
MAX_INPUT_BYTES = 40 * 1024 * 1024
MAX_RESULT_BYTES = 80 * 1024 * 1024
_INITIALIZED_DATABASES: set[str] = set()
_INITIALIZE_LOCK = threading.Lock()


def remote_compute_enabled() -> bool:
    return os.getenv("SIGANUSMORPH_REMOTE_COMPUTE", "").strip().lower() in {"1", "true", "yes", "on"}


def remote_job_root(project_root: Path | None = None) -> Path:
    configured = os.getenv("SIGANUS_REMOTE_JOB_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    root = Path(project_root or Path.cwd()).resolve()
    return root / "results" / "remote_measurement_jobs"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect_raw(root: Path) -> sqlite3.Connection:
    root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(root / "jobs.sqlite3", timeout=5, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=5000")
    return connection


def initialize_remote_job_database(*, project_root: Path | None = None) -> Path:
    """Initialize the queue schema once per process.

    Reissuing ``PRAGMA journal_mode=WAL`` for every poll can require a write
    lock and stall the API while Streamlit is reading the same database.
    """
    root = remote_job_root(project_root)
    database_key = str((root / "jobs.sqlite3").resolve())
    if database_key in _INITIALIZED_DATABASES:
        return root
    with _INITIALIZE_LOCK:
        if database_key in _INITIALIZED_DATABASES:
            return root
        connection = _connect_raw(root)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    progress TEXT NOT NULL DEFAULT '',
                    image_name TEXT NOT NULL,
                    specimen_id TEXT NOT NULL DEFAULT '',
                    weight_g REAL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    worker_id TEXT NOT NULL DEFAULT '',
                    lease_until TEXT,
                    error_message TEXT NOT NULL DEFAULT ''
                )
                """
            )
        finally:
            connection.close()
        _INITIALIZED_DATABASES.add(database_key)
    return root


def _connect(root: Path) -> sqlite3.Connection:
    initialize_remote_job_database(project_root=root.parent.parent)
    return _connect_raw(root)


def _job_dir(root: Path, job_id: str) -> Path:
    if not job_id or any(char not in "0123456789abcdef-" for char in job_id.lower()):
        raise ValueError("Invalid remote job id")
    return root / "jobs" / job_id


def _public_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "job_id": row["job_id"],
        "status": row["status"],
        "progress": row["progress"],
        "image_name": row["image_name"],
        "specimen_id": row["specimen_id"],
        "weight_g": row["weight_g"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "worker_id": row["worker_id"],
        "error_message": row["error_message"],
    }


def submit_remote_job(
    image_bytes: bytes,
    *,
    image_name: str,
    specimen_id: str = "",
    weight_g: float | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    if not image_bytes or len(image_bytes) > MAX_INPUT_BYTES:
        raise ValueError("Uploaded image is empty or exceeds the 40 MB remote-job limit")
    root = remote_job_root(project_root)
    job_id = str(uuid4())
    directory = _job_dir(root, job_id)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "input.bin").write_bytes(image_bytes)
    metadata = {
        "job_id": job_id,
        "image_name": Path(str(image_name or "uploaded_image")).name,
        "specimen_id": str(specimen_id or ""),
        "weight_g": weight_g,
        "submitted_at": _now(),
    }
    (directory / "input_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    now = _now()
    with _connect(root) as connection:
        connection.execute(
            "INSERT INTO jobs(job_id,status,progress,image_name,specimen_id,weight_g,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
            (job_id, "queued", "等待计算节点领取", metadata["image_name"], metadata["specimen_id"], weight_g, now, now),
        )
        row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    return _public_row(row) or {}


def get_remote_job(job_id: str, *, project_root: Path | None = None) -> dict[str, Any] | None:
    root = remote_job_root(project_root)
    with _connect(root) as connection:
        row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    return _public_row(row)


def get_remote_jobs(
    job_ids: list[str] | tuple[str, ...],
    *,
    project_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Fetch a batch of queue rows with one short read transaction."""
    valid_ids = [str(job_id) for job_id in job_ids if str(job_id)]
    for job_id in valid_ids:
        _job_dir(remote_job_root(project_root), job_id)
    if not valid_ids:
        return {}
    root = remote_job_root(project_root)
    placeholders = ",".join("?" for _ in valid_ids)
    with _connect(root) as connection:
        rows = connection.execute(
            f"SELECT * FROM jobs WHERE job_id IN ({placeholders})",  # noqa: S608 - placeholders only.
            valid_ids,
        ).fetchall()
    return {
        str(row["job_id"]): public
        for row in rows
        if (public := _public_row(row)) is not None
    }


def claim_remote_job(
    worker_id: str,
    *,
    lease_seconds: int = 7200,
    project_root: Path | None = None,
) -> dict[str, Any] | None:
    root = remote_job_root(project_root)
    now = datetime.now(timezone.utc)
    now_text = now.isoformat(timespec="seconds")
    lease_until = (now + timedelta(seconds=max(300, lease_seconds))).isoformat(timespec="seconds")
    connection = _connect(root)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "UPDATE jobs SET status='queued',progress='计算节点租约过期，重新排队',worker_id='',lease_until=NULL,updated_at=? "
            "WHERE status IN ('claimed','processing') AND lease_until IS NOT NULL AND lease_until < ?",
            (now_text, now_text),
        )
        row = connection.execute(
            "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1"
        ).fetchone()
        if row is None:
            connection.execute("COMMIT")
            return None
        connection.execute(
            "UPDATE jobs SET status='claimed',progress='计算节点已领取',worker_id=?,lease_until=?,updated_at=? WHERE job_id=?",
            (worker_id, lease_until, now_text, row["job_id"]),
        )
        connection.execute("COMMIT")
        claimed = connection.execute("SELECT * FROM jobs WHERE job_id=?", (row["job_id"],)).fetchone()
        return _public_row(claimed)
    except Exception:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def update_remote_job(
    job_id: str,
    worker_id: str,
    *,
    status: str = "processing",
    progress: str = "",
    lease_seconds: int = 7200,
    project_root: Path | None = None,
) -> dict[str, Any]:
    if status not in {"claimed", "processing"}:
        raise ValueError("Invalid active remote-job status")
    root = remote_job_root(project_root)
    now = datetime.now(timezone.utc)
    lease_until = (now + timedelta(seconds=max(300, lease_seconds))).isoformat(timespec="seconds")
    with _connect(root) as connection:
        cursor = connection.execute(
            "UPDATE jobs SET status=?,progress=?,lease_until=?,updated_at=? WHERE job_id=? AND worker_id=?",
            (status, progress, lease_until, now.isoformat(timespec="seconds"), job_id, worker_id),
        )
        if cursor.rowcount != 1:
            raise KeyError("Remote job was not claimed by this worker")
        row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    return _public_row(row) or {}


def complete_remote_job(
    job_id: str,
    worker_id: str,
    result_bytes: bytes,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    if not result_bytes or len(result_bytes) > MAX_RESULT_BYTES:
        raise ValueError("Remote result is empty or exceeds the 80 MB limit")
    root = remote_job_root(project_root)
    directory = _job_dir(root, job_id)
    temporary = directory / "result.zip.tmp"
    temporary.write_bytes(result_bytes)
    with ZipFile(BytesIO(result_bytes), "r") as archive:
        if "result.json" not in archive.namelist():
            temporary.unlink(missing_ok=True)
            raise ValueError("Remote result package is missing result.json")
        json.loads(archive.read("result.json").decode("utf-8"))
    temporary.replace(directory / "result.zip")
    with _connect(root) as connection:
        cursor = connection.execute(
            "UPDATE jobs SET status='completed',progress='测量完成',lease_until=NULL,updated_at=? WHERE job_id=? AND worker_id=?",
            (_now(), job_id, worker_id),
        )
        if cursor.rowcount != 1:
            raise KeyError("Remote job was not claimed by this worker")
        row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    return _public_row(row) or {}


def fail_remote_job(
    job_id: str,
    worker_id: str,
    error_message: str,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = remote_job_root(project_root)
    message = str(error_message or "Remote measurement failed")[:2000]
    with _connect(root) as connection:
        cursor = connection.execute(
            "UPDATE jobs SET status='failed',progress='测量失败',error_message=?,lease_until=NULL,updated_at=? "
            "WHERE job_id=? AND worker_id=?",
            (message, _now(), job_id, worker_id),
        )
        if cursor.rowcount != 1:
            raise KeyError("Remote job was not claimed by this worker")
        row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    return _public_row(row) or {}


def remote_job_input(job_id: str, *, project_root: Path | None = None) -> tuple[Path, dict[str, Any]]:
    root = remote_job_root(project_root)
    directory = _job_dir(root, job_id)
    path = directory / "input.bin"
    metadata_path = directory / "input_metadata.json"
    if not path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Remote job input is unavailable")
    return path, json.loads(metadata_path.read_text(encoding="utf-8"))


def load_remote_result(
    job_id: str,
    *,
    project_root: Path | None = None,
    include_warped: bool = True,
) -> dict[str, Any]:
    root = remote_job_root(project_root)
    path = _job_dir(root, job_id) / "result.zip"
    if not path.exists():
        raise FileNotFoundError("Remote result package is unavailable")
    with ZipFile(path, "r") as archive:
        payload = json.loads(archive.read("result.json").decode("utf-8"))
        members = set(archive.namelist())
        # The formal page only needs the corrected image. Loading all three
        # rendered images triples peak memory on a 2 GB web host.
        warped_member = "warped.jpg" if "warped.jpg" in members else "warped.png"
        if include_warped and warped_member in members:
            payload["warped_image"] = image_from_bytes(archive.read(warped_member))
    return payload
