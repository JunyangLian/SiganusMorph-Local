"""Pull remote image jobs, run SiganusMorph locally, and upload results."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping
from zipfile import ZIP_DEFLATED, ZipFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import requests
from PIL import Image

from siganusmorph.formal_ui import (
    calibrate_with_aruco,
    draw_formal_measurement_overlay,
    run_recommended_measurement,
)
from siganusmorph.image_utils import image_from_bytes
from siganusmorph.runtime_resources import configure_runtime_threads


REMOTE_METADATA_KEYS = {
    "measurement_axis_mode",
    "measurement_axes",
    "selected_measurement_axis",
    "measurements_needs_review",
    "measurement_axis_review_reason",
    "dual_axis_disagreement",
    "body_depth_geometry",
    "peduncle_depth_geometry",
    "tail_qc",
    "P6_gap_derivation",
    "operculum_qc",
    "qc_results",
    "hybrid_qc_results",
    "v06_qc_results",
    "v06_point_sources",
    "point_sources",
    "compressed_tail_tl",
    "review_reason",
    "needs_review",
    "mm_per_pixel",
}


def _image_bytes_jpeg(image: np.ndarray, *, quality: int = 88, max_side: int | None = None) -> bytes:
    """Encode result imagery compactly without changing measurement coordinates."""
    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
    if max_side and max(pil_image.size) > max_side:
        pil_image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality, optimize=True, progressive=True)
    return buffer.getvalue()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _calibration_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    warped = result.get("warped_image")
    return _jsonable(
        {
            "status": result.get("status", "failed"),
            "success": bool(result.get("success")),
            "calibration_message": result.get("calibration_message", ""),
            "status_text": result.get("status_text", ""),
            "marker_count": int(result.get("marker_count") or 0),
            "detected_markers": result.get("detected_markers", []),
            "detected_board_corners_raw": result.get("detected_board_corners_raw", []),
            "mm_per_pixel": result.get("mm_per_pixel"),
            "mm_per_pixel_source": result.get("mm_per_pixel_source", ""),
            "homography_raw_to_warped": result.get("homography_raw_to_warped"),
            "warped_width_px": int(warped.shape[1]) if isinstance(warped, np.ndarray) else None,
            "warped_height_px": int(warped.shape[0]) if isinstance(warped, np.ndarray) else None,
            "coordinate_space": "warped_image",
            "remote_compute": True,
        }
    )


def _measurement_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    metadata = result.get("metadata", {}) if isinstance(result.get("metadata", {}), Mapping) else {}
    remote_metadata = {key: metadata[key] for key in REMOTE_METADATA_KEYS if key in metadata}
    return _jsonable(
        {
            "full_keypoints": result.get("full_keypoints", {}),
            "formal_measurement_points": result.get("formal_measurement_points", {}),
            "formal_measurements": result.get("formal_measurements", {}),
            "formal_derived_points": result.get("formal_derived_points", {}),
            "metadata": remote_metadata,
        }
    )


class RemoteClient:
    def __init__(self, base_url: str, token: str, worker_id: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.worker_id = worker_id
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {token}"})

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def claim(self) -> dict[str, Any] | None:
        response = self.session.post(
            self._url("worker/claim"), json={"worker_id": self.worker_id}, timeout=30
        )
        response.raise_for_status()
        return response.json().get("job")

    def heartbeat(self, job_id: str, progress: str) -> None:
        response = self.session.post(
            self._url(f"worker/jobs/{job_id}/heartbeat"),
            json={"worker_id": self.worker_id, "progress": progress},
            timeout=30,
        )
        response.raise_for_status()

    def download(self, job_id: str) -> bytes:
        response = self.session.get(self._url(f"worker/jobs/{job_id}/input"), timeout=180)
        response.raise_for_status()
        return response.content

    def complete(self, job_id: str, package: bytes) -> None:
        response = self.session.post(
            self._url(f"worker/jobs/{job_id}/result"),
            params={"worker_id": self.worker_id},
            data=package,
            headers={"Content-Type": "application/zip"},
            timeout=600,
        )
        response.raise_for_status()

    def fail(self, job_id: str, message: str) -> None:
        response = self.session.post(
            self._url(f"worker/jobs/{job_id}/fail"),
            json={"worker_id": self.worker_id, "error_message": message[:2000]},
            timeout=30,
        )
        response.raise_for_status()


def process_job(client: RemoteClient, job: Mapping[str, Any]) -> None:
    job_id = str(job["job_id"])
    image_name = str(job.get("image_name") or "uploaded_image")
    client.heartbeat(job_id, "正在下载原始图片")
    raw_bytes = client.download(job_id)
    raw_image = image_from_bytes(raw_bytes)

    client.heartbeat(job_id, "正在检测校准板并生成校正图像")
    calibration = calibrate_with_aruco(raw_image)
    payload: dict[str, Any] = {
        "schema_version": "remote_measurement_result_v1",
        "job_id": job_id,
        "image_name": image_name,
        "specimen_id": job.get("specimen_id", ""),
        "weight_g": job.get("weight_g"),
        "status": "calibration_failed",
        "calibration_state": _calibration_payload(calibration),
        "measurement_result": None,
    }

    archive_buffer = BytesIO()
    if calibration.get("status") == "success" and isinstance(calibration.get("warped_image"), np.ndarray):
        warped = calibration["warped_image"]
        client.heartbeat(job_id, "校准完成，正在运行关键点与几何测量")
        measurement = run_recommended_measurement(
            warped,
            image_name,
            PROJECT_ROOT,
            mm_per_pixel=float(calibration["mm_per_pixel"]),
        )
        payload["status"] = "success"
        payload["measurement_result"] = _measurement_payload(measurement)
        preview = draw_formal_measurement_overlay(
            warped,
            measurement.get("formal_measurement_points", {}),
            measurement.get("formal_measurements", {}),
            measurement.get("metadata", {}),
            image_name=image_name,
            specimen_id=str(job.get("specimen_id") or ""),
        )
        client.heartbeat(job_id, "正在打包并回传测量结果")
        with ZipFile(archive_buffer, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("result.json", json.dumps(payload, ensure_ascii=False).encode("utf-8"))
            archive.writestr("warped.jpg", _image_bytes_jpeg(warped, quality=88))
            archive.writestr("preview.jpg", _image_bytes_jpeg(preview, quality=86, max_side=1600))
            if calibration.get("raw_marker_overlay") is not None:
                archive.writestr(
                    "raw_marker_overlay.jpg",
                    _image_bytes_jpeg(calibration["raw_marker_overlay"], quality=84, max_side=1280),
                )
    else:
        with ZipFile(archive_buffer, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("result.json", json.dumps(payload, ensure_ascii=False).encode("utf-8"))
            if calibration.get("raw_marker_overlay") is not None:
                archive.writestr(
                    "raw_marker_overlay.jpg",
                    _image_bytes_jpeg(calibration["raw_marker_overlay"], quality=84, max_side=1280),
                )
    client.complete(job_id, archive_buffer.getvalue())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Process at most one job, then exit")
    parser.add_argument("--poll-seconds", type=float, default=3.0)
    args = parser.parse_args()
    base_url = os.getenv("SIGANUS_REMOTE_API_URL", "").strip()
    token = os.getenv("SIGANUS_REMOTE_WORKER_TOKEN", "").strip()
    worker_id = os.getenv("SIGANUS_REMOTE_WORKER_ID", "bio-server-worker-01").strip()
    if not base_url.startswith("https://"):
        raise SystemExit("SIGANUS_REMOTE_API_URL must use HTTPS")
    if len(token) < 32:
        raise SystemExit("SIGANUS_REMOTE_WORKER_TOKEN must contain at least 32 characters")

    configure_runtime_threads()
    client = RemoteClient(base_url, token, worker_id)
    print(f"Remote worker {worker_id} polling {base_url}", flush=True)
    while True:
        try:
            job = client.claim()
            if not job:
                if args.once:
                    return
                time.sleep(max(1.0, args.poll_seconds))
                continue
            print(f"Claimed {job['job_id']}: {job.get('image_name', '')}", flush=True)
            try:
                process_job(client, job)
            except Exception as exc:
                print(f"Job {job['job_id']} failed: {exc}", file=sys.stderr, flush=True)
                try:
                    client.fail(str(job["job_id"]), f"{type(exc).__name__}: {exc}")
                except Exception as report_exc:
                    print(f"Could not report failure: {report_exc}", file=sys.stderr, flush=True)
            else:
                print(f"Completed {job['job_id']}", flush=True)
            if args.once:
                return
        except KeyboardInterrupt:
            return
        except Exception as exc:
            print(f"Worker polling error: {exc}", file=sys.stderr, flush=True)
            if args.once:
                raise
            time.sleep(max(3.0, args.poll_seconds))


if __name__ == "__main__":
    main()
