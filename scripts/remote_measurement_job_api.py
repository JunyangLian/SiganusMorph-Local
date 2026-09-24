"""Authenticated pull API for remote SiganusMorph measurement workers."""

from __future__ import annotations

import argparse
import hmac
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from siganusmorph.remote_jobs import (
    claim_remote_job,
    complete_remote_job,
    fail_remote_job,
    remote_job_input,
    update_remote_job,
    initialize_remote_job_database,
)


class WorkerRequest(BaseModel):
    worker_id: str
    progress: str = ""


class FailureRequest(BaseModel):
    worker_id: str
    error_message: str


def require_worker_token(authorization: str = Header(default="")) -> None:
    expected = os.getenv("SIGANUS_REMOTE_WORKER_TOKEN", "")
    provided = authorization.removeprefix("Bearer ").strip()
    if len(expected) < 32 or not hmac.compare_digest(expected, provided):
        raise HTTPException(status_code=401, detail="Invalid worker token")


app = FastAPI(title="SiganusMorph Remote Worker API", version="1.0")


@app.on_event("startup")
def initialize_queue() -> None:
    initialize_remote_job_database(project_root=PROJECT_ROOT)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "siganusmorph-remote-worker-api"}


@app.post("/worker/claim", dependencies=[Depends(require_worker_token)])
def claim(request: WorkerRequest) -> dict[str, object]:
    job = claim_remote_job(request.worker_id, project_root=PROJECT_ROOT)
    return {"job": job}


@app.get("/worker/jobs/{job_id}/input", dependencies=[Depends(require_worker_token)])
def download_input(job_id: str) -> FileResponse:
    try:
        path, metadata = remote_job_input(job_id, project_root=PROJECT_ROOT)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, filename=str(metadata.get("image_name") or "input_image"))


@app.post("/worker/jobs/{job_id}/heartbeat", dependencies=[Depends(require_worker_token)])
def heartbeat(job_id: str, request: WorkerRequest) -> dict[str, object]:
    try:
        job = update_remote_job(
            job_id,
            request.worker_id,
            status="processing",
            progress=request.progress or "远程计算中",
            project_root=PROJECT_ROOT,
        )
    except KeyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"job": job}


@app.post("/worker/jobs/{job_id}/result", dependencies=[Depends(require_worker_token)])
async def upload_result(job_id: str, request: Request, worker_id: str) -> dict[str, object]:
    content = await request.body()
    try:
        job = complete_remote_job(job_id, worker_id, content, project_root=PROJECT_ROOT)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"job": job}


@app.post("/worker/jobs/{job_id}/fail", dependencies=[Depends(require_worker_token)])
def fail(job_id: str, request: FailureRequest) -> dict[str, object]:
    try:
        job = fail_remote_job(
            job_id,
            request.worker_id,
            request.error_message,
            project_root=PROJECT_ROOT,
        )
    except KeyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"job": job}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if len(os.getenv("SIGANUS_REMOTE_WORKER_TOKEN", "")) < 32:
        raise SystemExit("SIGANUS_REMOTE_WORKER_TOKEN must contain at least 32 characters")
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
