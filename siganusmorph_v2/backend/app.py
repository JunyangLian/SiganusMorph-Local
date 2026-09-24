from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routers.api import router as api_router
from .services.session_store import SESSION_ROOT


def create_app() -> FastAPI:
    app = FastAPI(
        title="SiganusMorph Local V2.0 API",
        version="2.0.0",
        description="FastAPI backend for SiganusMorph Local V2.0.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "tauri://localhost"],
        allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    SESSION_ROOT.mkdir(parents=True, exist_ok=True)
    app.mount("/api/assets", StaticFiles(directory=str(SESSION_ROOT)), name="v2-assets")
    app.include_router(api_router)
    return app


app = create_app()
