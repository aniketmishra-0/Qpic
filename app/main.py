"""FastAPI application entrypoint."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import anthropic
import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .routers.crop import router as crop_router
from .routers.rename import router as rename_router
from .routers.tools import router as tools_router
from .utils.file_utils import cleanup_old_jobs, has_pending_jobs

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

logger = logging.getLogger("mcq_cropper")

REQUEST_ID_HEADER = "X-Request-ID"

BASE_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = BASE_DIR / "static"


async def _cleanup_loop(temp_root: Path, older_than_seconds: int) -> None:
    """Periodically delete old temp job directories.

    Idle-aware: when there are no job directories on disk there is nothing to
    clean, so the loop sleeps for a long interval instead of waking the CPU
    every minute. This keeps an open-but-unused desktop app from spinning a
    timer all day (battery-friendly). As soon as a job exists it polls at the
    active cadence so stale jobs are still reaped promptly.
    """

    active_interval = 60
    idle_interval = 600

    while True:
        try:
            deleted = await asyncio.to_thread(cleanup_old_jobs, str(temp_root), older_than_seconds)
            if deleted:
                logger.info("cleanup_deleted=%s temp_root=%s", deleted, temp_root)
            busy = await asyncio.to_thread(has_pending_jobs, str(temp_root))
        except Exception as exc:
            logger.exception("cleanup_failed error=%s", str(exc))
            busy = True
        await asyncio.sleep(active_interval if busy else idle_interval)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: init shared clients and background tasks."""

    settings = Settings()
    temp_root = (BASE_DIR / settings.TEMP_DIR).resolve()
    await asyncio.to_thread(temp_root.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(STATIC_DIR.mkdir, parents=True, exist_ok=True)

    app.state.settings = settings
    app.state.temp_root = str(temp_root)
    api_key = settings.ANTHROPIC_API_KEY
    if api_key:
        app.state.anthropic_client = anthropic.AsyncAnthropic(
            api_key=api_key,
            timeout=httpx.Timeout(settings.API_TIMEOUT_SECONDS),
        )
    else:
        app.state.anthropic_client = None

    cleanup_task = asyncio.create_task(
        _cleanup_loop(temp_root=temp_root, older_than_seconds=settings.CLEANUP_AFTER_SECONDS),
        name="cleanup_old_jobs",
    )

    try:
        yield
    finally:
        cleanup_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await cleanup_task

        client = getattr(app.state, "anthropic_client", None)
        if client is not None:
            close_fn = getattr(client, "aclose", None) or getattr(client, "close", None)
            if close_fn is not None:
                try:
                    result = close_fn()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    logger.exception("Failed to close anthropic client")


app = FastAPI(
    title="Qpic",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next: Any):
    """Attach request ID and log request/response summary."""

    request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
    request.state.request_id = request_id

    start = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        duration_ms = int((time.perf_counter() - start) * 1000)
        status_code = getattr(response, "status_code", 500)
        logger.info(
            "request_id=%s method=%s path=%s status=%s duration_ms=%s",
            request_id,
            request.method,
            request.url.path,
            status_code,
            duration_ms,
        )
        if response is not None:
            response.headers[REQUEST_ID_HEADER] = request_id


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return JSON for any unhandled exception."""

    request_id = getattr(request.state, "request_id", None)
    logger.exception("request_id=%s unhandled_error=%s", request_id, str(exc))
    headers = {}
    if request_id:
        headers[REQUEST_ID_HEADER] = request_id
    return JSONResponse(status_code=500, content={"detail": str(exc)}, headers=headers)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Serve the minimal upload UI."""

    # The whole UI (markup + JS) lives in this one file, so never let the
    # browser serve a stale copy after an update — always revalidate.
    return FileResponse(
        str(STATIC_DIR / "index.html"),
        media_type="text/html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/edit", include_in_schema=False)
async def edit_page() -> FileResponse:
    """Serve the full-screen Acrobat-style document editor."""

    return FileResponse(
        str(STATIC_DIR / "edit.html"),
        media_type="text/html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


app.include_router(crop_router, prefix="/api")
app.include_router(rename_router, prefix="/api")
app.include_router(tools_router, prefix="/api")
