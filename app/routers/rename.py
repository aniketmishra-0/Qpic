"""Standalone batch-rename API endpoints.

The Rename Batch tool is independent of the cropper: a user uploads images,
chooses a naming pattern (with a ``#`` number token), start value and zero
padding, then downloads a ZIP of the renamed files. Image bytes are copied
verbatim, so each file keeps its exact original format.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse

from ..config import Settings
from ..dependencies import get_settings
from ..models.schemas import RenamePlanItem, RenamePreviewResponse
from ..services.rename_service import ALLOWED_EXTENSIONS, plan_renames, split_extension, write_rename_zip
from ..utils.file_utils import create_job_dir, generate_job_id, safe_cleanup_job

router = APIRouter(tags=["rename"])
logger = logging.getLogger(__name__)

RENAME_ZIP = "qpic_renamed_{job_id}.zip"


def _get_temp_root(request: Request, settings: Settings) -> str:
    return str(getattr(request.app.state, "temp_root", settings.TEMP_DIR))


def _is_allowed(filename: str) -> bool:
    _, ext = split_extension(filename)
    return ext in ALLOWED_EXTENSIONS


@router.post("/rename/preview", response_model=RenamePreviewResponse)
async def rename_preview(
    names: list[str] = Form(
        ...,
        description="Original filenames, in the order they should be numbered.",
    ),
    pattern: str = Form("#", description="Naming pattern; '#' is replaced by the running number."),
    start: int = Form(1, ge=0, le=1_000_000),
    padding: int = Form(0, ge=0, le=12),
) -> RenamePreviewResponse:
    """Preview the new names for a list of originals — no files uploaded.

    Lets the UI show a live before/after list as the user tweaks the pattern,
    start number and padding, without shipping the (potentially large) image
    bytes on every keystroke.
    """

    clean = [n for n in names if n and n.strip()]
    if not clean:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No filenames supplied.")

    new_names = plan_renames(clean, pattern=pattern, start=start, padding=padding)
    items = [
        RenamePlanItem(original=o, renamed=n) for o, n in zip(clean, new_names)
    ]
    return RenamePreviewResponse(count=len(items), items=items)


@router.post("/rename")
async def rename_batch(
    request: Request,
    files: list[UploadFile] = File(..., description="Images to rename."),
    pattern: str = Form("#"),
    start: int = Form(1, ge=0, le=1_000_000),
    padding: int = Form(0, ge=0, le=12),
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """Rename uploaded images and return them as a downloadable ZIP.

    The number token ``#`` in ``pattern`` is replaced by a running number from
    ``start``, zero-padded to ``padding`` digits. Each image is written into the
    ZIP byte-for-byte under its new name, so the original format is preserved.
    """

    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files uploaded.")

    max_bytes = settings.MAX_PDF_SIZE_MB * 1024 * 1024
    payload: list[tuple[str, bytes]] = []
    total = 0
    for upload in files:
        name = upload.filename or ""
        if not _is_allowed(name):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type: {name or 'unnamed'}. Images only.",
            )
        raw = await upload.read()
        total += len(raw)
        if total > max_bytes * 4:
            # Generous combined cap (4x the single-PDF limit) so a big batch is
            # allowed without letting an unbounded upload exhaust disk/memory.
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Total upload is too large. Try fewer images at a time.",
            )
        payload.append((name, raw))

    job_id = generate_job_id()
    temp_root = _get_temp_root(request, settings)
    request_id = getattr(request.state, "request_id", None)

    job_dir: Optional[Path] = None
    try:
        job_dir = await asyncio.to_thread(create_job_dir, job_id, temp_root)
        zip_path = job_dir / RENAME_ZIP.format(job_id=job_id)
        count = await asyncio.to_thread(
            write_rename_zip,
            zip_path,
            payload,
            pattern=pattern,
            start=start,
            padding=padding,
        )
        logger.info(
            "request_id=%s job_id=%s stage=rename_done files=%s", request_id, job_id, count
        )
        return FileResponse(
            str(zip_path),
            media_type="application/zip",
            filename="renamed_images.zip",
        )
    except HTTPException:
        if job_dir is not None:
            await asyncio.to_thread(safe_cleanup_job, job_id, temp_root)
        raise
    except Exception as exc:
        logger.exception("request_id=%s job_id=%s stage=rename_error error=%s", request_id, job_id, str(exc))
        if job_dir is not None:
            await asyncio.to_thread(safe_cleanup_job, job_id, temp_root)
        raise
