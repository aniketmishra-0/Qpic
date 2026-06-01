"""PDF power-tool API endpoints: Compress, Edit (+ OCR), and Preflight.

These tools are independent of the cropper. Each accepts a PDF upload, does its
work with PyMuPDF on disk-backed job dirs (so a result can be downloaded later),
and returns a small JSON summary plus a download URL.

The Edit tool is a three-step flow:
    1. POST /tools/edit/open      → stage the PDF, return editable spans + page
       previews.
    2. POST /tools/edit/apply     → apply font-matched text edits to that job.
    3. GET  /tools/edit/download  → download the edited PDF.
OCR (POST /tools/edit/ocr) turns a scanned PDF into a searchable one in place.
"""

from __future__ import annotations

import asyncio
import io
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse

from ..config import Settings
from ..dependencies import get_settings
from ..models.schemas import (
    CompressResponse,
    EditApplyRequest,
    EditApplyResponse,
    EditExtractResponse,
    EditPageModel,
    EditableSpanModel,
    OcrResponse,
    PreflightFixResponse,
    PreflightResponse,
)
from ..services.pdf_service import render_page_image, validate_pdf
from ..services.pdf_tools.compress_service import LEVELS, compress_pdf
from ..services.pdf_tools.edit_service import (
    EditOp,
    Operation,
    apply_operations,
    apply_text_edits,
    extract_text_spans,
    ocr_pdf,
)
from ..services.pdf_tools.preflight_service import normalize_page_sizes, preflight_pdf
from ..utils.file_utils import create_job_dir, generate_job_id, get_job_dir, safe_cleanup_job
from ..utils.image_utils import ensure_rgb

router = APIRouter(tags=["tools"], prefix="/tools")
logger = logging.getLogger(__name__)

ERR_INVALID_FILE_TYPE = "Invalid file type. PDF required."
ERR_INVALID_PDF = "Invalid PDF file"

# Filenames used inside a tool's job dir.
_SOURCE_PDF = "source.pdf"
_COMPRESSED_PDF = "compressed.pdf"
_EDITED_PDF = "edited.pdf"
_OCR_PDF = "searchable.pdf"
_NORMALIZED_PDF = "normalized.pdf"

# Preview render DPI for the editor canvas (high enough to stay crisp when
# zoomed in; the page is downscaled by CSS to fit, so quality holds up).
_EDIT_PREVIEW_DPI = 200


def _get_temp_root(request: Request, settings: Settings) -> str:
    return str(getattr(request.app.state, "temp_root", settings.TEMP_DIR))


async def _read_pdf_upload(file: UploadFile, settings: Settings) -> bytes:
    """Validate the upload is a PDF within limits and return its bytes."""

    if file.content_type not in ("application/pdf", "application/octet-stream"):
        # Some browsers send octet-stream for .pdf; fall through to the magic check.
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=ERR_INVALID_FILE_TYPE)

    file_bytes = await file.read()
    if not file_bytes.startswith(b"%PDF"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=ERR_INVALID_PDF)

    # The standalone tools (Compress/Edit/Preflight) do cheap PyMuPDF work with
    # no per-page AI/OCR, so they get the much larger tool ceilings rather than
    # the cropper's 50MB / 100-page limits.
    validate_pdf(
        file_bytes=file_bytes,
        settings=settings,
        max_size_mb=settings.MAX_TOOLS_PDF_SIZE_MB,
        max_pages=settings.MAX_TOOLS_PAGES,
    )
    return file_bytes


# ============================================================================
#  Compress
# ============================================================================


@router.post("/compress", response_model=CompressResponse)
async def compress_endpoint(
    request: Request,
    file: UploadFile = File(..., description="The PDF to compress."),
    level: str = Form(
        "balanced",
        description="Compression strength: light | balanced | strong | extreme. "
        "Ignored when target_mb is set.",
    ),
    target_mb: Optional[float] = Form(
        None,
        description="Optional target size in MB. When set, the tool pushes quality "
        "down until the file fits (best-effort) and the level is ignored.",
    ),
    settings: Settings = Depends(get_settings),
) -> CompressResponse:
    """Shrink a PDF by recompressing images, subsetting fonts and cleaning streams.

    Drive it either by ``level`` (a quality preset) or by ``target_mb`` (squeeze
    until it fits). The result is stored for download.
    """

    lvl = (level or "balanced").strip().lower()
    if lvl not in LEVELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown level '{level}'. Use one of: {', '.join(LEVELS)}.",
        )
    if target_mb is not None and target_mb <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="target_mb must be greater than 0.",
        )

    file_bytes = await _read_pdf_upload(file, settings)

    job_id = generate_job_id()
    temp_root = _get_temp_root(request, settings)
    request_id = getattr(request.state, "request_id", None)
    job_dir: Optional[Path] = None
    try:
        job_dir = await asyncio.to_thread(create_job_dir, job_id, temp_root)
        result = await asyncio.to_thread(
            compress_pdf, file_bytes, level=lvl, target_mb=target_mb
        )

        out_path = job_dir / _COMPRESSED_PDF
        await asyncio.to_thread(out_path.write_bytes, result.data)

        logger.info(
            "request_id=%s job_id=%s stage=compress_done level=%s original=%s compressed=%s ratio=%.3f",
            request_id, job_id, result.level, result.original_size,
            result.compressed_size, result.ratio,
        )

        return CompressResponse(
            job_id=job_id,
            original_size=result.original_size,
            compressed_size=result.compressed_size,
            ratio=round(result.ratio, 4),
            level=result.level,
            target_met=result.target_met,
            note=result.note,
            download_url=f"/api/tools/compress/download/{job_id}",
        )
    except HTTPException:
        if job_dir is not None:
            await asyncio.to_thread(safe_cleanup_job, job_id, temp_root)
        raise
    except Exception as exc:
        logger.exception("request_id=%s job_id=%s stage=compress_error error=%s", request_id, job_id, str(exc))
        if job_dir is not None:
            await asyncio.to_thread(safe_cleanup_job, job_id, temp_root)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Couldn't compress that PDF.",
        ) from exc


@router.get("/compress/download/{job_id}")
async def compress_download(
    job_id: str,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """Stream a finished compressed PDF."""

    temp_root = _get_temp_root(request, settings)
    path = get_job_dir(job_id, temp_root) / _COMPRESSED_PDF
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Nothing to download.")
    return FileResponse(str(path), media_type="application/pdf", filename="compressed.pdf")


# ============================================================================
#  Preflight
# ============================================================================


@router.post("/preflight", response_model=PreflightResponse)
async def preflight_endpoint(
    file: UploadFile = File(..., description="The PDF to inspect."),
    settings: Settings = Depends(get_settings),
) -> PreflightResponse:
    """Run a read-only print/production-readiness inspection of a PDF."""

    file_bytes = await _read_pdf_upload(file, settings)
    report = await asyncio.to_thread(preflight_pdf, file_bytes)

    return PreflightResponse(
        verdict=report.verdict,
        page_count=report.page_count,
        page_sizes=report.page_sizes,
        file_size=report.file_size,
        is_encrypted=report.is_encrypted,
        has_text_layer=report.has_text_layer,
        checks=[c.__dict__ for c in report.checks],  # type: ignore[misc]
        fonts=[f.__dict__ for f in report.fonts],  # type: ignore[misc]
        images=[i.__dict__ for i in report.images],  # type: ignore[misc]
        metadata=report.metadata,
        distinct_page_sizes=report.distinct_page_sizes,
        mixed_page_sizes=report.mixed_page_sizes,
        page_details=report.page_details,
    )


@router.post("/preflight/fix-page-sizes", response_model=PreflightFixResponse)
async def preflight_fix_page_sizes(
    request: Request,
    file: UploadFile = File(..., description="The PDF whose pages should be normalized."),
    target: str = Form(
        "auto",
        description="Target size: 'auto' (most common page size), 'max' (largest "
        "page, never downscales), a named size: a3 | a4 | a5 | letter | legal | "
        "square, or 'custom:<W_mm>x<H_mm>' for an explicit size in millimetres.",
    ),
    fill_mode: str = Form(
        "fit",
        description="How content is placed: 'fit' (scale proportionally, preserve "
        "aspect ratio) or 'stretch' (fill entire target, may distort).",
    ),
    skip_pages: str = Form(
        "",
        description="Comma-separated page numbers or ranges to skip, e.g. "
        "'2,5,10-12'. These pages are copied through untouched.",
    ),
    settings: Settings = Depends(get_settings),
) -> PreflightFixResponse:
    """Normalize every page to one uniform size (Acrobat-style preflight fix).

    Pages that differ from the chosen target are rebuilt at the target size with
    their content scaled proportionally and centred — text and vectors stay
    crisp. The normalized PDF is stored for download.
    """

    file_bytes = await _read_pdf_upload(file, settings)

    # Parse skip_pages string into a list of 1-indexed page numbers.
    skip_list: list[int] = []
    if skip_pages and skip_pages.strip():
        for part in skip_pages.split(","):
            part = part.strip()
            if "-" in part:
                try:
                    a, b = part.split("-", 1)
                    skip_list.extend(range(int(a.strip()), int(b.strip()) + 1))
                except (ValueError, TypeError):
                    pass
            else:
                try:
                    skip_list.append(int(part))
                except (ValueError, TypeError):
                    pass

    # Validate fill_mode
    fm = (fill_mode or "fit").strip().lower()
    if fm not in ("fit", "stretch"):
        fm = "fit"

    job_id = generate_job_id()
    temp_root = _get_temp_root(request, settings)
    request_id = getattr(request.state, "request_id", None)
    job_dir: Optional[Path] = None
    try:
        job_dir = await asyncio.to_thread(create_job_dir, job_id, temp_root)
        result = await asyncio.to_thread(
            normalize_page_sizes,
            file_bytes,
            target=target,
            fill_mode=fm,
            skip_pages=skip_list or None,
        )

        out_path = job_dir / _NORMALIZED_PDF
        await asyncio.to_thread(out_path.write_bytes, result.data)

        logger.info(
            "request_id=%s job_id=%s stage=preflight_fix_done target=%s changed=%s/%s",
            request_id, job_id, result.target_label, result.pages_changed, result.pages_total,
        )

        return PreflightFixResponse(
            job_id=job_id,
            target_label=result.target_label,
            target_width=round(result.target_width, 2),
            target_height=round(result.target_height, 2),
            pages_total=result.pages_total,
            pages_changed=result.pages_changed,
            note=result.note,
            download_url=f"/api/tools/preflight/download/{job_id}",
        )
    except HTTPException:
        if job_dir is not None:
            await asyncio.to_thread(safe_cleanup_job, job_id, temp_root)
        raise
    except Exception as exc:
        logger.exception("request_id=%s job_id=%s stage=preflight_fix_error error=%s", request_id, job_id, str(exc))
        if job_dir is not None:
            await asyncio.to_thread(safe_cleanup_job, job_id, temp_root)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Couldn't normalize that PDF's page sizes.",
        ) from exc


@router.get("/preflight/download/{job_id}")
async def preflight_download(
    job_id: str,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """Stream a finished normalized PDF."""

    temp_root = _get_temp_root(request, settings)
    path = get_job_dir(job_id, temp_root) / _NORMALIZED_PDF
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Nothing to download.")
    return FileResponse(str(path), media_type="application/pdf", filename="normalized.pdf")


# ============================================================================
#  Edit (open → apply → download) + OCR
# ============================================================================


def _page_geometry(file_bytes: bytes) -> list[tuple[int, float, float]]:
    """Return (page_number, width_pt, height_pt) for each page — no rendering."""

    import fitz

    out: list[tuple[int, float, float]] = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for pno in range(doc.page_count):
            rect = doc.load_page(pno).rect
            out.append((pno + 1, rect.width, rect.height))
    return out


@router.post("/edit/open", response_model=EditExtractResponse)
async def edit_open(
    request: Request,
    file: UploadFile = File(..., description="The PDF to edit."),
    settings: Settings = Depends(get_settings),
) -> EditExtractResponse:
    """Stage a PDF for editing: extract editable text spans and page geometry.

    The PDF is saved under a job dir so subsequent /apply and /download calls
    operate on the same source. Each span is returned with its geometry so the
    UI can lay an editable field over the rendered page. Page previews are
    rendered lazily by ``/edit/{job_id}/page/{n}`` (one page at a time) so
    opening is fast even on a big document instead of base64-ing every page.
    """

    file_bytes = await _read_pdf_upload(file, settings)

    job_id = generate_job_id()
    temp_root = _get_temp_root(request, settings)
    job_dir: Optional[Path] = None
    try:
        job_dir = await asyncio.to_thread(create_job_dir, job_id, temp_root)
        await asyncio.to_thread((job_dir / _SOURCE_PDF).write_bytes, file_bytes)

        extracted = await asyncio.to_thread(extract_text_spans, file_bytes)
        geometry = await asyncio.to_thread(_page_geometry, file_bytes)

        page_models = [
            EditPageModel(
                page=p,
                width=w,
                height=h,
                preview_url=f"/api/tools/edit/{job_id}/page/{p}",
            )
            for (p, w, h) in geometry
        ]
        span_models = [
            EditableSpanModel(
                id=s.id, page=s.page, text=s.text, bbox=list(s.bbox),
                font=s.font, size=s.size, color=s.color, bold=s.bold, italic=s.italic,
            )
            for s in extracted.spans
        ]
        return EditExtractResponse(
            job_id=job_id,
            has_text=extracted.has_text,
            pages=page_models,
            spans=span_models,
        )
    except HTTPException:
        if job_dir is not None:
            await asyncio.to_thread(safe_cleanup_job, job_id, temp_root)
        raise
    except Exception as exc:
        logger.exception("job_id=%s stage=edit_open_error error=%s", job_id, str(exc))
        if job_dir is not None:
            await asyncio.to_thread(safe_cleanup_job, job_id, temp_root)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Couldn't open that PDF for editing.",
        ) from exc


@router.get("/edit/{job_id}/state", response_model=EditExtractResponse)
async def edit_state(
    job_id: str,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> EditExtractResponse:
    """Re-extract spans + geometry for an already-staged edit job.

    Lets the full-screen editor (``/edit``) pick up a PDF that was opened in the
    inline tool, by passing ``?job=<job_id>`` — no re-upload needed.
    """

    temp_root = _get_temp_root(request, settings)
    source = get_job_dir(job_id, temp_root) / _SOURCE_PDF
    if not source.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Edit session not found or expired.")

    file_bytes = await asyncio.to_thread(source.read_bytes)
    try:
        extracted = await asyncio.to_thread(extract_text_spans, file_bytes)
        geometry = await asyncio.to_thread(_page_geometry, file_bytes)
    except Exception as exc:
        logger.exception("job_id=%s stage=edit_state_error error=%s", job_id, str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Couldn't reopen that PDF for editing.",
        ) from exc

    page_models = [
        EditPageModel(
            page=p,
            width=w,
            height=h,
            preview_url=f"/api/tools/edit/{job_id}/page/{p}",
        )
        for (p, w, h) in geometry
    ]
    span_models = [
        EditableSpanModel(
            id=s.id, page=s.page, text=s.text, bbox=list(s.bbox),
            font=s.font, size=s.size, color=s.color, bold=s.bold, italic=s.italic,
        )
        for s in extracted.spans
    ]
    return EditExtractResponse(
        job_id=job_id,
        has_text=extracted.has_text,
        pages=page_models,
        spans=span_models,
    )


def _render_single_page_png(file_bytes: bytes, page_no: int, dpi: int) -> bytes:
    """Render one page (1-indexed) to PNG bytes."""

    import fitz

    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        if page_no < 1 or page_no > doc.page_count:
            raise IndexError(page_no)
        img = render_page_image(doc, page_no - 1, dpi)
    buff = io.BytesIO()
    ensure_rgb(img).save(buff, format="PNG")
    return buff.getvalue()


@router.get("/edit/{job_id}/page/{page_no}")
async def edit_page_preview(
    job_id: str,
    page_no: int,
    request: Request,
    settings: Settings = Depends(get_settings),
):
    """Render a single page of a staged edit job to a PNG (lazy preview)."""

    from fastapi.responses import Response

    temp_root = _get_temp_root(request, settings)
    source = get_job_dir(job_id, temp_root) / _SOURCE_PDF
    if not source.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Edit session not found or expired.")

    file_bytes = await asyncio.to_thread(source.read_bytes)
    try:
        png = await asyncio.to_thread(_render_single_page_png, file_bytes, page_no, _EDIT_PREVIEW_DPI)
    except IndexError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Page out of range.")
    return Response(content=png, media_type="image/png", headers={"Cache-Control": "max-age=3600"})


@router.post("/edit/apply", response_model=EditApplyResponse)
async def edit_apply(
    payload: EditApplyRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> EditApplyResponse:
    """Apply Acrobat-style edits (text/add-text/image/link/erase) to a job PDF."""

    if not payload.operations and not payload.edits:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No edits supplied.")

    temp_root = _get_temp_root(request, settings)
    job_dir = get_job_dir(payload.job_id, temp_root)
    source = job_dir / _SOURCE_PDF
    if not source.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Edit session not found or expired.")

    file_bytes = await asyncio.to_thread(source.read_bytes)

    # Prefer the rich operations list; fall back to the legacy text-only edits.
    if payload.operations:
        ops = [
            Operation(
                type=o.type,
                page=o.page,
                bbox=(o.bbox[0], o.bbox[1], o.bbox[2], o.bbox[3]),
                text=o.text,
                font=o.font,
                size=o.size,
                color=o.color,
                bold=o.bold,
                italic=o.italic,
                align=o.align,
                image_b64=o.image_b64,
                url=o.url,
                fill=o.fill,
            )
            for o in payload.operations
            if len(o.bbox) == 4
        ]
        if not ops:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No valid operations (bbox must have 4 values).")
        try:
            edited = await asyncio.to_thread(apply_operations, file_bytes, ops)
        except Exception as exc:
            logger.exception("job_id=%s stage=edit_apply_error error=%s", payload.job_id, str(exc))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Couldn't apply those edits.",
            ) from exc
        applied = len(ops)
    else:
        edits = [
            EditOp(
                page=e.page,
                bbox=(e.bbox[0], e.bbox[1], e.bbox[2], e.bbox[3]),
                new_text=e.new_text,
                font=e.font,
                size=e.size,
                color=e.color,
            )
            for e in payload.edits
            if len(e.bbox) == 4
        ]
        if not edits:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No valid edits (bbox must have 4 values).")
        try:
            edited = await asyncio.to_thread(apply_text_edits, file_bytes, edits)
        except Exception as exc:
            logger.exception("job_id=%s stage=edit_apply_error error=%s", payload.job_id, str(exc))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Couldn't apply those edits.",
            ) from exc
        applied = len(edits)

    out_path = job_dir / _EDITED_PDF
    await asyncio.to_thread(out_path.write_bytes, edited)
    logger.info("job_id=%s stage=edit_apply_done ops=%s", payload.job_id, applied)

    return EditApplyResponse(
        job_id=payload.job_id,
        edits_applied=applied,
        download_url=f"/api/tools/edit/download/{payload.job_id}",
    )


@router.get("/edit/download/{job_id}")
async def edit_download(
    job_id: str,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """Stream the edited (or OCR'd) PDF for a job, preferring the most recent."""

    temp_root = _get_temp_root(request, settings)
    job_dir = get_job_dir(job_id, temp_root)
    for candidate, name in ((_EDITED_PDF, "edited.pdf"), (_OCR_PDF, "searchable.pdf")):
        path = job_dir / candidate
        if path.exists():
            return FileResponse(str(path), media_type="application/pdf", filename=name)
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Nothing to download.")


@router.post("/edit/ocr", response_model=OcrResponse)
async def edit_ocr(
    request: Request,
    file: UploadFile = File(..., description="A scanned/image PDF to make searchable."),
    languages: str = Form(
        "",
        description="Tesseract language spec, e.g. 'eng' or 'eng+hin'. "
        "Defaults to the server's configured OCR languages.",
    ),
    dpi: int = Form(300, ge=150, le=600, description="Rasterization DPI for OCR."),
    settings: Settings = Depends(get_settings),
) -> OcrResponse:
    """Add an invisible, selectable OCR text layer to a scanned PDF.

    Pages that already have selectable text are passed through untouched. The
    result is a searchable PDF whose text can then be opened in the Edit tool.
    """

    file_bytes = await _read_pdf_upload(file, settings)
    lang = (languages or getattr(settings, "OCR_LANGUAGES", "eng") or "eng").strip()

    job_id = generate_job_id()
    temp_root = _get_temp_root(request, settings)
    job_dir: Optional[Path] = None
    try:
        job_dir = await asyncio.to_thread(create_job_dir, job_id, temp_root)
        # Keep the source so the OCR'd file can be opened in Edit afterwards.
        await asyncio.to_thread((job_dir / _SOURCE_PDF).write_bytes, file_bytes)

        result = await asyncio.to_thread(ocr_pdf, file_bytes, languages=lang, dpi=dpi)
        out_path = job_dir / _OCR_PDF
        await asyncio.to_thread(out_path.write_bytes, result.data)
        # Make the searchable file the new editable source too.
        await asyncio.to_thread((job_dir / _SOURCE_PDF).write_bytes, result.data)

        logger.info("job_id=%s stage=ocr_done pages=%s lang=%s", job_id, result.pages_ocred, result.languages)
        return OcrResponse(
            job_id=job_id,
            pages_ocred=result.pages_ocred,
            languages=result.languages,
            note=result.note,
            download_url=f"/api/tools/edit/download/{job_id}",
        )
    except HTTPException:
        if job_dir is not None:
            await asyncio.to_thread(safe_cleanup_job, job_id, temp_root)
        raise
    except Exception as exc:
        logger.exception("job_id=%s stage=ocr_error error=%s", job_id, str(exc))
        if job_dir is not None:
            await asyncio.to_thread(safe_cleanup_job, job_id, temp_root)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Couldn't OCR that PDF.",
        ) from exc
