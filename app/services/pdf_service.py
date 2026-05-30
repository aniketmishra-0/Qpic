"""PDF validation and rendering helpers."""

from __future__ import annotations

import logging

import fitz
from fastapi import HTTPException, status
from PIL import Image, ImageDraw

from ..config import Settings

logger = logging.getLogger(__name__)

ERR_INVALID_PDF = "Invalid PDF file"


def validate_pdf(file_bytes: bytes, settings: Settings) -> None:
    """Validate PDF bytes and enforce size/page limits.

    Raises HTTPException if:
    - File is not a valid PDF
    - File exceeds MAX_PDF_SIZE_MB
    - Page count exceeds MAX_PAGES
    """

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > settings.MAX_PDF_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"PDF exceeds size limit ({size_mb:.2f}MB > {settings.MAX_PDF_SIZE_MB}MB)",
        )

    if not file_bytes.startswith(b"%PDF"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=ERR_INVALID_PDF)

    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            page_count = doc.page_count
    except (fitz.FileDataError, fitz.FileNotFoundError, ValueError) as exc:
        logger.error("pdf_open_failed error=%s", str(exc))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=ERR_INVALID_PDF) from exc

    if page_count > settings.MAX_PAGES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"PDF exceeds page limit ({page_count} > {settings.MAX_PAGES})",
        )


def pdf_to_images(file_bytes: bytes, dpi: int) -> list[Image.Image]:
    """Convert PDF bytes to list of PIL Images.

    Uses fitz.open(stream=...) and does not write the PDF to disk.
    Renders using RGB colorspace.
    """

    images: list[Image.Image] = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(img)

    return images


def render_page_region(
    doc: "fitz.Document",
    page_index: int,
    *,
    x_start_pct: float,
    x_end_pct: float,
    y_start_pct: float,
    y_end_pct: float,
    dpi: int,
    furniture_rects: "list | None" = None,
) -> Image.Image:
    """Render a sub-region of a PDF page straight from the vector source.

    The region is given as percentages of the page size so it can be derived
    from the same detection coordinates used everywhere else. Rendering from the
    PDF (rather than cropping an already-rasterized page image) means the output
    is sharp at whatever ``dpi`` we choose, so zooming into a question/solution
    crop never shows the soft, upscaled pixels of the detection render.

    ``furniture_rects`` is an optional list of ``(x0, y0, x1, y1)`` rectangles in
    PDF points marking page furniture (branding footers, logos, decorative
    rules). Any part of them that lands inside the rendered region is painted
    white, so furniture that sits *inside* a crop — e.g. a footer in the middle
    of a cross-page solution — is physically removed from the output.
    """

    page = doc.load_page(page_index)
    rect = page.rect
    page_w = float(rect.width)
    page_h = float(rect.height)

    x0 = rect.x0 + (x_start_pct / 100.0) * page_w
    x1 = rect.x0 + (x_end_pct / 100.0) * page_w
    y0 = rect.y0 + (y_start_pct / 100.0) * page_h
    y1 = rect.y0 + (y_end_pct / 100.0) * page_h

    # Guard against inverted / empty clips.
    if x1 <= x0:
        x0, x1 = rect.x0, rect.x1
    if y1 <= y0:
        y0, y1 = rect.y0, rect.y1

    clip = fitz.Rect(x0, y0, x1, y1)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, clip=clip, colorspace=fitz.csRGB, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    if furniture_rects:
        img = _paint_out_furniture(
            img,
            furniture_rects,
            clip_x0=x0,
            clip_y0=y0,
            zoom=zoom,
        )

    return img


def _paint_out_furniture(
    img: Image.Image,
    furniture_rects: list,
    *,
    clip_x0: float,
    clip_y0: float,
    zoom: float,
) -> Image.Image:
    """Paint white over furniture rectangles intersecting the rendered region.

    ``furniture_rects`` are ``(x0, y0, x1, y1)`` tuples in PDF points; the clip
    origin and ``zoom`` map them onto the rendered pixel grid.
    """

    from ..services.detector.furniture import paint_pad_pts

    pad = paint_pad_pts()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for fr in furniture_rects:
        fx0, fy0, fx1, fy1 = float(fr[0]), float(fr[1]), float(fr[2]), float(fr[3])
        # Grow slightly to cover anti-aliased edges, then map points -> pixels.
        px0 = int((fx0 - pad - clip_x0) * zoom)
        py0 = int((fy0 - pad - clip_y0) * zoom)
        px1 = int((fx1 + pad - clip_x0) * zoom)
        py1 = int((fy1 + pad - clip_y0) * zoom)
        # Clamp to image bounds; skip if no overlap.
        px0 = max(0, min(w, px0))
        px1 = max(0, min(w, px1))
        py0 = max(0, min(h, py0))
        py1 = max(0, min(h, py1))
        if px1 <= px0 or py1 <= py0:
            continue
        draw.rectangle([px0, py0, px1 - 1, py1 - 1], fill=(255, 255, 255))

    return img
