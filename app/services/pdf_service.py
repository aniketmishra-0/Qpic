"""PDF validation and rendering helpers."""

from __future__ import annotations

import logging
import threading
from typing import Iterator, List, Union

import fitz
from fastapi import HTTPException, status
from PIL import Image, ImageDraw

from ..config import Settings

logger = logging.getLogger(__name__)

ERR_INVALID_PDF = "Invalid PDF file"


def validate_pdf(
    file_bytes: bytes,
    settings: Settings,
    *,
    max_size_mb: int | None = None,
    max_pages: int | None = None,
) -> None:
    """Validate PDF bytes and enforce size/page limits.

    By default the cropper limits (``MAX_PDF_SIZE_MB`` / ``MAX_PAGES``) apply.
    Callers that do cheap, non-AI work (the Compress/Edit/Preflight tools) pass
    their own, much larger ceilings via ``max_size_mb`` / ``max_pages``.

    Raises HTTPException if:
    - File is not a valid PDF
    - File exceeds the effective size limit
    - Page count exceeds the effective page limit
    """

    size_limit = max_size_mb if max_size_mb is not None else settings.MAX_PDF_SIZE_MB
    page_limit = max_pages if max_pages is not None else settings.MAX_PAGES

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > size_limit:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"PDF exceeds size limit ({size_mb:.2f}MB > {size_limit}MB)",
        )

    if not file_bytes.startswith(b"%PDF"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=ERR_INVALID_PDF)

    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            page_count = doc.page_count
    except (fitz.FileDataError, fitz.FileNotFoundError, ValueError) as exc:
        logger.error("pdf_open_failed error=%s", str(exc))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=ERR_INVALID_PDF) from exc

    if page_count > page_limit:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"PDF exceeds page limit ({page_count} > {page_limit})",
        )


def render_page_image(doc: "fitz.Document", page_index: int, dpi: int) -> Image.Image:
    """Rasterize a single page of an open PDF to a PIL Image (RGB)."""

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    page = doc.load_page(page_index)
    pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def pdf_to_images(file_bytes: bytes, dpi: int) -> list[Image.Image]:
    """Convert PDF bytes to a list of PIL Images.

    Eager renderer kept for callers (and tests) that genuinely want every page
    materialised at once. The detection path uses :class:`LazyPageImages`
    instead so a searchable PDF renders zero pages and a scanned one holds at
    most a single page in memory.
    """

    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        return [render_page_image(doc, i, dpi) for i in range(doc.page_count)]


class LazyPageImages:
    """A list-like view of a PDF's pages that renders each page on demand.

    The detection pipeline only needs page *bitmaps* for the OCR and AI tiers;
    the text tier (which wins on every searchable/digital PDF) never touches
    them. Eagerly rasterising the whole document up front therefore burns CPU
    and holds hundreds of megabytes of bitmaps in RAM that are usually thrown
    away unused — wasteful on battery and memory both.

    This view behaves like ``list[Image.Image]`` for the access patterns the
    detectors use (``len()``, ``for img in pages``, ``pages[i]``, ``pages[a:b]``,
    ``enumerate(pages, start=1)``) but renders a page only when it is actually
    requested. By default each rendered page is released as soon as the next one
    is fetched during iteration, so a 100-page scan peaks at ~one page of
    bitmap instead of all 100. Slices (used by the AI tier's batching) and
    explicit indexing render just the pages asked for.

    Thread-safe: rendering is guarded by a lock because detectors run inside
    ``asyncio.to_thread`` worker threads.
    """

    def __init__(self, file_bytes: bytes, dpi: int, *, cache: bool = False) -> None:
        self._doc = fitz.open(stream=file_bytes, filetype="pdf")
        self._dpi = dpi
        self._count = self._doc.page_count
        self._lock = threading.Lock()
        # When cache=False (default) we keep at most the most-recently rendered
        # page, so iteration never accumulates bitmaps. cache=True keeps every
        # page (used only where a caller really needs random repeat access).
        self._cache = cache
        self._store: dict[int, Image.Image] = {}

    def __len__(self) -> int:
        return self._count

    def _render(self, index: int) -> Image.Image:
        if index < 0:
            index += self._count
        if index < 0 or index >= self._count:
            raise IndexError(index)
        with self._lock:
            cached = self._store.get(index)
            if cached is not None:
                return cached
            img = render_page_image(self._doc, index, self._dpi)
            if self._cache:
                self._store[index] = img
            else:
                # Keep only this page so sequential iteration stays flat in RAM.
                self._store = {index: img}
            return img

    def __getitem__(
        self, key: Union[int, slice]
    ) -> Union[Image.Image, List[Image.Image]]:
        if isinstance(key, slice):
            return [self._render(i) for i in range(*key.indices(self._count))]
        return self._render(int(key))

    def __iter__(self) -> Iterator[Image.Image]:
        for i in range(self._count):
            yield self._render(i)

    def close(self) -> None:
        with self._lock:
            self._store.clear()
            try:
                self._doc.close()
            except Exception:
                pass

    def __enter__(self) -> "LazyPageImages":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


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
