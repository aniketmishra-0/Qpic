"""Edit a PDF — text replacement that preserves the original font + OCR.

Two capabilities power the Edit tool:

1. **Extract editable text.** Every text span on a page is returned with its
   exact geometry (bbox), font name, size and colour. The UI shows these as
   editable fields laid over a page preview, so the user edits text *in place*.

2. **Apply edits font-matched.** For each changed span we redact (physically
   remove) the original glyphs in that bbox, then re-insert the new text using
   the *same embedded font* (extracted from the PDF and re-registered), at the
   same size and colour, fitted to the original box. When the original font
   can't be reused (e.g. a non-embedded Base-14 font) we fall back to the
   closest standard font so the look is preserved.

3. **OCR.** A scanned/image PDF has no text to edit. ``ocr_pdf`` runs Tesseract
   over each page and lays an invisible, selectable text layer behind the image
   — turning a scan into a searchable PDF whose words can then be extracted and
   edited like any digital PDF.

All functions operate on bytes and are CPU-bound; call them inside
``asyncio.to_thread``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import fitz

from ..detector.tesseract_locator import configure_tesseract, resolve_languages

logger = logging.getLogger(__name__)


# Map common embedded/Base-14 font basenames to PyMuPDF's built-in aliases, so a
# span set in "Helvetica-Bold" still re-inserts in a bold sans face even when
# the program bytes aren't embedded to reuse.
_BASE14_ALIASES = {
    "helvetica": "helv",
    "arial": "helv",
    "helvetica-bold": "hebo",
    "arial-bold": "hebo",
    "helvetica-oblique": "heit",
    "helvetica-italic": "heit",
    "times": "tiro",
    "times-roman": "tiro",
    "timesnewroman": "tiro",
    "times-bold": "tibo",
    "times-italic": "tiit",
    "times-bolditalic": "tibi",
    "courier": "cour",
    "courier-bold": "cobo",
    "courier-oblique": "coit",
    "symbol": "symb",
    "zapfdingbats": "zadb",
}

# Text-span style flags reported by PyMuPDF's ``get_text("dict")``.
_FLAG_SUPERSCRIPT = 1
_FLAG_ITALIC = 2
_FLAG_SERIFED = 4
_FLAG_BOLD = 16


@dataclass
class EditableSpan:
    """One run of text on a page, addressable for editing."""

    id: str  # stable address: "p{page}_b{block}_l{line}_s{span}"
    page: int
    text: str
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 in PDF points
    font: str
    size: float
    color: int  # packed sRGB int (0xRRGGBB)
    flags: int
    bold: bool
    italic: bool


@dataclass
class PageGeometry:
    page: int
    width: float
    height: float


@dataclass
class ExtractResult:
    pages: list[PageGeometry] = field(default_factory=list)
    spans: list[EditableSpan] = field(default_factory=list)
    has_text: bool = True


def _color_to_int(color: Any) -> int:
    """Normalize a span colour (already a packed int in get_text('dict'))."""

    try:
        return int(color) & 0xFFFFFF
    except (TypeError, ValueError):
        return 0


def extract_text_spans(file_bytes: bytes) -> ExtractResult:
    """Return every editable text span across the document, with geometry.

    Spans are addressed by a stable id derived from their block/line/span index
    so the apply step can locate the exact same run again.
    """

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        result = ExtractResult()
        total_chars = 0
        for pno in range(doc.page_count):
            page = doc.load_page(pno)
            rect = page.rect
            result.pages.append(PageGeometry(page=pno + 1, width=rect.width, height=rect.height))

            data = page.get_text("dict")
            for bi, block in enumerate(data.get("blocks", [])):
                if block.get("type", 0) != 0:
                    continue  # image block — not editable text
                for li, line in enumerate(block.get("lines", [])):
                    for si, span in enumerate(line.get("spans", [])):
                        text = span.get("text", "")
                        if not text.strip():
                            continue
                        total_chars += len(text)
                        flags = int(span.get("flags", 0))
                        result.spans.append(EditableSpan(
                            id=f"p{pno + 1}_b{bi}_l{li}_s{si}",
                            page=pno + 1,
                            text=text,
                            bbox=tuple(round(v, 2) for v in span.get("bbox", (0, 0, 0, 0))),  # type: ignore[arg-type]
                            font=span.get("font", ""),
                            size=round(float(span.get("size", 0.0)), 2),
                            color=_color_to_int(span.get("color", 0)),
                            flags=flags,
                            bold=bool(flags & _FLAG_BOLD),
                            italic=bool(flags & _FLAG_ITALIC),
                        ))
        result.has_text = total_chars > 0
        return result
    finally:
        doc.close()


def _resolve_font_for_span(
    doc: "fitz.Document",
    page: "fitz.Page",
    span: dict,
    *,
    font_cache: dict,
) -> tuple[str, Optional[bytes]]:
    """Pick the best (fontname, fontbuffer) to re-insert a span's text.

    Preference order:
      1. Reuse the span's *embedded* font program (true font match) by extracting
         its bytes from the PDF and registering them under a private name.
      2. Map a Base-14 / well-known font name to PyMuPDF's built-in alias.
      3. Synthesize from style flags (serif/bold/italic) as a last resort.
    """

    raw_name = (span.get("font") or "").strip()
    key = raw_name.lower()
    # Strip a subset prefix like "ABCDEF+Arial".
    if "+" in key:
        key = key.split("+", 1)[1]

    # 1. Try to reuse the embedded program bytes for this exact font.
    if raw_name in font_cache:
        return font_cache[raw_name]

    buffer = _find_embedded_font_buffer(doc, raw_name)
    if buffer:
        alias = f"emb_{abs(hash(raw_name)) % 10_000_000}"
        font_cache[raw_name] = (alias, buffer)
        return alias, buffer

    # 2. Base-14 / common alias.
    if key in _BASE14_ALIASES:
        font_cache[raw_name] = (_BASE14_ALIASES[key], None)
        return _BASE14_ALIASES[key], None

    # 3. Synthesize from flags.
    flags = int(span.get("flags", 0))
    serif = bool(flags & _FLAG_SERIFED)
    bold = bool(flags & _FLAG_BOLD)
    italic = bool(flags & _FLAG_ITALIC)
    if serif:
        alias = {"00": "tiro", "10": "tibo", "01": "tiit", "11": "tibi"}[f"{int(bold)}{int(italic)}"]
    else:
        alias = {"00": "helv", "10": "hebo", "01": "heit", "11": "hebo"}[f"{int(bold)}{int(italic)}"]
    font_cache[raw_name] = (alias, None)
    return alias, None


def _find_embedded_font_buffer(doc: "fitz.Document", font_name: str) -> Optional[bytes]:
    """Extract the embedded program bytes for ``font_name`` if present."""

    if not font_name:
        return None
    target = font_name.split("+", 1)[-1].lower()
    for pno in range(doc.page_count):
        for f in doc.get_page_fonts(pno, full=False):
            xref, ext, _ftype, basefont = f[0], f[1], f[2], f[3]
            if ext == "n/a":
                continue  # not embedded
            if (basefont or "").split("+", 1)[-1].lower() != target:
                continue
            try:
                _name, _ext, _ftype2, buf = doc.extract_font(xref)
                if buf:
                    return buf
            except Exception:
                continue
    return None


@dataclass
class EditOp:
    """A single span edit: replace the text within ``bbox`` on ``page``."""

    page: int
    bbox: tuple[float, float, float, float]
    new_text: str
    # Optional style overrides; when omitted the original span's are reused.
    font: Optional[str] = None
    size: Optional[float] = None
    color: Optional[int] = None


@dataclass
class Operation:
    """A single Acrobat-style edit operation on a page.

    ``type`` selects what happens inside ``bbox`` (PDF points, origin top-left):

      * ``edit_text``  — remove the text under the box and redraw ``text`` in the
        matched/overridden font, size, colour, weight.
      * ``add_text``   — draw ``text`` in a fresh box (no redaction first).
      * ``add_image``  — paste a raster image (``image_b64`` data-URL/base64)
        scaled into the box.
      * ``add_link``   — make the box a clickable hyperlink to ``url``; if
        ``text`` is given it is drawn as the visible (blue, underlined) label.
      * ``erase``      — paint the box with ``fill`` colour (default white) to
        white-out / cover existing content.
    """

    type: str
    page: int
    bbox: tuple[float, float, float, float]
    text: str = ""
    font: Optional[str] = None
    size: Optional[float] = None
    color: Optional[int] = None
    bold: bool = False
    italic: bool = False
    align: int = 0  # 0 left, 1 center, 2 right
    image_b64: Optional[str] = None
    url: Optional[str] = None
    fill: Optional[int] = None  # background/erase colour as packed sRGB int


def _builtin_font(*, serif: bool, bold: bool, italic: bool) -> str:
    """Return the PyMuPDF Base-14 alias matching a style combination."""

    if serif:
        return {"00": "tiro", "10": "tibo", "01": "tiit", "11": "tibi"}[f"{int(bold)}{int(italic)}"]
    return {"00": "helv", "10": "hebo", "01": "heit", "11": "hebo"}[f"{int(bold)}{int(italic)}"]


def _decode_image(image_b64: str) -> bytes:
    """Decode a data-URL or bare base64 string into raw image bytes."""

    import base64

    s = image_b64 or ""
    if s.startswith("data:"):
        _, _, s = s.partition(",")
    return base64.b64decode(s)


def apply_operations(file_bytes: bytes, ops: list[Operation]) -> bytes:
    """Apply a mixed list of Acrobat-style operations and return new PDF bytes.

    Operations are grouped per page. Text edits are redacted first (all on the
    page at once) so the removal can't clip later insertions, then every
    insertion (edited text, new text, images, links, erases) is painted on top.
    """

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        by_page: dict[int, list[Operation]] = {}
        for op in ops:
            by_page.setdefault(op.page, []).append(op)

        font_cache: dict[str, tuple[str, Optional[bytes]]] = {}

        for pno, page_ops in by_page.items():
            if pno < 1 or pno > doc.page_count:
                continue
            page = doc.load_page(pno - 1)

            original = page.get_text("dict")
            span_lookup = _index_spans_by_bbox(original)

            # Pass 1 — redact the regions of all text edits + erases up front.
            edit_text_ops = [o for o in page_ops if o.type == "edit_text"]
            erase_ops = [o for o in page_ops if o.type == "erase"]
            src_for_edit: dict[int, dict] = {}
            for o in edit_text_ops:
                src_for_edit[id(o)] = _match_span(span_lookup, o.bbox) or {}
                page.add_redact_annot(fitz.Rect(*o.bbox))
            for o in erase_ops:
                fill = fitz.sRGB_to_pdf(o.fill if o.fill is not None else 0xFFFFFF)
                page.add_redact_annot(fitz.Rect(*o.bbox), fill=fill)
            if edit_text_ops or erase_ops:
                # text=0 removes glyphs in the edit boxes; erase boxes are filled.
                page.apply_redactions(images=2, graphics=1, text=0)

            # Pass 2 — paint every insertion on top.
            for o in page_ops:
                if o.type == "edit_text":
                    if not o.text:
                        continue  # deletion only
                    src = src_for_edit.get(id(o), {})
                    _draw_text_op(doc, page, o, src=src, font_cache=font_cache)
                elif o.type == "add_text":
                    _draw_text_op(doc, page, o, src={}, font_cache=font_cache)
                elif o.type == "add_image":
                    _draw_image_op(page, o)
                elif o.type == "add_link":
                    _draw_link_op(page, o)
                # erase already handled by the redaction fill above.

        return doc.tobytes(garbage=4, deflate=True, clean=True)
    finally:
        doc.close()


def _draw_text_op(
    doc: "fitz.Document",
    page: "fitz.Page",
    op: Operation,
    *,
    src: dict,
    font_cache: dict,
) -> None:
    """Render an ``edit_text`` / ``add_text`` operation into its box."""

    # Resolve font: an explicit bold/italic request forces a synthesized
    # Base-14 face; otherwise reuse the source span's (embedded) font.
    if op.bold or op.italic or op.font:
        flags = int(src.get("flags", 0))
        serif = bool(flags & _FLAG_SERIFED)
        fontname = _builtin_font(serif=serif, bold=op.bold, italic=op.italic)
        fontbuffer = None
        # If a concrete embedded font was named and no weight override, reuse it.
        if op.font and not (op.bold or op.italic):
            fontname, fontbuffer = _resolve_font_for_span(
                doc, page, {"font": op.font, "flags": flags}, font_cache=font_cache
            )
    else:
        fontname, fontbuffer = _resolve_font_for_span(
            doc, page, {"font": src.get("font", ""), "flags": src.get("flags", 0)},
            font_cache=font_cache,
        )

    fontsize = op.size or float(src.get("size", 0) or 11)
    if op.color is not None:
        color_int = op.color
    else:
        color_int = _color_to_int(src.get("color", 0))
    rgb = fitz.sRGB_to_pdf(color_int)

    _insert_fitted_text(
        page,
        rect=fitz.Rect(*op.bbox),
        text=op.text,
        fontname=fontname,
        fontbuffer=fontbuffer,
        fontsize=fontsize,
        color=rgb,
        align=op.align,
        grow=(op.type == "add_text"),
    )


def _draw_image_op(page: "fitz.Page", op: Operation) -> None:
    """Insert a raster image into the operation's box."""

    if not op.image_b64:
        return
    try:
        raw = _decode_image(op.image_b64)
        page.insert_image(fitz.Rect(*op.bbox), stream=raw, keep_proportion=True)
    except Exception as exc:
        logger.warning("add_image_failed error=%s", str(exc))


def _draw_link_op(page: "fitz.Page", op: Operation) -> None:
    """Draw an optional visible label and attach a URI link over the box."""

    rect = fitz.Rect(*op.bbox)
    if op.text:
        # Blue, underlined label so it reads as a link.
        _insert_fitted_text(
            page,
            rect=rect,
            text=op.text,
            fontname=_builtin_font(serif=False, bold=False, italic=False),
            fontbuffer=None,
            fontsize=op.size or 11,
            color=(0.04, 0.32, 0.84),
            align=op.align,
            grow=True,
        )
        try:
            page.draw_line(
                fitz.Point(rect.x0, rect.y1 - 1),
                fitz.Point(rect.x1, rect.y1 - 1),
                color=(0.04, 0.32, 0.84),
                width=0.6,
            )
        except Exception:
            pass
    if op.url:
        try:
            page.insert_link({"kind": fitz.LINK_URI, "from": rect, "uri": op.url})
        except Exception as exc:
            logger.warning("add_link_failed error=%s", str(exc))


def apply_text_edits(file_bytes: bytes, edits: list[EditOp]) -> bytes:
    """Apply font-matched text replacements and return the new PDF bytes.

    Thin wrapper over :func:`apply_operations` kept for the simple text-only
    path (and its tests): each :class:`EditOp` becomes an ``edit_text``
    operation.
    """

    ops = [
        Operation(
            type="edit_text",
            page=e.page,
            bbox=e.bbox,
            text=e.new_text,
            font=e.font,
            size=e.size,
            color=e.color,
        )
        for e in edits
    ]
    return apply_operations(file_bytes, ops)


def _index_spans_by_bbox(text_dict: dict) -> list[dict]:
    """Flatten all spans of a page into a list (with their bbox) for matching."""

    spans: list[dict] = []
    for block in text_dict.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                spans.append(span)
    return spans


def _match_span(spans: list[dict], bbox: tuple[float, float, float, float]) -> Optional[dict]:
    """Return the span whose bbox best overlaps ``bbox`` (highest IoU)."""

    target = fitz.Rect(*bbox)
    best: Optional[dict] = None
    best_iou = 0.0
    for span in spans:
        r = fitz.Rect(*span.get("bbox", (0, 0, 0, 0)))
        inter = r & target
        if inter.is_empty:
            continue
        inter_area = inter.width * inter.height
        union = (r.width * r.height) + (target.width * target.height) - inter_area
        iou = inter_area / union if union > 0 else 0.0
        if iou > best_iou:
            best_iou = iou
            best = span
    return best


def _insert_fitted_text(
    page: "fitz.Page",
    *,
    rect: "fitz.Rect",
    text: str,
    fontname: str,
    fontbuffer: Optional[bytes],
    fontsize: float,
    color: tuple,
    align: int = 0,
    grow: bool = False,
) -> None:
    """Draw ``text`` inside ``rect``, fitting the font to the box.

    Uses ``insert_textbox`` which returns a negative number when the text
    overflows the box. When ``grow`` is False (editing an existing run) we step
    the font size *down* until it fits, so a longer replacement never spills
    outside the original region. When ``grow`` is True (a freshly added box) we
    first try to grow the box downward to honour the requested size, then fall
    back to shrinking — so newly typed text keeps the size the user picked.
    """

    # Register an embedded font buffer once per page under its private name.
    if fontbuffer is not None:
        try:
            page.insert_font(fontname=fontname, fontbuffer=fontbuffer)
        except Exception:
            fontname, fontbuffer = "helv", None

    # Give a little breathing room — a tight bbox leaves no space for
    # ascenders/descenders.
    box = fitz.Rect(rect.x0, rect.y0 - 1, rect.x1 + 2, rect.y1 + 2)
    if grow:
        # Allow the text to extend below the drawn box (down to the page edge)
        # so the chosen size is respected for newly added text.
        box = fitz.Rect(rect.x0, rect.y0 - 1, rect.x1 + 2, page.rect.y1 - 2)

    def _try(size: float) -> bool:
        nonlocal fontname, fontbuffer
        try:
            rc = page.insert_textbox(
                box, text, fontname=fontname, fontsize=size, color=color, align=align,
            )
            return rc >= 0
        except Exception:
            if fontname != "helv":
                fontname, fontbuffer = "helv", None
                return _try(size)
            raise

    size = max(2.0, fontsize)
    floor = 3.0
    while size >= floor:
        if _try(size):
            return
        size -= 0.5

    # Last resort: place a single baseline of text so nothing is silently lost.
    try:
        page.insert_text(
            fitz.Point(rect.x0, rect.y1),
            text,
            fontname=fontname,
            fontsize=floor,
            color=color,
        )
    except Exception:
        logger.warning("insert_text_failed bbox=%s", tuple(rect))


# --- OCR ---------------------------------------------------------------------


@dataclass
class OcrResult:
    data: bytes
    pages_ocred: int
    languages: str
    note: str = ""


def ocr_pdf(file_bytes: bytes, *, languages: str = "eng", dpi: int = 300) -> OcrResult:
    """Add an invisible, selectable text layer to a scanned PDF via Tesseract.

    Each page is rasterised at ``dpi`` and fed to Tesseract, which returns a
    single-page searchable PDF (the image with an invisible text layer behind
    it). Those pages are reassembled into one document. Pages that already carry
    a real text layer are passed through untouched, so a mixed PDF (some digital,
    some scanned) only OCRs the pages that need it.
    """

    import pytesseract
    from PIL import Image

    configure_tesseract()
    lang = resolve_languages(languages or "eng")

    src = fitz.open(stream=file_bytes, filetype="pdf")
    out = fitz.open()
    try:
        pages_ocred = 0
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        for pno in range(src.page_count):
            page = src.load_page(pno)
            existing = (page.get_text("text") or "").strip()
            if len(existing) > 20:
                # Already has selectable text — keep the original page as-is.
                out.insert_pdf(src, from_page=pno, to_page=pno)
                continue

            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
            pil_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            try:
                pdf_bytes = pytesseract.image_to_pdf_or_hocr(
                    pil_img, extension="pdf", lang=lang
                )
            except Exception as exc:
                logger.warning("ocr_page_failed page=%s error=%s", pno + 1, str(exc))
                out.insert_pdf(src, from_page=pno, to_page=pno)
                continue

            with fitz.open(stream=pdf_bytes, filetype="pdf") as ocr_doc:
                out.insert_pdf(ocr_doc)
            pages_ocred += 1

        data = out.tobytes(garbage=4, deflate=True, clean=True)
        note = (
            f"OCR added a searchable text layer to {pages_ocred} page(s)."
            if pages_ocred
            else "Every page already had selectable text — nothing to OCR."
        )
        return OcrResult(data=data, pages_ocred=pages_ocred, languages=lang, note=note)
    finally:
        src.close()
        out.close()
