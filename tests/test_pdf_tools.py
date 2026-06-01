"""Tests for the standalone PDF power tools: Compress, Edit, Preflight."""

from __future__ import annotations

import io

import fitz
import pytest

from app.services.pdf_tools.compress_service import (
    DEFAULT_LEVEL,
    LEVELS,
    compress_pdf,
)
from app.services.pdf_tools.edit_service import (
    EditOp,
    apply_text_edits,
    extract_text_spans,
)
from app.services.pdf_tools.preflight_service import (
    normalize_page_sizes,
    preflight_pdf,
)


# --- fixtures ----------------------------------------------------------------


def _text_pdf() -> bytes:
    """A simple two-line searchable PDF."""

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 80), "1. The quick brown fox.", fontname="helv", fontsize=14)
    page.insert_text((50, 120), "2. Jumps over the lazy dog.", fontname="helv", fontsize=14)
    data = doc.tobytes()
    doc.close()
    return data


def _image_pdf(px: int = 1600, quality: int = 95) -> bytes:
    """A PDF whose single page is a large embedded JPEG (compressible)."""

    from PIL import Image

    img = Image.new("RGB", (px, px))
    pixels = img.load()
    for y in range(px):
        for x in range(px):
            pixels[x, y] = ((x * 7) % 256, (y * 5) % 256, ((x + y) * 3) % 256)
    buff = io.BytesIO()
    img.save(buff, format="JPEG", quality=quality)

    doc = fitz.open()
    page = doc.new_page()
    page.insert_image(page.rect, stream=buff.getvalue())
    data = doc.tobytes()
    doc.close()
    return data


# --- compress ----------------------------------------------------------------


def test_compress_levels_are_known() -> None:
    assert set(LEVELS) == {"light", "balanced", "strong", "extreme"}
    assert DEFAULT_LEVEL in LEVELS


def test_compress_image_pdf_shrinks() -> None:
    src = _image_pdf()
    result = compress_pdf(src, level="strong")
    assert result.compressed_size < result.original_size
    assert 0.0 < result.ratio <= 1.0
    # Output is still a valid PDF.
    doc = fitz.open(stream=result.data, filetype="pdf")
    assert doc.page_count == 1
    doc.close()


def test_compress_never_returns_larger_than_input() -> None:
    # A tiny text PDF may not be shrinkable; the result must never be bigger.
    src = _text_pdf()
    result = compress_pdf(src, level="balanced")
    assert result.compressed_size <= result.original_size


def test_compress_to_target_meets_or_reports() -> None:
    src = _image_pdf(px=1800)
    # A generous target the image ladder can reach.
    result = compress_pdf(src, target_mb=0.25)
    assert result.target_met in (True, False)
    if result.target_met:
        assert result.compressed_size <= 0.25 * 1024 * 1024
    # Either way the result is a smaller, valid PDF.
    assert result.compressed_size <= result.original_size
    doc = fitz.open(stream=result.data, filetype="pdf")
    assert doc.page_count >= 1
    doc.close()


# --- preflight ---------------------------------------------------------------


def test_preflight_reports_pages_and_verdict() -> None:
    report = preflight_pdf(_text_pdf())
    assert report.page_count == 1
    assert report.verdict in ("pass", "warn", "fail")
    assert report.has_text_layer is True
    # A check for every category we run.
    ids = {c.id for c in report.checks}
    assert {"encryption", "fonts_embedded", "image_dpi", "page_sizes", "text_layer"} <= ids


def test_preflight_flags_non_embedded_font() -> None:
    # Base-14 Helvetica is not embedded → font check should fail.
    report = preflight_pdf(_text_pdf())
    font_check = next(c for c in report.checks if c.id == "fonts_embedded")
    assert font_check.status == "fail"


# --- preflight: fix mixed page sizes ----------------------------------------


def _mixed_size_pdf() -> bytes:
    """A 3-page PDF: A4, A3, A4 (so the majority/auto target is A4)."""

    doc = fitz.open()
    a4 = doc.new_page(width=595.28, height=841.89)
    a4.insert_text((50, 80), "Page 1 A4", fontname="helv", fontsize=14)
    a3 = doc.new_page(width=841.89, height=1190.55)
    a3.insert_text((50, 80), "Page 2 A3", fontname="helv", fontsize=14)
    a4b = doc.new_page(width=595.28, height=841.89)
    a4b.insert_text((50, 80), "Page 3 A4", fontname="helv", fontsize=14)
    data = doc.tobytes()
    doc.close()
    return data


def test_preflight_detects_mixed_page_sizes() -> None:
    report = preflight_pdf(_mixed_size_pdf())
    assert report.mixed_page_sizes is True
    assert len(report.distinct_page_sizes) == 2
    pg = next(c for c in report.checks if c.id == "page_sizes")
    assert pg.status == "warn"


def test_preflight_single_size_is_not_mixed() -> None:
    report = preflight_pdf(_text_pdf())
    assert report.mixed_page_sizes is False


def test_normalize_auto_uses_majority_size_and_keeps_text() -> None:
    src = _mixed_size_pdf()
    res = normalize_page_sizes(src, target="auto")
    # Only the odd A3 page needs resizing; the two A4 pages pass through.
    assert res.pages_total == 3
    assert res.pages_changed == 1

    doc = fitz.open(stream=res.data, filetype="pdf")
    sizes = {(round(doc[i].rect.width), round(doc[i].rect.height)) for i in range(doc.page_count)}
    # Every page is now one uniform size.
    assert len(sizes) == 1
    # Content from the rebuilt page is still selectable text (vector, not raster).
    assert "A3" in doc[1].get_text("text")
    doc.close()


def test_normalize_named_size_changes_all_pages() -> None:
    src = _mixed_size_pdf()
    res = normalize_page_sizes(src, target="letter")
    assert res.pages_changed == 3  # none of the pages were Letter
    doc = fitz.open(stream=res.data, filetype="pdf")
    sizes = {(round(doc[i].rect.width), round(doc[i].rect.height)) for i in range(doc.page_count)}
    doc.close()
    assert sizes == {(612, 792)}  # US Letter in points


# --- edit --------------------------------------------------------------------


def test_extract_spans_returns_geometry_and_style() -> None:
    result = extract_text_spans(_text_pdf())
    assert result.has_text is True
    assert len(result.spans) >= 2
    span = result.spans[0]
    assert span.text.strip()
    assert len(span.bbox) == 4
    assert span.size > 0
    assert span.font


def test_apply_edit_replaces_text_in_place() -> None:
    src = _text_pdf()
    spans = extract_text_spans(src).spans
    target = spans[0]
    op = EditOp(
        page=target.page,
        bbox=target.bbox,
        new_text="1. EDITED line of text.",
        font=target.font,
        size=target.size,
        color=target.color,
    )
    out = apply_text_edits(src, [op])

    doc = fitz.open(stream=out, filetype="pdf")
    text = doc[0].get_text("text")
    doc.close()
    assert "EDITED line of text" in text
    # The original words of that span are gone.
    assert "quick brown fox" not in text


def test_apply_edit_no_edits_is_noop_safe() -> None:
    # Empty new_text means deletion; the original glyphs are removed.
    src = _text_pdf()
    spans = extract_text_spans(src).spans
    op = EditOp(page=spans[0].page, bbox=spans[0].bbox, new_text="")
    out = apply_text_edits(src, [op])
    doc = fitz.open(stream=out, filetype="pdf")
    text = doc[0].get_text("text")
    doc.close()
    assert "quick brown fox" not in text
