"""Regression test: text underlines must not be painted out as furniture.

PW solution papers underline bold labels ("Statement 1 is correct:") with a
thin horizontal stroke that rides directly on the text line. That stroke was
classed as a decorative horizontal rule and painted white (plus a 2pt pad) by
the hi-res renderer, slicing the bottom of the glyphs above it — the reported
"words getting cut" defect.

The fix keeps any thin horizontal stroke that overlaps a real content line (an
underline/strikethrough) out of the furniture set, while a genuine standalone
divider rule sitting in whitespace is still removed.
"""

from __future__ import annotations

import fitz
import numpy as np

from app.services.crop_service import crop_and_stitch_hires
from app.services.detector.furniture import (
    collect_document_furniture,
    collect_page_furniture,
)
from app.services.detector.text_detector import TextDetector
from app.services.pdf_service import render_page_region

W, H = 595, 842


def _build_underlined_pdf() -> bytes:
    """A solution whose bold label is underlined by a thin stroke on the text
    line, plus a standalone divider rule in whitespace further down.

    The strokes are thin *filled* rectangles (height ~0.7pt), matching how PW
    papers render underlines/dividers — a zero-height ``draw_line`` hairline is
    skipped before rule classification and would not exercise the fix.
    """

    doc = fitz.open()
    p = doc.new_page(width=W, height=H)
    p.insert_text((60, 90), "Q4 Text Solution:", fontsize=12)
    p.insert_text((60, 115), "Ans: (a)", fontsize=11)

    # Bold label with an underline stroke riding on the baseline (~y=152).
    label = "Statement 1 is correct:"
    p.insert_text((60, 150), label, fontsize=11, fontname="helvetica-bold")
    label_w = fitz.get_text_length(label, fontname="helvetica-bold", fontsize=11)
    p.draw_rect(fitz.Rect(60, 151.6, 60 + label_w, 152.3), color=None, fill=(0, 0, 0))

    p.insert_text((60, 175), "The article cited establishes this right clearly.", fontsize=11)

    # A standalone decorative divider rule in whitespace (should be removed).
    p.draw_rect(fitz.Rect(60, 229.6, 535, 230.3), color=None, fill=(0, 0, 0))

    p.insert_text((60, 255), "Further explanation continues after the divider.", fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def _underline_y() -> float:
    return 152.0


def test_underline_stroke_is_not_furniture() -> None:
    pdf = _build_underlined_pdf()
    with fitz.open(stream=pdf, filetype="pdf") as doc:
        furn = collect_page_furniture(doc.load_page(0))

    uy = _underline_y()
    # No furniture rect should sit on the underline stroke's row.
    on_underline = [f for f in furn if abs((f.y0 + f.y1) / 2.0 - uy) <= 2.0]
    assert on_underline == [], on_underline


def test_standalone_divider_is_still_furniture() -> None:
    pdf = _build_underlined_pdf()
    with fitz.open(stream=pdf, filetype="pdf") as doc:
        furn = collect_page_furniture(doc.load_page(0))

    # The divider rule at y=230 sits in whitespace -> still removed.
    on_divider = [f for f in furn if abs((f.y0 + f.y1) / 2.0 - 230.0) <= 2.0]
    assert on_divider, "standalone divider rule should be collected as furniture"


def test_rendered_label_glyphs_survive_furniture_paint() -> None:
    """Painting furniture must not erase the underlined label's glyph ink."""

    pdf = _build_underlined_pdf()
    with fitz.open(stream=pdf, filetype="pdf") as doc:
        page = doc.load_page(0)
        furn = [(f.x0, f.y0, f.x1, f.y1) for f in collect_page_furniture(page)]
        # Tight region around the underlined label line.
        ys = (140.0 / H) * 100.0
        ye = (156.0 / H) * 100.0
        xs = (55.0 / W) * 100.0
        xe = (300.0 / W) * 100.0
        plain = render_page_region(doc, 0, x_start_pct=xs, x_end_pct=xe,
                                   y_start_pct=ys, y_end_pct=ye, dpi=300)
        painted = render_page_region(doc, 0, x_start_pct=xs, x_end_pct=xe,
                                     y_start_pct=ys, y_end_pct=ye, dpi=300,
                                     furniture_rects=furn)

    a = np.asarray(plain.convert("L")).astype(int)
    b = np.asarray(painted.convert("L")).astype(int)
    assert a.shape == b.shape
    # Pixels inked in the plain render that became white after painting = sliced.
    lost = int(((a < 128) & (b >= 200)).sum())
    assert lost <= 5, f"furniture paint erased {lost} label/underline pixels"
