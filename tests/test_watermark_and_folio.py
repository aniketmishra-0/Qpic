"""Regression tests for two cross-page-crop defects on PW solution papers:

1. A background **watermark** (a faint brand logo printed at the same spot on
   every page) was extracted as a question *figure* and folded into whichever
   solution surrounded it — ballooning that crop down to the watermark's bottom
   (~64% of the page) and pulling the footer band into the crop.

2. A page-number **folio** ("5/5") in the bottom margin was not recognised as
   running furniture (stripping its per-page digit leaves an empty repeat key),
   so it dragged a column's crop bound to the page bottom.

Both showed up together on a two-column "Hints & Solutions" paper where the last
question's solution flows across a page break: the stitched crop contained the
"Android App | iOS App | PW Website" footer and a tall blank gap.
"""

from __future__ import annotations

import io

import fitz
import numpy as np
from PIL import Image

from app.services.crop_service import crop_and_stitch_hires
from app.services.detector.figure_detector import (
    extract_figures,
    extract_figures_for_page,
    filter_watermark_figures,
)
from app.services.detector.furniture import collect_document_furniture
from app.services.detector.text_detector import TextDetector

W, H = 595, 842
N_PAGES = 5


def _watermark_png() -> bytes:
    b = io.BytesIO()
    # A large, light-grey logo-like block (a stand-in for the faint PW logo).
    Image.new("RGB", (320, 320), (235, 235, 235)).save(b, "PNG")
    return b.getvalue()


def _build_watermarked_pdf() -> bytes:
    """A 5-page paper with a watermark at a fixed mid-page position on every
    page, a per-page folio ("n/5") in the bottom margin, and a solution (Q5)
    that finishes near the top of the last page."""

    doc = fitz.open()
    wm = _watermark_png()
    for pno in range(N_PAGES):
        p = doc.new_page(width=W, height=H)
        # Background watermark: identical box on EVERY page (x 24-71%, y 31-64%).
        p.insert_image(fitz.Rect(146, 264, 423, 541), stream=wm, overlay=False)
        # Per-page folio in the bottom margin.
        p.insert_text((545, 820), f"{pno + 1}/5", fontsize=8)

    # Page 5 (index 4): Q5 solution, finishing well above the watermark.
    p5 = doc.load_page(4)
    p5.insert_text((60, 55), "Q5 Text Solution:", fontsize=11)
    p5.insert_text((60, 80), "Ans: (b)", fontsize=10)
    for i, t in enumerate([
        "The social cost of carbon (SCC) refers to the",
        "monetary value associated with the damage",
        "caused by each additional ton of carbon dioxide.",
    ]):
        p5.insert_text((60, 105 + i * 18), t, fontsize=10)

    data = doc.tobytes()
    doc.close()
    return data


def test_watermark_is_filtered_from_figures() -> None:
    pdf = _build_watermarked_pdf()
    with fitz.open(stream=pdf, filetype="pdf") as doc:
        # The watermark IS extracted as a (large) figure on each page.
        raw = extract_figures_for_page(doc.load_page(4), 5)
        assert any(f.y_bottom / H > 0.6 for f in raw)
        # But the document-level filter removes it (repeats on all 5 pages).
        kept = extract_figures(doc)
    assert kept == [] or all(f.y_bottom / H <= 0.6 for f in kept)


def test_filter_keeps_a_non_repeating_figure() -> None:
    """A genuine diagram (appears on one page only) must be kept."""

    from app.services.detector.base import FigureRegion

    page_dims = {p: (W, H) for p in range(1, 6)}
    watermark = [
        FigureRegion(page_num=p, y_top=264, y_bottom=541, x_left=146, x_right=423)
        for p in range(1, 6)
    ]
    diagram = FigureRegion(page_num=2, y_top=120, y_bottom=260, x_left=40, x_right=300)
    kept = filter_watermark_figures(watermark + [diagram], page_dims)
    assert diagram in kept
    assert all(f not in kept for f in watermark)


def test_solution_crop_excludes_watermark_and_folio() -> None:
    pdf = _build_watermarked_pdf()
    questions = TextDetector().detect(pdf, padding_px=10)
    q5 = next((q for q in questions if q.q_num == "5"), None)
    assert q5 is not None

    # The page-5 segment must end at the real text bottom (~20%), NOT be dragged
    # down to the watermark (~64%) or the folio (~97%).
    seg = next((s for s in q5.segments if s.page == 5), None)
    assert seg is not None
    assert seg.y_end_pct < 40.0, seg.y_end_pct


def test_rendered_solution_has_no_large_blank_gap() -> None:
    pdf = _build_watermarked_pdf()
    questions = TextDetector().detect(pdf, padding_px=10)
    q5 = next((q for q in questions if q.q_num == "5"), None)
    assert q5 is not None

    with fitz.open(stream=pdf, filetype="pdf") as doc:
        furniture = collect_document_furniture(doc)
        img = crop_and_stitch_hires(
            doc, q5, padding_px=10, detection_dpi=200, crop_dpi=150,
            furniture_by_page=furniture,
        )

    arr = np.asarray(img.convert("L"))
    h = arr.shape[0]
    dark_per_row = (arr < 128).sum(axis=1)
    # Longest run of fully-blank rows must be a small fraction of the crop.
    longest = run = 0
    for d in dark_per_row:
        run = run + 1 if d == 0 else 0
        longest = max(longest, run)
    assert longest / h < 0.25, longest / h
