"""Regression tests for bordered content note-boxes.

Exam papers frame asides such as "Additional Information" / "PW ONLYIAS SUPER
HINT" in a thin rectangle that *encloses real body text*. Three defects used to
damage them:

  1. each thin border stroke was classed as a decorative rule and painted white,
     erasing the box frame;
  2. the crop bottom was driven only by text extents, so the box's bottom border
     (just below the last line) fell outside the crop — slicing the box and
     cutting its final sentence; and
  3. ``trim_edge_rules`` ate the box's vertical side border, mistaking it for a
     lone decorative accent bar.

These tests pin the fixes: the frame is detected, kept out of furniture, folded
into the owning question's crop, and its side border survives trimming.
"""

from __future__ import annotations

import fitz
import numpy as np
from PIL import Image, ImageDraw

from app.services.detector.figure_detector import extract_figures_for_page
from app.services.detector.furniture import (
    collect_page_furniture,
    detect_content_boxes,
)
from app.services.detector.text_detector import TextDetector
from app.utils.image_utils import trim_edge_rules

W, H = 595, 842


def _build_box_pdf() -> bytes:
    """A solution whose last paragraph sits inside an 'Additional Information'
    box; the box's bottom border is just below the final text line."""

    doc = fitz.open()
    p = doc.new_page(width=W, height=H)
    p.insert_text((60, 90), "Q3 Text Solution:", fontsize=12)
    p.insert_text((60, 115), "Ans: (d)", fontsize=11)
    p.insert_text((60, 150), "The correct option follows from the article cited.", fontsize=11)

    # Bordered note-box drawn as four thin strokes (a real frame).
    bx0, by0, bx1, by1 = 55, 200, 540, 320
    p.draw_line((bx0, by0), (bx1, by0), width=0.8)  # top
    p.draw_line((bx0, by1), (bx1, by1), width=0.8)  # bottom
    p.draw_line((bx0, by0), (bx0, by1), width=0.8)  # left
    p.draw_line((bx1, by0), (bx1, by1), width=0.8)  # right
    p.insert_text((70, 225), "Additional Information:", fontsize=11)
    p.insert_text((70, 250), "Six Fundamental Rights are listed in Part III, and", fontsize=10)
    p.insert_text((70, 270), "the final one — Right to Constitutional Remedies", fontsize=10)
    p.insert_text((70, 300), "(Article 32) — sits on the last line of this box.", fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data


def test_detect_content_boxes_finds_the_frame() -> None:
    pdf = _build_box_pdf()
    with fitz.open(stream=pdf, filetype="pdf") as doc:
        boxes = detect_content_boxes(doc.load_page(0))
    assert len(boxes) == 1
    box = boxes[0]
    assert box.y0 < 210 and box.y1 > 315
    assert box.x0 < 60 and box.x1 > 535


def _build_single_rect_box_pdf() -> bytes:
    """Same as ``_build_box_pdf`` but the frame is drawn as ONE stroked
    rectangle path (``draw_rect``) instead of four separate line strokes.

    This is how PW "ONLYIAS SUPER HINT" boxes are actually drawn, and it used to
    slip past detection — PyMuPDF reports it as a single ``"re"`` drawing, so the
    two-vertical-strokes heuristic never matched and the crop sliced the box.
    """

    doc = fitz.open()
    p = doc.new_page(width=W, height=H)
    p.insert_text((60, 90), "Q4 Text Solution:", fontsize=12)
    p.insert_text((60, 115), "Ans: (b)", fontsize=11)
    p.insert_text((60, 150), "The correct option follows from the article cited.", fontsize=11)

    bx0, by0, bx1, by1 = 55, 200, 540, 320
    p.draw_rect(fitz.Rect(bx0, by0, bx1, by1), width=0.8)  # single rectangle frame
    p.insert_text((70, 225), "PW ONLYIAS SUPER HINT", fontsize=11)
    p.insert_text((70, 250), "The phrase 'explicitly mentioned' is a trap; it is", fontsize=10)
    p.insert_text((70, 270), "not written verbatim in Part III, so the statement", fontsize=10)
    p.insert_text((70, 300), "sitting on the last line of this box is likely false.", fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data


def test_detect_content_boxes_finds_single_rect_frame() -> None:
    """A frame drawn as one rectangle path is detected, just like four strokes."""

    pdf = _build_single_rect_box_pdf()
    with fitz.open(stream=pdf, filetype="pdf") as doc:
        boxes = detect_content_boxes(doc.load_page(0))
    assert len(boxes) == 1, boxes
    box = boxes[0]
    assert box.y0 < 210 and box.y1 > 315
    assert box.x0 < 60 and box.x1 > 535


def test_single_rect_box_crop_grows_to_the_box_bottom() -> None:
    """The single-rectangle SUPER HINT box must grow the crop to its bottom
    border, not stop at the last text line (the reported cut-off bug)."""

    pdf = _build_single_rect_box_pdf()
    questions = TextDetector().detect(pdf, padding_px=10)
    s4 = next((q for q in questions if q.q_num == "4"), None)
    assert s4 is not None
    seg = next((s for s in s4.segments if s.page == 1), None)
    assert seg is not None
    assert seg.y_end_pct >= (320.0 / H) * 100.0 - 0.5, seg.y_end_pct


def test_box_borders_are_not_collected_as_furniture() -> None:
    """The frame strokes must not be painted out (they are kept content)."""

    pdf = _build_box_pdf()
    with fitz.open(stream=pdf, filetype="pdf") as doc:
        page = doc.load_page(0)
        boxes = detect_content_boxes(page)
        furn = collect_page_furniture(page)
    box = boxes[0]
    tol = 4.0
    for fr in furn:
        on_perimeter = (
            abs(fr.x0 - box.x0) <= tol or abs(fr.x1 - box.x1) <= tol
            or abs(fr.y0 - box.y0) <= tol or abs(fr.y1 - box.y1) <= tol
        )
        inside = fr.x0 >= box.x0 - tol and fr.x1 <= box.x1 + tol and fr.y0 >= box.y0 - tol and fr.y1 <= box.y1 + tol
        assert not (on_perimeter and inside), f"box border wrongly furniture: {fr}"


def test_box_is_emitted_as_a_figure_region() -> None:
    """The frame is returned as a figure so the crop grows to contain it."""

    pdf = _build_box_pdf()
    with fitz.open(stream=pdf, filetype="pdf") as doc:
        figs = extract_figures_for_page(doc.load_page(0), 1)
    # A region covering the full box must be present.
    assert any(f.y_top < 210 and f.y_bottom > 315 and f.x_left < 60 and f.x_right > 535 for f in figs)


def test_solution_crop_grows_to_the_box_bottom() -> None:
    """The detected solution segment must reach the box's bottom border, not
    stop at the last text line above it."""

    pdf = _build_box_pdf()
    questions = TextDetector().detect(pdf, padding_px=10)
    s3 = next((q for q in questions if q.q_num == "3"), None)
    assert s3 is not None
    seg = next((s for s in s3.segments if s.page == 1), None)
    assert seg is not None
    # Box bottom border at y=320 => ~38% of page height. The crop must reach it.
    assert seg.y_end_pct >= (320.0 / H) * 100.0 - 0.5, seg.y_end_pct


def test_trim_edge_rules_keeps_a_box_side_border() -> None:
    """A vertical strip joined by horizontal corners (a box border) is kept,
    while a lone accent bar is still trimmed."""

    # A framed box near the left edge with text inside.
    img = Image.new("RGB", (1200, 1600), (255, 255, 255))
    d = ImageDraw.Draw(img)
    bx0, by0, bx1, by1 = 40, 100, 1150, 1500
    d.rectangle([bx0, by0, bx1, by1], outline=(10, 10, 10), width=4)
    for y in range(160, 1450, 60):
        d.rectangle([120, y, 120 + 900, y + 26], fill=(15, 15, 15))
    out = trim_edge_rules(img)
    # The left border column (~40) must survive: width is essentially unchanged.
    assert out.width >= img.width - 6, (img.width, out.width)


def test_trim_edge_rules_still_trims_lone_accent_bar() -> None:
    """A lone vertical accent bar (no horizontal corners) is still trimmed."""

    img = Image.new("RGB", (1591, 2229), (255, 255, 255))
    d = ImageDraw.Draw(img)
    top, bot = int(2229 * 0.05), int(2229 * 0.94)
    d.rectangle([45, top, 56, bot], fill=(194, 179, 180))  # accent bar, no corners
    for y in range(top, bot, 60):
        d.rectangle([120, y, 120 + 1200, y + 26], fill=(15, 15, 15))
    out = trim_edge_rules(img)
    assert out.width < img.width
