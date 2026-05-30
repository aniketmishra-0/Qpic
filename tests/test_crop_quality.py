"""Regression tests for two reported issues:

1. A repeating page footer (e.g. PW branding "Android App | iOS App | PW
   Website") that sits a little above the bottom margin must not be stitched
   into the middle of a cross-page solution crop.
2. Crops rendered for download must be high-resolution (rendered straight from
   the PDF vector source) so they stay sharp when zoomed.
"""

from __future__ import annotations

import fitz
from PIL import Image

from app.config import Settings
from app.services.crop_service import (
    crop_and_stitch,
    crop_and_stitch_hires,
    save_question_image,
)
from app.services.detector.text_detector import TextDetector
from app.services.pdf_service import pdf_to_images

FOOTER = "Android App | iOS App | PW Website"


def _build_cross_page_solution_pdf() -> bytes:
    """Solution 5 starts low on page 1 (with a repeating footer ~88% down) and
    continues at the top of page 2."""

    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((72, 80), "Solutions", fontsize=14)
    p1.insert_text((72, 120), "4. Solution four. Ans: A", fontsize=11)
    p1.insert_text((72, 140), "Exp: explanation for four.", fontsize=11)
    p1.insert_text((72, 600), "5. Text Solution: Ans: B", fontsize=11)
    p1.insert_text((72, 620), "Statement 1 is correct: UPSC conducts exams.", fontsize=11)
    p1.insert_text((72, 640), "Statement 2 is incorrect: The Chairman of SPSC.", fontsize=11)
    p1.insert_text((72, 740), FOOTER, fontsize=10)  # ~88% of an 842pt page

    p2 = doc.new_page()
    p2.insert_text((72, 80), "Statement 3 is incorrect: expenses of SPSC.", fontsize=11)
    p2.insert_text((72, 100), "6. Solution six. Ans: C", fontsize=11)
    p2.insert_text((72, 740), FOOTER, fontsize=10)

    data = doc.tobytes()
    doc.close()
    return data


def _build_paginated_footer_pdf() -> bytes:
    """Four solution pages whose footer carries a per-page page number.

    The footer text differs on every page ("… | PW Website  1", "… 2", …) so an
    exact-text repeat check never recognises it. Each solution's content runs
    down to ~71% of the page; the footer sits at ~90%. The footer must be
    stripped so it isn't dragged into the crop.
    """

    doc = fitz.open()
    height = 842
    for pno in range(1, 5):
        page = doc.new_page(width=595, height=height)
        if pno == 1:
            page.insert_text((72, 80), "Hints & Solutions", fontsize=14)
        page.insert_text((72, 120), f"Q{pno} Text Solution: Ans: B", fontsize=11)
        page.insert_text((72, 580), "The social cost of carbon refers to the", fontsize=11)
        page.insert_text((72, 600), "monetary value of damage from emissions.", fontsize=11)
        # Footer with a trailing per-page page number -> unique text per page.
        page.insert_text((180, 760), f"{FOOTER}    {pno}", fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data


def test_section_local_footer_excluded_from_crops() -> None:
    """A footer that appears only in the solutions *section* of a long paper —
    far fewer than half the document's pages — must still be recognised as
    running furniture and kept out of the crops.

    It sits at ~88% (above the tight bottom margin), so only the position-stable
    + margin-confined rule catches it; a half-the-pages threshold would not.
    """

    doc = fitz.open()
    height = 842
    total_pages = 30
    footer_from = 25  # footer on only 6 of 30 pages
    for pno in range(1, total_pages + 1):
        page = doc.new_page(width=595, height=height)
        if pno < footer_from:
            page.insert_text((72, 100), f"Q{pno}. Question {pno}?", fontsize=12)
            page.insert_text((72, 130), "A) one B) two C) three D) four", fontsize=12)
        else:
            snum = pno - footer_from + 1
            if pno == footer_from:
                page.insert_text((72, 70), "Hints & Solutions", fontsize=14)
            page.insert_text((72, 100), f"Q{snum} Text Solution: Ans: B", fontsize=12)
            page.insert_text((72, 600), "Statement analysis line one.", fontsize=12)
            page.insert_text((72, 625), "Statement analysis line two.", fontsize=12)
            page.insert_text((150, 742), FOOTER, fontsize=11)  # ~88%
    pdf_bytes = doc.tobytes()
    doc.close()

    footer_pct = (742.0 / height) * 100.0
    questions = TextDetector().detect(pdf_bytes, padding_px=10)
    solutions = [q for q in questions if q.is_solution]
    assert solutions
    for q in solutions:
        for seg in q.segments:
            assert seg.y_end_pct < footer_pct - 1.0, (
                f"S{q.q_num} segment reaches {seg.y_end_pct:.1f}% "
                f"(footer at {footer_pct:.1f}%)"
            )


def test_paginated_footer_does_not_strip_real_body_content() -> None:
    """Body lines that merely share a digit-stripped key across pages
    ("Statement 1 …", "Statement 2 …") must be kept — only outer-margin
    furniture is removed."""

    doc = fitz.open()
    height = 842
    positions = [300, 320, 340, 360]
    for pno in range(1, 5):
        page = doc.new_page(width=595, height=height)
        if pno == 1:
            page.insert_text((72, 80), "Hints & Solutions", fontsize=14)
        page.insert_text((72, 120), f"Q{pno} Text Solution: Ans: B", fontsize=11)
        page.insert_text(
            (72, positions[pno - 1]), f"Statement {pno} is correct here.", fontsize=11
        )
        page.insert_text((180, 760), f"{FOOTER}    {pno}", fontsize=10)
    pdf_bytes = doc.tobytes()
    doc.close()

    questions = TextDetector().detect(pdf_bytes, padding_px=10)
    assert questions
    # Each solution's content must reach down to its mid-page "Statement" line
    # (~35%+), proving the body line was not mistaken for footer furniture.
    for q in questions:
        assert any(seg.y_end_pct >= 35.0 for seg in q.segments)


def test_repeating_footer_excluded_from_cross_page_solution() -> None:
    pdf_bytes = _build_cross_page_solution_pdf()
    questions = TextDetector().detect(pdf_bytes, padding_px=10)

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page_h = float(doc.load_page(0).rect.height)
    footer_pct = (740.0 / page_h) * 100.0

    s5 = next((q for q in questions if q.is_solution and q.q_num == "5"), None)
    assert s5 is not None
    # The first (page-1) segment must end above the footer band.
    seg1 = s5.segments[0]
    assert seg1.page == 1
    assert not (seg1.y_start_pct <= footer_pct <= seg1.y_end_pct)


def test_paginated_footer_excluded_from_crops() -> None:
    """A footer whose only per-page change is a page number must still be
    recognised as furniture and kept out of every crop."""

    pdf_bytes = _build_paginated_footer_pdf()
    questions = TextDetector().detect(pdf_bytes, padding_px=10)

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page_h = float(doc.load_page(0).rect.height)
    footer_pct = (760.0 / page_h) * 100.0  # ~90%

    assert questions
    for q in questions:
        for seg in q.segments:
            # No segment may extend into the footer band.
            assert seg.y_end_pct < footer_pct, (
                f"{'S' if q.is_solution else 'Q'}{q.q_num} segment "
                f"reaches {seg.y_end_pct:.1f}% (footer at {footer_pct:.1f}%)"
            )


def test_hires_crop_is_higher_resolution_than_detection_raster() -> None:
    pdf_bytes = _build_cross_page_solution_pdf()
    settings = Settings()
    questions = TextDetector().detect(pdf_bytes, padding_px=10)
    s5 = next(q for q in questions if q.is_solution and q.q_num == "5")

    detection_dpi = 200
    crop_dpi = max(detection_dpi, settings.CROP_RENDER_DPI)

    page_images = pdf_to_images(pdf_bytes, detection_dpi)
    raster = crop_and_stitch(page_images=page_images, question=s5, padding_px=10)

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        hires = crop_and_stitch_hires(
            doc, s5, padding_px=10, detection_dpi=detection_dpi, crop_dpi=crop_dpi
        )

    # Higher DPI render → more pixels along the width → sharper when zoomed.
    assert hires.size[0] > raster.size[0]
    assert hires.size[1] > raster.size[1]


def test_hires_crop_cross_page_has_stitched_height() -> None:
    """A two-segment (cross-page) solution stitches into a single tall image."""

    pdf_bytes = _build_cross_page_solution_pdf()
    settings = Settings()
    questions = TextDetector().detect(pdf_bytes, padding_px=10)
    s5 = next(q for q in questions if q.is_solution and q.q_num == "5")
    assert len(s5.segments) == 2

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        hires = crop_and_stitch_hires(
            doc, s5, padding_px=10, detection_dpi=200, crop_dpi=settings.CROP_RENDER_DPI
        )

    assert hires.size[0] > 1
    assert hires.size[1] > 1


def test_save_uses_default_prefixes(tmp_path) -> None:
    img = Image.new("RGB", (10, 10), (255, 255, 255))
    q_path = save_question_image(img, "3", tmp_path, is_solution=False)
    s_path = save_question_image(img, "3", tmp_path, is_solution=True)
    assert q_path.name == "Q003.png"
    assert s_path.name == "S003.png"


def test_save_custom_prefixes(tmp_path) -> None:
    img = Image.new("RGB", (10, 10), (255, 255, 255))
    path = save_question_image(
        img, "7", tmp_path, is_solution=False, question_prefix="Ques", solution_prefix="Sol"
    )
    assert path.name == "Ques007.png"
    sol = save_question_image(
        img, "7", tmp_path, is_solution=True, question_prefix="Ques", solution_prefix="Sol"
    )
    assert sol.name == "Sol007.png"


def test_save_start_number_offsets_numbering(tmp_path) -> None:
    img = Image.new("RGB", (10, 10), (255, 255, 255))
    # Detected number 1 with start_number=11 -> 11; detected 2 -> 12.
    first = save_question_image(img, "1", tmp_path, start_number=11)
    second = save_question_image(img, "2", tmp_path, start_number=11)
    assert first.name == "Q011.png"
    assert second.name == "Q012.png"


def test_top_of_page_question_keeps_its_options() -> None:
    """A question whose stem starts at the very top of the page must keep its
    options line.

    Regression: the options line ("A) … B) …") of a top-of-page question lands
    inside the top-margin band and repeats verbatim across pages, so the
    furniture heuristic deleted it as a phantom running *header* — leaving each
    crop with only the stem and cutting every answer. The header test must be
    marker-aware: a line below a question marker is body content, never a
    header.
    """

    options = "A) First option   B) Second option   C) Third option   D) Fourth"
    doc = fitz.open()
    height = 842
    for pno in range(1, 3):
        page = doc.new_page(width=595, height=height)
        qn = pno * 2 - 1
        # Stem at the very top (~3%), options just below (~5%) — inside the
        # top-margin band and identical on both pages.
        page.insert_text((30, 30), f"{qn}. This is question {qn} - find the answer.", fontsize=12)
        page.insert_text((30, 50), options, fontsize=12)
        page.insert_text((30, 76), f"{qn + 1}. This is question {qn + 1} - find the answer.", fontsize=12)
        page.insert_text((30, 96), options, fontsize=12)
    pdf_bytes = doc.tobytes()
    doc.close()

    questions = TextDetector().detect(pdf_bytes, padding_px=10)
    assert len(questions) == 4

    # The options line's baseline sits at y=50 (~5.9%). Each top-of-page
    # question's crop must extend at least to it, proving the options were not
    # stripped as a header (the buggy stem-only crop ended near ~4%).
    options_baseline_pct = (50.0 / height) * 100.0
    for q in questions:
        seg = q.segments[0]
        assert seg.y_end_pct >= options_baseline_pct, (
            f"Q{q.q_num} ends at {seg.y_end_pct:.1f}% — options line "
            f"(baseline {options_baseline_pct:.1f}%) was cut off"
        )
