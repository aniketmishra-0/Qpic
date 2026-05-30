"""Regression test for the same-row marker/stem split that dropped a question.

A question number ("20.") is often a separate text block whose top sits a hair
below the first words of its own stem on the same row ("2.5 mL of 2 ..."). The
region builder orders content by y_top, so that stem fragment — being slightly
*above* the number — was filed under the PREVIOUS question, which then swallowed
the next question's opening line. The next question came out half (or was
dropped downstream), producing the "Q20 missing, numbering jumps 19 -> 21" bug.

Snapping each marker's boundary to its own row top keeps the current question's
first line with it and stops the previous question before it.
"""

from __future__ import annotations

import fitz

from app.services.detector.text_detector import TextDetector

W, H = 595, 842


def _build_same_row_marker_pdf() -> bytes:
    """Page 2's Q20 has its number block slightly BELOW its same-row stem."""

    doc = fitz.open()

    # Page 1: a tall Q19 that nearly fills the page.
    p1 = doc.new_page(width=W, height=H)
    p1.insert_text((40, 60), "19.  Which of the following is a correct statement?", fontsize=11)
    y = 90
    for t in [
        "(A) Brownian motion destabilizes sols.",
        "(B) Any amount of dispersed phase can be added.",
        "(C) Mixing two oppositely charged sols neutralizes.",
        "(D) Presence of equal and similar charges provides stability.",
    ]:
        p1.insert_text((70, y), t, fontsize=10)
        y += 18
    for i in range(28):
        p1.insert_text((70, y), f"explanation line {i} for question nineteen.", fontsize=9)
        y += 22
        if y > H - 40:
            break

    # Page 2: Q20 — stem first words a hair ABOVE the number block (same row).
    p2 = doc.new_page(width=W, height=H)
    p2.insert_text((60, 48), "2.5 mL of 2", fontsize=11)            # stem, y_top higher
    p2.insert_text((33, 50), "20.", fontsize=11)                     # number, y_top lower
    p2.insert_text((360, 50), "5  M weak monoacidic base. The", fontsize=11)
    p2.insert_text((60, 72), "concentration of H+ at equivalence point is", fontsize=11)
    p2.insert_text((60, 96), "(A) 3.7e-13 M", fontsize=10)
    p2.insert_text((315, 96), "(B) 3.2e-7 M", fontsize=10)
    p2.insert_text((60, 116), "(C) 3.2e-2 M", fontsize=10)
    p2.insert_text((315, 116), "(D) 2.7e-2 M", fontsize=10)
    p2.insert_text((40, 300), "21.  Which of the following is/are true?", fontsize=11)
    p2.insert_text((70, 322), "(A) option a", fontsize=10)
    p2.insert_text((315, 322), "(B) option b", fontsize=10)

    data = doc.tobytes()
    doc.close()
    return data


def test_q20_not_dropped_by_same_row_marker() -> None:
    pdf = _build_same_row_marker_pdf()
    questions = TextDetector().detect(pdf, padding_px=15)
    nums = [q.q_num for q in questions if not q.is_solution]
    assert "20" in nums, nums

    q20 = next(q for q in questions if q.q_num == "20")
    seg = next((s for s in q20.segments if s.page == 2), None)
    assert seg is not None
    # Q20 must include its options (down to ~14% of the page), not just the stem.
    assert seg.y_end_pct >= 13.0, seg.y_end_pct


def test_q19_does_not_steal_q20_first_line() -> None:
    pdf = _build_same_row_marker_pdf()
    questions = TextDetector().detect(pdf, padding_px=15)
    q19 = next(q for q in questions if q.q_num == "19")
    # Q19 must end on page 1 — it must NOT continue onto page 2 (where Q20 is).
    assert all(s.page == 1 for s in q19.segments), [s.page for s in q19.segments]
