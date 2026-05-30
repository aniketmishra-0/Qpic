"""Regression test: a section/part divider between two questions must not be
swallowed by the preceding question's crop.

Real papers separate question groups with a "PART-II (CHEMISTRY)" / "SECTION-1"
title followed by a marking-scheme rubric. These lines sit between the last
option of one question and the marker of the next, so the boundary builder used
to fold the whole divider block (and often the next question's first line) into
the earlier question. The crop should instead end at that question's own last
option.
"""

from __future__ import annotations

import fitz

from app.services.detector.base import match_section_header
from app.services.detector.text_detector import TextDetector

W, H = 595, 842


def _build_section_pdf() -> bytes:
    doc = fitz.open()
    p = doc.new_page(width=W, height=H)
    y = 60
    lh = 15

    def line(text: str, x: float = 50, size: float = 11) -> None:
        nonlocal y
        p.insert_text((x, y), text, fontsize=size, fontname="helv")
        y += lh

    line("16. Let XL, XC be the inductive reactance and capacitive reactance and R")
    line("     be the resistance in each of the circuits given in List-I.")
    line("     (A) I-P,R,S,T; II-Q,R,S; III-P,R,S,T; IV-P,R,S,T")
    line("     (B) I-Q,P,R,S; II-S,Q,R; III-R,S,Q,T; IV-Q,P,R,S")
    line("     (C) I-P,Q,R,S; II-S,R,Q; III-P,S,Q,T; IV-R,S,P,Q")
    line("     (D) I-P,S,R,Q; II-R,S,Q; III-P,Q,R,S; IV-Q,P,R,S")
    q16_options_bottom = y
    y += 20

    line("PART-II (CHEMISTRY)", x=240, size=13)
    y += 6
    line("SECTION-1 (Maximum Marks: 12)", x=200)
    line("- This section contains FOUR (04) questions.")
    line("- Each question has FOUR options (A), (B), (C) and (D). ONLY ONE is correct.")
    line("     Full Marks   : +3 If ONLY the correct option is chosen;")
    line("     Negative Marks : -1 In all other cases.")
    y += 10

    line("17. In the scheme given below, X and Y, respectively, are")
    line("     (A) option one")
    line("     (B) option two")
    line("     (C) option three")
    line("     (D) option four")

    data = doc.tobytes()
    doc.close()
    return data, (q16_options_bottom / H) * 100.0


def test_match_section_header_recognizes_dividers() -> None:
    assert match_section_header("PART-II (CHEMISTRY)")
    assert match_section_header("PART - A")
    assert match_section_header("SECTION-1 (Maximum Marks: 12)")
    assert match_section_header("SECTION 2")
    assert match_section_header("This section contains FOUR (04) questions.")
    # Not dividers: ordinary question content mentioning the words.
    assert not match_section_header(
        "16. In the part of the circuit shown the section modulus is constant"
    )
    assert not match_section_header("(A) the section of the beam is rectangular")


def test_question_crop_stops_before_section_divider() -> None:
    pdf, q16_options_bottom_pct = _build_section_pdf()
    questions = TextDetector().detect(pdf, padding_px=0)

    q16 = next((q for q in questions if q.q_num == "16"), None)
    q17 = next((q for q in questions if q.q_num == "17"), None)
    assert q16 is not None and q17 is not None

    seg = q16.segments[0]
    # Q16 must end at its own last option, not down at the divider/instructions.
    # Allow a small margin past the option baseline for glyph descenders.
    assert seg.y_end_pct <= q16_options_bottom_pct + 2.0, seg.y_end_pct
    # And well above where Q17 begins.
    assert seg.y_end_pct < q17.segments[0].y_start_pct
