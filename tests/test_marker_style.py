"""Tests for the forced question-numbering style (auto / q / numbered)."""

from __future__ import annotations

import fitz

from app.services.detector.base import match_question_start_ex
from app.services.detector.text_detector import TextDetector


def test_match_ex_style_q_rejects_bare_numbers() -> None:
    assert match_question_start_ex("1. Some statement", "q") is None
    assert match_question_start_ex("Q1. Real question", "q") == ("1", True)
    assert match_question_start_ex("Question 3 here", "q") == ("3", True)


def test_match_ex_style_numbered_rejects_q_prefix() -> None:
    assert match_question_start_ex("Q1. text", "numbered") is None
    assert match_question_start_ex("1. text", "numbered") == ("1", False)
    assert match_question_start_ex("2) text", "numbered") == ("2", False)


def test_match_ex_auto_accepts_both() -> None:
    assert match_question_start_ex("Q5. text", "auto") == ("5", True)
    assert match_question_start_ex("5. text", "auto") == ("5", False)


def _make_pdf(lines: list[tuple[float, str]]) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    for y, text in lines:
        page.insert_text((72, y), text, fontsize=12)
    data = doc.tobytes()
    doc.close()
    return data


def test_text_detector_style_q_ignores_substatements() -> None:
    # A Q-numbered paper whose stem lists bare-numbered sub-statements. With
    # style "q" only the two real Q-markers become questions.
    pdf = _make_pdf(
        [
            (72, "Q1. Consider the following statements:"),
            (96, "1. First statement here"),
            (120, "2. Second statement here"),
            (160, "Q2. Another real question"),
            (184, "1. sub point"),
        ]
    )
    questions = TextDetector().detect(pdf, padding_px=0, marker_style="q")
    assert sorted(q.q_num for q in questions) == ["1", "2"]


def test_text_detector_style_numbered_only_bare_numbers() -> None:
    pdf = _make_pdf(
        [
            (72, "1. First question " + ("x" * 40)),
            (130, "2. Second question " + ("y" * 40)),
            (190, "3. Third question " + ("z" * 40)),
        ]
    )
    questions = TextDetector().detect(pdf, padding_px=0, marker_style="numbered")
    assert sorted(q.q_num for q in questions) == ["1", "2", "3"]
