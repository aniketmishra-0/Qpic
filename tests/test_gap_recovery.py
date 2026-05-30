"""Regression tests for OCR gap-recovery of misread question numbers.

A question number that OCR mangles ("20." -> "2O.", "Z0.") never matches a
marker pattern, so the question silently vanishes and the numbering jumps
(19 -> 21). The gap-recovery pass notices the missing number in an otherwise
sequential run and re-reads the candidate line's leading token with digit
fixups, re-inserting the marker so the question is cropped instead of dropped.
"""

from __future__ import annotations

from app.services.detector.base import (
    ContentLine,
    QuestionStart,
    _ocr_token_to_int,
    line_matches_expected_number,
    starts_to_questions,
)

H, W = 1000.0, 600.0


def _line(y: float, text: str, x0: float = 50.0, x1: float = 550.0) -> ContentLine:
    return ContentLine(page_num=1, y_top=y, y_bottom=y + 15, x_left=x0, x_right=x1, text=text)


def test_ocr_token_to_int_fixups() -> None:
    assert _ocr_token_to_int("2O") == 20
    assert _ocr_token_to_int("Z0") == 20
    assert _ocr_token_to_int("l9") == 19
    assert _ocr_token_to_int("20") == 20
    # Non-numeric tokens are rejected (no spurious marker).
    assert _ocr_token_to_int("This") is None
    assert _ocr_token_to_int("") is None
    assert _ocr_token_to_int("1234") is None  # too long for a question number


def test_line_matches_expected_number() -> None:
    assert line_matches_expected_number("2O. some stem text", 20) is True
    assert line_matches_expected_number("Z0) another stem", 20) is True
    assert line_matches_expected_number("This is question text", 20) is False
    assert line_matches_expected_number("21. different number", 20) is False


def test_missing_number_recovered_from_misread_marker() -> None:
    """A run 18,19,(20),21,22 where 20's marker was misread as '2O.' recovers."""

    lines = [
        _line(100, "18. Question eighteen stem."),
        _line(120, "(A) a  (B) b  (C) c  (D) d"),
        _line(200, "19. Question nineteen stem."),
        _line(220, "(A) a  (B) b  (C) c  (D) d"),
        _line(300, "2O. Question twenty stem (misread marker)."),
        _line(320, "(A) a  (B) b  (C) c  (D) d"),
        _line(400, "21. Question twenty-one stem."),
        _line(420, "(A) a  (B) b  (C) c  (D) d"),
        _line(500, "22. Question twenty-two stem."),
        _line(520, "(A) a  (B) b  (C) c  (D) d"),
    ]
    # 20 is absent because "2O." didn't match a marker pattern.
    starts = [
        QuestionStart(page_num=1, y_top=100, q_num="18", x_left=50, x_right=200, is_strong=False),
        QuestionStart(page_num=1, y_top=200, q_num="19", x_left=50, x_right=200, is_strong=False),
        QuestionStart(page_num=1, y_top=400, q_num="21", x_left=50, x_right=200, is_strong=False),
        QuestionStart(page_num=1, y_top=500, q_num="22", x_left=50, x_right=200, is_strong=False),
    ]

    qs = starts_to_questions(
        starts=starts, page_heights={1: H}, total_pages=1,
        content_lines=lines, page_widths={1: W},
    )
    nums = [q.q_num for q in qs if not q.is_solution]
    assert "20" in nums, nums
    # Recovered Q20 starts at its own line (~30%), not merged into a neighbour.
    q20 = next(q for q in qs if q.q_num == "20")
    assert q20.segments[0].y_start_pct < 32.0


def test_no_recovery_when_sequence_too_sparse() -> None:
    """A scattered, non-sequential set of numbers must not trigger recovery."""

    lines = [
        _line(100, "3. stem"),
        _line(300, "40. stem"),
        _line(500, "77. stem"),
    ]
    starts = [
        QuestionStart(page_num=1, y_top=100, q_num="3", x_left=50, x_right=200, is_strong=False),
        QuestionStart(page_num=1, y_top=300, q_num="40", x_left=50, x_right=200, is_strong=False),
        QuestionStart(page_num=1, y_top=500, q_num="77", x_left=50, x_right=200, is_strong=False),
    ]
    qs = starts_to_questions(
        starts=starts, page_heights={1: H}, total_pages=1,
        content_lines=lines, page_widths={1: W},
    )
    nums = sorted(int(q.q_num) for q in qs if not q.is_solution)
    # No spurious intermediate numbers invented.
    assert nums == [3, 40, 77], nums
