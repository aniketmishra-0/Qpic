"""Tests for answer-key parsing and key-driven gap recovery."""

from __future__ import annotations

from app.services.detector.answer_key import (
    expected_question_numbers,
    extract_answer_key_from_text,
)
from app.services.detector.base import (
    ContentLine,
    QuestionStart,
    starts_to_questions,
)


def test_parses_dotted_paren_key() -> None:
    key = extract_answer_key_from_text(
        "1. (b)  2. (a)  3. (d)  4. (c)  5. (a)  6. (b)  7. (c)  8. (d)"
    )
    assert key == {1: "B", 2: "A", 3: "D", 4: "C", 5: "A", 6: "B", 7: "C", 8: "D"}


def test_parses_dash_key() -> None:
    key = extract_answer_key_from_text("1-B 2-A 3-D 4-C 5-A 6-B")
    assert key == {1: "B", 2: "A", 3: "D", 4: "C", 5: "A", 6: "B"}


def test_rejects_non_key_text() -> None:
    # A normal sentence with a couple of "number letter" coincidences is not a
    # key (too few pairs, not sequential).
    assert extract_answer_key_from_text(
        "In 1 a body of mass 2 b moves with velocity 3 c metres."
    ) == {}


def test_expected_numbers_fills_internal_hole() -> None:
    # Key itself missing 4 (its line mangled) — expected run still covers 1..6.
    key = extract_answer_key_from_text("1-B 2-A 3-D 5-A 6-B 7-C")
    assert expected_question_numbers(key) == {1, 2, 3, 4, 5, 6, 7}


def test_key_drives_recovery_at_sequence_end() -> None:
    """A question missing from the END of the run (no upper neighbour) is
    recovered because the answer key proves it exists."""

    H, W = 1000.0, 600.0

    def line(y, text):
        return ContentLine(page_num=1, y_top=y, y_bottom=y + 15, x_left=50, x_right=550, text=text)

    # Answer key present in the text (gives expected 1..7).
    lines = [
        line(60, "1. q one"),
        line(120, "2. q two"),
        line(180, "3. q three"),
        line(240, "4. q four"),
        line(300, "5. q five"),
        line(360, "6. q six"),
        line(420, "7. q seven misread marker"),  # "7" detected fine below
        line(700, "Answer Key"),
        line(720, "1-B 2-A 3-D 4-C 5-A 6-B 7-D"),
    ]
    # 7 was misread so it's absent from starts (1..6 detected).
    starts = [
        QuestionStart(page_num=1, y_top=60 + i * 60, q_num=str(i + 1), x_left=50, x_right=200, is_strong=False)
        for i in range(6)
    ]
    qs = starts_to_questions(
        starts=starts, page_heights={1: H}, total_pages=1,
        content_lines=lines, page_widths={1: W},
    )
    nums = [q.q_num for q in qs if not q.is_solution]
    assert "7" in nums, nums
