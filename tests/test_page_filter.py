import pytest

from app.models.schemas import DetectedQuestion, QuestionSegment
from app.services.page_filter import (
    PageRangeError,
    apply_page_ranges,
    parse_page_ranges,
)


def _q(q_num: str, page: int, is_solution: bool = False) -> DetectedQuestion:
    return DetectedQuestion(
        q_num=q_num,
        is_solution=is_solution,
        segments=[QuestionSegment(page=page, y_start_pct=10.0, y_end_pct=20.0)],
    )


def test_parse_empty_returns_empty_set():
    assert parse_page_ranges(None) == set()
    assert parse_page_ranges("") == set()
    assert parse_page_ranges("   ") == set()


def test_parse_simple_range():
    assert parse_page_ranges("1-5") == {1, 2, 3, 4, 5}


def test_parse_to_keyword():
    assert parse_page_ranges("1 to 5") == {1, 2, 3, 4, 5}
    assert parse_page_ranges("1 TO 3") == {1, 2, 3}


def test_parse_single_page():
    assert parse_page_ranges("8") == {8}


def test_parse_mixed_list():
    assert parse_page_ranges("1-5, 8, 10-12") == {1, 2, 3, 4, 5, 8, 10, 11, 12}


def test_parse_reversed_range_is_normalized():
    assert parse_page_ranges("5-1") == {1, 2, 3, 4, 5}


def test_parse_respects_max_page():
    assert parse_page_ranges("1-100", max_page=3) == {1, 2, 3}


def test_parse_invalid_raises():
    with pytest.raises(PageRangeError):
        parse_page_ranges("abc")
    with pytest.raises(PageRangeError):
        parse_page_ranges("1-")


def test_apply_no_ranges_passthrough():
    questions = [_q("1", 1), _q("2", 2)]
    out = apply_page_ranges(questions, set(), set())
    assert out == questions


def test_apply_relabels_answers_and_questions():
    detected = [_q("1", 1), _q("2", 2), _q("1", 7), _q("2", 8)]
    out = apply_page_ranges(detected, question_pages={1, 2}, answer_pages={7, 8})

    assert len(out) == 4
    by_key = {(q.q_num, q.is_solution) for q in out}
    assert ("1", False) in by_key
    assert ("2", False) in by_key
    assert ("1", True) in by_key
    assert ("2", True) in by_key


def test_apply_drops_pages_outside_ranges():
    detected = [_q("1", 1), _q("2", 6), _q("3", 9)]
    out = apply_page_ranges(detected, question_pages={1, 2}, answer_pages={9})

    pages_kept = sorted(min(s.page for s in q.segments) for q in out)
    assert pages_kept == [1, 9]


def test_apply_question_only_range_keeps_rest_auto_detected():
    # Only the question range is given. Pages in it are forced to questions;
    # everything else falls back to its auto-detected label rather than being
    # dropped (filling one field must not discard the other category).
    detected = [_q("1", 1), _q("2", 5, is_solution=True), _q("3", 9, is_solution=True)]
    out = apply_page_ranges(detected, question_pages={1, 2, 3, 4, 5}, answer_pages=set())

    assert len(out) == 3
    by_page = {min(s.page for s in q.segments): q.is_solution for q in out}
    # Page 5 is inside the question range -> relabeled to a question.
    assert by_page[1] is False
    assert by_page[5] is False
    # Page 9 is outside the only given range -> kept with its detected label.
    assert by_page[9] is True


def test_apply_answer_only_range_keeps_questions():
    # Only the answer range is given (the reported bug). Questions on other
    # pages must survive instead of being thrown away.
    detected = [_q("1", 1), _q("2", 1), _q("1", 5), _q("2", 6)]
    out = apply_page_ranges(detected, question_pages=set(), answer_pages={5, 6})

    assert len(out) == 4
    solutions = sorted(min(s.page for s in q.segments) for q in out if q.is_solution)
    questions = sorted(min(s.page for s in q.segments) for q in out if not q.is_solution)
    assert solutions == [5, 6]
    assert questions == [1, 1]


def test_apply_both_ranges_drops_outside():
    # When BOTH ranges are supplied the document is fully partitioned, so a page
    # in neither range is dropped.
    detected = [_q("1", 1), _q("2", 6), _q("3", 9)]
    out = apply_page_ranges(detected, question_pages={1, 2}, answer_pages={9})

    pages_kept = sorted(min(s.page for s in q.segments) for q in out)
    assert pages_kept == [1, 9]


def test_strict_drops_pages_outside_question_range_only():
    # Strict mode with only a question range: items on any other page are
    # dropped (crop exactly the listed pages, no auto-detected extras).
    detected = [_q("1", 1), _q("2", 2), _q("3", 5, is_solution=True)]
    out = apply_page_ranges(
        detected, question_pages={1, 2}, answer_pages=set(), strict=True
    )

    pages_kept = sorted(min(s.page for s in q.segments) for q in out)
    assert pages_kept == [1, 2]
    assert all(q.is_solution is False for q in out)


def test_strict_keeps_only_listed_question_and_answer_pages():
    detected = [_q("1", 1), _q("2", 3), _q("3", 7), _q("4", 9)]
    out = apply_page_ranges(
        detected, question_pages={1, 2}, answer_pages={7, 8}, strict=True
    )

    by_page = {min(s.page for s in q.segments): q.is_solution for q in out}
    assert set(by_page) == {1, 7}
    assert by_page[1] is False
    assert by_page[7] is True
