"""Regression tests for MCQ-aware option validation.

Standard MCQs have four options (A)-(D). When detection captures only some of
them — typically the left column "(A)/(C)" of a 2-up option grid whose right
half got clipped — the review step flags the item so the user can re-select the
full question. Complete (ABCD) and non-option bodies are never flagged.
"""

from __future__ import annotations

from app.models.schemas import DetectedQuestion, QuestionSegment
from app.services.detector.base import _option_letters_in
from app.services.review_service import (
    _missing_options_reason,
    build_analyzed_items,
    build_review_notes,
)


def _q(q_num: str, labels: str, y0: float = 5.0, y1: float = 20.0) -> DetectedQuestion:
    return DetectedQuestion(
        q_num=q_num,
        segments=[QuestionSegment(page=1, y_start_pct=y0, y_end_pct=y1)],
        option_labels=labels,
    )


def test_option_letters_in_requires_label_punctuation() -> None:
    assert _option_letters_in("(A) one  (B) two") == {"A", "B"}
    assert _option_letters_in("C) three  D. four") == {"C", "D"}
    # A capital inside a word is not an option label.
    assert _option_letters_in("Acceleration Brings Change Daily") == set()
    assert _option_letters_in("") == set()


def test_partial_options_flagged() -> None:
    reason = _missing_options_reason(_q("2", "AC"))
    assert reason is not None
    assert "(B)" in reason and "(D)" in reason


def test_complete_options_not_flagged() -> None:
    assert _missing_options_reason(_q("1", "ABCD")) is None


def test_too_few_labels_not_flagged() -> None:
    # 0 or 1 label is not a reliable "clipped options" signal.
    assert _missing_options_reason(_q("3", "")) is None
    assert _missing_options_reason(_q("4", "A")) is None


def test_build_analyzed_items_flags_partial_options() -> None:
    detected = [_q("1", "ABCD", y0=5.0, y1=20.0), _q("2", "AC", y0=25.0, y1=40.0)]
    items = build_analyzed_items(detected)
    by_num = {it.q_num: it for it in items}
    assert by_num["1"].flagged is False
    assert by_num["2"].flagged is True
    assert "option" in (by_num["2"].flag_reason or "").lower()


def test_build_review_notes_includes_option_note() -> None:
    # Three items so numbering-gap logic stays quiet; one has clipped options.
    # Distinct rows so they don't trip the overlap check.
    detected = [
        _q("1", "ABCD", y0=5.0, y1=20.0),
        _q("2", "ABCD", y0=25.0, y1=40.0),
        _q("3", "AC", y0=45.0, y1=60.0),
    ]
    notes = build_review_notes(detected, method_used="text")
    assert any(
        n.kind == "incomplete" and n.q_num == "3" and "option" in n.message.lower()
        for n in notes
    ), [(n.kind, n.message) for n in notes]
