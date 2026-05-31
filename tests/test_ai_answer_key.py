"""Tests for the AI answer-key vision reader's parsing/coercion."""

from __future__ import annotations

from app.services.detector.ai_answer_key import _coerce_answer_map


def test_coerce_handles_answers_wrapper():
    data = {"answers": {"1": "B", "2": "a", "3": "(D)"}}
    assert _coerce_answer_map(data) == {1: "B", 2: "A", 3: "D"}


def test_coerce_handles_bare_map():
    assert _coerce_answer_map({"1": "C"}) == {1: "C"}


def test_coerce_drops_invalid_entries():
    data = {"answers": {"1": "Z", "two": "B", "3": "", "4": "A"}}
    # "Z" not A-E, "two" has no digits, "" empty -> only 4 survives.
    assert _coerce_answer_map(data) == {4: "A"}


def test_coerce_first_value_wins_on_duplicate():
    data = {"answers": {"1": "A"}}
    out = _coerce_answer_map(data)
    assert out == {1: "A"}


def test_coerce_rejects_out_of_range_numbers():
    assert _coerce_answer_map({"answers": {"0": "A", "1000": "B", "5": "C"}}) == {5: "C"}


def test_coerce_extracts_letter_embedded_in_text():
    assert _coerce_answer_map({"answers": {"7": "Option B"}}) == {7: "B"}


def test_coerce_non_dict_returns_empty():
    assert _coerce_answer_map([1, 2, 3]) == {}
    assert _coerce_answer_map("nope") == {}
