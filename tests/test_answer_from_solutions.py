"""Tests for reading answers out of solution write-ups (no compact grid)."""

from __future__ import annotations

from app.services.detector.answer_key import (
    extract_answer_from_solution_text,
    extract_answers_from_solution_section,
    extract_answer_key_from_text,
)


def test_single_solution_answer_labels():
    assert extract_answer_from_solution_text("Long reasoning. Ans. (B)") == "B"
    assert extract_answer_from_solution_text("Answer: C") == "C"
    assert extract_answer_from_solution_text("The correct option is (D)") == "D"
    assert extract_answer_from_solution_text("Sol: A") == "A"


def test_single_solution_no_label_returns_none():
    # A stray option label in prose without an answer word is not an answer.
    assert extract_answer_from_solution_text("As shown in (B) the graph rises") is None
    assert extract_answer_from_solution_text("just an explanation") is None


def test_section_with_ans_labels():
    text = (
        "1. Explanation one. Ans. (B)\n"
        "2. Explanation two. Answer: C\n"
        "3. Explanation three. Ans (D)\n"
        "4. Explanation four. Correct option: A"
    )
    assert extract_answers_from_solution_section(text) == {1: "B", 2: "C", 3: "D", 4: "A"}


def test_section_with_q_and_sol_prefixes():
    assert extract_answers_from_solution_section(
        "Q1. blah Answer (C)\nQ2. text Ans: A\nQ3. stuff Answer (D)"
    ) == {1: "C", 2: "A", 3: "D"}
    assert extract_answers_from_solution_section(
        "Sol. 1 reasoning Ans (A)\nSol. 2 reasoning Answer B\nSol. 3 Ans. (C)"
    ) == {1: "A", 2: "B", 3: "C"}


def test_section_correct_answer_is_phrasing():
    text = (
        "1) The correct answer is (B) because...\n"
        "2) The correct answer is (D)\n"
        "3) Correct answer is A"
    )
    assert extract_answers_from_solution_section(text) == {1: "B", 2: "D", 3: "A"}


def test_section_too_sparse_returns_empty():
    # Fewer than 2 labelled answers -> not trusted.
    assert extract_answers_from_solution_section("1. only one. Ans (A)\n2. none here") == {}


def test_section_ignores_unlabelled_prose():
    assert extract_answers_from_solution_section(
        "1. This explains the concept.\n2. Another explanation."
    ) == {}


def test_short_grid_now_accepted():
    # Threshold lowered to 4 so a small (5-question) key is no longer rejected.
    assert extract_answer_key_from_text("1. (B)  2. (A)  3. (D)  4. (C)  5. (A)") == {
        1: "B",
        2: "A",
        3: "D",
        4: "C",
        5: "A",
    }
