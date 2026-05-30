"""Tests for collapsing near-identical crops in the finalize flow.

The review popup pre-fills auto-detected items, and the user can draw a box on
top of one to fix it. If that drawn box is added as a *new* item (with a fresh
auto number) instead of replacing the original, the same physical question ends
up in the output twice — the "22 questions show as 25 with duplicates" bug.

``_dedupe_by_overlap`` is the backend safety net: it collapses items of the same
kind whose regions overlap substantially, keeping the larger-extent crop.
"""

from __future__ import annotations

import os

os.environ["ANTHROPIC_API_KEY"] = ""

from app.routers.crop import _dedupe_by_overlap  # noqa: E402
from app.models.schemas import DetectedQuestion, QuestionSegment  # noqa: E402


def _q(num: str, y0: float, y1: float, *, page: int = 1, is_solution: bool = False) -> DetectedQuestion:
    return DetectedQuestion(
        q_num=num,
        is_solution=is_solution,
        segments=[QuestionSegment(page=page, x_start_pct=5, x_end_pct=95, y_start_pct=y0, y_end_pct=y1)],
    )


def test_overlapping_boxes_with_different_numbers_collapse() -> None:
    # An auto "3" and a hand-drawn box over the same spot that got auto-numbered
    # "23" are the same question -> only one survives, the taller (more complete).
    detected = [
        _q("3", 30.0, 45.0),
        _q("23", 29.0, 50.0),  # overlaps "3" heavily, slightly taller
    ]
    out = _dedupe_by_overlap(detected)
    assert len(out) == 1
    assert out[0].q_num == "23"


def test_adjacent_questions_are_kept() -> None:
    # Two stacked questions touch at an edge but don't overlap -> both kept.
    detected = [
        _q("1", 10.0, 30.0),
        _q("2", 30.0, 50.0),
    ]
    out = _dedupe_by_overlap(detected)
    assert len(out) == 2


def test_question_and_solution_at_same_spot_both_kept() -> None:
    # Same region but different side (question vs solution) -> not a duplicate.
    detected = [
        _q("1", 10.0, 40.0, is_solution=False),
        _q("1", 10.0, 40.0, is_solution=True),
    ]
    out = _dedupe_by_overlap(detected)
    assert len(out) == 2


def test_same_region_different_pages_both_kept() -> None:
    detected = [
        _q("1", 10.0, 40.0, page=1),
        _q("1", 10.0, 40.0, page=2),
    ]
    out = _dedupe_by_overlap(detected)
    assert len(out) == 2
