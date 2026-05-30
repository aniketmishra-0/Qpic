"""Tests for dropping phantom question numbers from inline values.

A detector (most often the AI vision tier, which never runs through the marker
matcher) sometimes mistakes an inline number for a question marker: an angle or
constant inside an equation ("E3 = 5E0 cos(wt + 53)"), a year, a quantity. That
surfaces as a lone item whose number is wildly out of sequence — a paper
numbered 1..6 plus a stray "Q53" spanning two pages (the reported bug).

``drop_phantom_numbers`` peels such isolated top outliers off each side's run,
while leaving genuine high-numbered questions and tightly-packed runs intact.
"""

from __future__ import annotations

import os

os.environ["ANTHROPIC_API_KEY"] = ""

from app.models.schemas import DetectedQuestion, QuestionSegment  # noqa: E402
from app.services.review_service import drop_phantom_numbers  # noqa: E402


def _q(
    num: str,
    *,
    page: int = 1,
    is_solution: bool = False,
    y0: float = 10.0,
    y1: float = 40.0,
) -> DetectedQuestion:
    return DetectedQuestion(
        q_num=num,
        is_solution=is_solution,
        segments=[
            QuestionSegment(
                page=page, x_start_pct=5, x_end_pct=95, y_start_pct=y0, y_end_pct=y1
            )
        ],
    )


def test_drops_lone_spike_above_dense_run() -> None:
    # 1..5 are real; the "53" is an angle from Q4's equation misread as a marker.
    detected = [_q(str(n)) for n in (1, 2, 3, 4, 5)] + [_q("53")]
    out = drop_phantom_numbers(detected)
    nums = sorted(int(q.q_num) for q in out)
    assert nums == [1, 2, 3, 4, 5]


def test_keeps_consecutive_high_numbers() -> None:
    # A run that simply ends high (no big gap) is real and must be kept.
    detected = [_q(str(n)) for n in (48, 49, 50, 51, 52, 53)]
    out = drop_phantom_numbers(detected)
    assert len(out) == 6


def test_keeps_high_number_without_solid_run_below() -> None:
    # Only two numbers below the gap -> not enough run to trust dropping "53".
    detected = [_q("1"), _q("2"), _q("53")]
    out = drop_phantom_numbers(detected)
    assert {int(q.q_num) for q in out} == {1, 2, 53}


def test_peels_two_stacked_spikes() -> None:
    detected = [_q(str(n)) for n in (1, 2, 3, 4, 5)] + [_q("53"), _q("108")]
    out = drop_phantom_numbers(detected)
    assert sorted(int(q.q_num) for q in out) == [1, 2, 3, 4, 5]


def test_questions_and_solutions_judged_independently() -> None:
    # A real solution numbered like its question must not be dropped just
    # because the question side has a spike, and vice versa.
    detected = (
        [_q(str(n)) for n in (1, 2, 3, 4, 5)]
        + [_q("53")]  # phantom question
        + [_q(str(n), is_solution=True) for n in (1, 2, 3, 4, 5)]
    )
    out = drop_phantom_numbers(detected)
    q_nums = sorted(int(q.q_num) for q in out if not q.is_solution)
    s_nums = sorted(int(q.q_num) for q in out if q.is_solution)
    assert q_nums == [1, 2, 3, 4, 5]
    assert s_nums == [1, 2, 3, 4, 5]


def test_keeps_item_without_number() -> None:
    detected = [_q(str(n)) for n in (1, 2, 3, 4, 5)] + [_q("")]
    out = drop_phantom_numbers(detected)
    # The numberless item is always kept; nothing else is a spike.
    assert any(q.q_num == "" for q in out)
    assert len(out) == 6


def test_no_change_on_clean_run() -> None:
    detected = [_q(str(n)) for n in (1, 2, 3, 4, 5, 6)]
    out = drop_phantom_numbers(detected)
    assert len(out) == 6


def test_gap_just_below_threshold_is_kept() -> None:
    # 5 -> 14 is a gap of 9 (< 10): treated as legitimate numbering, kept.
    detected = [_q(str(n)) for n in (1, 2, 3, 4, 5)] + [_q("14")]
    out = drop_phantom_numbers(detected)
    assert {int(q.q_num) for q in out} == {1, 2, 3, 4, 5, 14}
