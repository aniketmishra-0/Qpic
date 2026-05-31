"""Regression tests for the 2-up option-grid false column split.

Many MCQ papers lay their four options in a 2x2 grid:

    (A) ...        (B) ...
    (C) ...        (D) ...

Repeated down the page, the gap between the left ((A)/(C)) and right ((B)/(D))
options opens a tall vertical whitespace gutter through the page middle. The
column detector reads that gutter as a two-column page layout, which then
confines every question's crop to the *left* column — slicing the (B)/(D)
options off the right half — and balloons the last question to swallow the
orphaned right strip.

The fix (`_validate_columns_with_markers`) only trusts a multi-column split when
question markers actually start in more than one column. An option grid has
every marker in the left column, so the page collapses back to one full-width
column and each crop keeps all four options. A genuine two-column paper (markers
on both sides) is left untouched — covered by ``test_structural_furniture``.
"""

from __future__ import annotations

import fitz

from app.services.detector.base import (
    QuestionStart,
    ContentLine,
    detect_columns,
    _validate_columns_with_markers,
)
from app.services.detector.text_detector import TextDetector

W, H = 595, 842


def _build_option_grid_pdf(num_questions: int = 6) -> bytes:
    """Single-column questions whose four options sit in a 2-up grid."""

    doc = fitz.open()
    page = doc.new_page(width=W, height=H)
    y = 70
    for i in range(1, num_questions + 1):
        page.insert_text((45, y), f"{i}.", fontsize=11)
        page.insert_textbox(
            fitz.Rect(70, y - 11, 290, y + 30),
            f"Question {i} stem text that is moderately long here.",
            fontsize=10,
        )
        y += 34
        # 2-up option grid: left options ~x=70, right options ~x=320.
        page.insert_text((70, y), f"(A) first option {i}", fontsize=10)
        page.insert_text((320, y), f"(B) second option {i}", fontsize=10)
        y += 20
        page.insert_text((70, y), f"(C) third option {i}", fontsize=10)
        page.insert_text((320, y), f"(D) fourth option {i}", fontsize=10)
        y += 42
    data = doc.tobytes()
    doc.close()
    return data


def test_option_grid_is_not_split_into_columns() -> None:
    """Each question's crop spans the full content width, not just the left
    half, so the (B) and (D) options are never clipped."""

    pdf = _build_option_grid_pdf()
    questions = TextDetector().detect(pdf, padding_px=10)
    assert len(questions) == 6

    # The right-hand options sit near x=320/595 ≈ 54%, so a correct crop must
    # extend well past the page middle. A left-column-only crop would stop ~45%.
    for q in questions:
        assert len(q.segments) == 1, (q.q_num, len(q.segments))
        seg = q.segments[0]
        assert seg.x_end_pct > 60.0, (q.q_num, seg.x_end_pct)


def test_option_grid_last_question_not_ballooned() -> None:
    """The final question must not absorb the whole right 'column' (the old bug
    left it spanning nearly the full page height)."""

    pdf = _build_option_grid_pdf()
    questions = TextDetector().detect(pdf, padding_px=10)
    last = max(questions, key=lambda q: int(q.q_num))
    total_height = sum(s.y_end_pct - s.y_start_pct for s in last.segments)
    # A single grid question occupies well under a quarter of the page.
    assert total_height < 25.0, total_height


def test_validate_collapses_single_marker_column() -> None:
    """Two markers both in the left column collapse a false 2-column split."""

    cols = [(0.0, W / 2), (W / 2, float(W))]
    starts = [
        QuestionStart(page_num=1, y_top=70.0, q_num="1", x_left=45.0, x_right=120.0),
        QuestionStart(page_num=1, y_top=200.0, q_num="2", x_left=45.0, x_right=120.0),
    ]
    out = _validate_columns_with_markers(cols, starts, 1, float(W))
    assert out == [(0.0, float(W))]


def test_validate_keeps_two_columns_when_markers_on_both_sides() -> None:
    """Markers starting in both columns are a real two-column page — keep it."""

    cols = [(0.0, W / 2), (W / 2, float(W))]
    starts = [
        QuestionStart(page_num=1, y_top=70.0, q_num="1", x_left=45.0, x_right=120.0),
        QuestionStart(
            page_num=1, y_top=70.0, q_num="2", x_left=W / 2 + 20, x_right=W / 2 + 120
        ),
    ]
    out = _validate_columns_with_markers(cols, starts, 1, float(W))
    assert out == cols


def test_validate_keeps_columns_on_marker_free_continuation_page() -> None:
    """A cross-page continuation page has no markers; its real two columns must
    be preserved so the stitched crop reads both halves."""

    cols = [(0.0, W / 2), (W / 2, float(W))]
    # No starts on page 2.
    starts = [
        QuestionStart(page_num=1, y_top=70.0, q_num="1", x_left=45.0, x_right=120.0),
    ]
    out = _validate_columns_with_markers(cols, starts, 2, float(W))
    assert out == cols


def _build_two_column_solutions_pdf() -> bytes:
    """A genuine two-column solutions page whose markers all sit in the left
    column while the right column carries independent explanation prose.

    Mirrors the "Hints & Solutions" pages of a UPSC DPP: ``Q1 Text Solution``,
    ``Q2 Text Solution`` open in the left column, and the right column is the
    continuation of their explanations (no markers, real sentences — not option
    labels). This must stay two columns, unlike an option grid.
    """

    doc = fitz.open()
    page = doc.new_page(width=W, height=H)
    left_x, right_x = 50, 320

    y = 80
    page.insert_text((left_x, y), "Q1 Text Solution:", fontsize=10); y += 16
    page.insert_text((left_x, y), "Ans: C", fontsize=10); y += 16
    for i in range(6):
        page.insert_text((left_x, y), f"left explanation sentence {i}", fontsize=10)
        y += 16
    y += 10
    page.insert_text((left_x, y), "Q2 Text Solution:", fontsize=10); y += 16
    page.insert_text((left_x, y), "Ans: C", fontsize=10); y += 16
    for i in range(6):
        page.insert_text((left_x, y), f"left q2 sentence {i}", fontsize=10)
        y += 16

    # Right column: independent prose, no markers and no option labels.
    y = 80
    for i in range(20):
        page.insert_text((right_x, y), f"right column explanation prose {i}", fontsize=10)
        y += 16

    data = doc.tobytes()
    doc.close()
    return data


def test_two_column_solutions_page_kept_split() -> None:
    """A real two-column page whose markers cluster in the left column must NOT
    collapse to full width — otherwise each crop stitches unrelated right-column
    text under the left column (the reported autocrop bug)."""

    pdf = _build_two_column_solutions_pdf()
    questions = TextDetector().detect(pdf, padding_px=0)
    assert questions, "expected detected solutions"

    # The first solution must be confined to the left column: its content ends
    # well before the page middle, not spanning into the right column.
    first = min(questions, key=lambda q: int(q.q_num))
    left_seg = min(first.segments, key=lambda s: s.x_start_pct)
    assert left_seg.x_end_pct < 45.0, (first.q_num, left_seg.x_end_pct)


def test_validate_keeps_split_when_other_column_has_prose() -> None:
    """Markers all in one column, but the other column carries independent prose
    → real two-column page, keep the split."""

    cols = [(0.0, W / 2), (W / 2, float(W))]
    starts = [
        QuestionStart(page_num=1, y_top=70.0, q_num="1", x_left=45.0, x_right=120.0),
        QuestionStart(page_num=1, y_top=200.0, q_num="2", x_left=45.0, x_right=120.0),
    ]
    lines = [
        ContentLine(
            page_num=1,
            y_top=70.0 + 14 * i,
            y_bottom=82.0 + 14 * i,
            x_left=W / 2 + 20,
            x_right=W / 2 + 200,
            text=f"independent prose line {i}",
        )
        for i in range(6)
    ]
    out = _validate_columns_with_markers(cols, starts, 1, float(W), lines)
    assert out == cols


def test_validate_collapses_when_other_column_is_only_options() -> None:
    """Markers all in one column and the other column holds only option labels
    → option grid, collapse to full width."""

    cols = [(0.0, W / 2), (W / 2, float(W))]
    starts = [
        QuestionStart(page_num=1, y_top=70.0, q_num="1", x_left=45.0, x_right=120.0),
        QuestionStart(page_num=1, y_top=200.0, q_num="2", x_left=45.0, x_right=120.0),
    ]
    lines = [
        ContentLine(
            page_num=1,
            y_top=70.0 + 14 * i,
            y_bottom=82.0 + 14 * i,
            x_left=W / 2 + 20,
            x_right=W / 2 + 200,
            text=label,
        )
        for i, label in enumerate(["(B) opt", "(D) opt", "(B) opt", "(D) opt"])
    ]
    out = _validate_columns_with_markers(cols, starts, 1, float(W), lines)
    assert out == [(0.0, float(W))]
