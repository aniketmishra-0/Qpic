"""Regression tests for stray horizontal separator/divider rules in crops.

A zero-thickness horizontal rule (a question separator or table border) used to
slip through furniture collection — the guard ``if w <= 0 or h <= 0: continue``
discarded every mathematically-flat line before it could be classed as a rule,
so the divider survived into the cropped image as a stray line. Genuine content
strokes (fraction bars, text underlines) must still be kept.
"""

from __future__ import annotations

import fitz

from app.services.detector.furniture import collect_page_furniture

W, H = 595, 842


def _overlaps(f, bx0, by0, bx1, by1) -> bool:
    return not (f.x1 < bx0 or f.x0 > bx1 or f.y1 < by0 or f.y0 > by1)


def test_full_width_separator_is_furniture() -> None:
    doc = fitz.open()
    p = doc.new_page(width=W, height=H)
    p.insert_text((60, 90), "1. Question one stem.", fontsize=11)
    p.insert_text((70, 110), "(A) a (B) b (C) c (D) d", fontsize=10)
    p.draw_line((40, 140), (555, 140), width=0.8)  # flat full-width divider
    p.insert_text((60, 170), "2. Question two stem.", fontsize=11)

    furn = collect_page_furniture(p)
    # The divider near y=140 spanning the page width must be collected.
    assert any(_overlaps(f, 40, 138, 555, 142) and (f.x1 - f.x0) > 0.5 * W for f in furn), furn


def test_fraction_bar_is_not_furniture() -> None:
    doc = fitz.open()
    p = doc.new_page(width=W, height=H)
    p.insert_text((70, 145), "500", fontsize=11)
    p.draw_line((70, 150), (100, 150), width=1.0)  # short fraction bar
    p.insert_text((78, 165), "3", fontsize=11)

    furn = collect_page_furniture(p)
    assert not any(_overlaps(f, 70, 148, 100, 152) for f in furn), furn


def test_text_underline_is_not_furniture() -> None:
    doc = fitz.open()
    p = doc.new_page(width=W, height=H)
    p.insert_text((60, 100), "This phrase is underlined for emphasis here.", fontsize=11)
    # Underline directly beneath the text line, overlapping it horizontally.
    p.draw_line((60, 104), (300, 104), width=0.8)

    furn = collect_page_furniture(p)
    assert not any(_overlaps(f, 60, 102, 300, 106) for f in furn), furn
