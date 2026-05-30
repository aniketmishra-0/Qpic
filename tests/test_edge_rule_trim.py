"""Regression test for the stray side line on crops.

Some papers (e.g. PW solution cards) draw a thin decorative accent rule down
the side of a question. Because it is a vector graphic with no text, the
detectors never measure it, so the crop's horizontal padding pulls the bar into
the image. ``trim_edge_rules`` removes such an isolated thin+solid vertical
strip while leaving clean crops untouched.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from app.utils.image_utils import trim_edge_rules


def _edge_rule_present(img: Image.Image, edge_frac: float = 0.08) -> bool:
    """True if any near-edge column is inked for more than half the height."""

    arr = np.asarray(img.convert("RGB")).astype(int)
    col_cov = (arr.min(axis=2) < 235).mean(axis=0)
    edge = max(1, int(img.width * edge_frac))
    return bool((col_cov[:edge] > 0.5).any() or (col_cov[-edge:] > 0.5).any())


def _make(with_rule: bool, w: int = 1591, h: int = 2229) -> Image.Image:
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    top, bot = int(h * 0.05), int(h * 0.94)
    if with_rule:
        # PW-style accent bar near the left edge, faint colour, then a gutter.
        d.rectangle([45, top, 56, bot], fill=(194, 179, 180))
    for y in range(top, bot, 60):  # body text well past the gutter
        d.rectangle([120, y, 120 + 1200, y + 26], fill=(15, 15, 15))
    return img


def test_side_accent_rule_is_trimmed() -> None:
    ruled = _make(with_rule=True)
    assert _edge_rule_present(ruled) is True
    trimmed = trim_edge_rules(ruled)
    assert _edge_rule_present(trimmed) is False
    # Content is preserved: only a thin strip is removed, height is untouched.
    assert trimmed.height == ruled.height
    assert trimmed.width < ruled.width
    assert ruled.width - trimmed.width < int(ruled.width * 0.1)


def test_clean_crop_is_untouched() -> None:
    clean = _make(with_rule=False)
    out = trim_edge_rules(clean)
    assert out.size == clean.size
    assert np.array_equal(np.asarray(out), np.asarray(clean))


def test_body_text_rows_are_not_trimmed_as_rules() -> None:
    """A line of body text is a thin solid horizontal strip; trimming must not
    mistake it for a top/bottom rule and eat content."""

    clean = _make(with_rule=False)
    out = trim_edge_rules(clean)
    assert out.height == clean.height
