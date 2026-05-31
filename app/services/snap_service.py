"""Snap a roughly-drawn box to the real content inside it.

When a user hand-draws a crop in the review popup they rarely trace the text
tightly — they sweep a loose rectangle that includes whitespace margins (or
slightly clips an edge). ``snap_region`` renders just that rectangle from the
PDF, finds the bounding box of the actual ink inside it, and returns a tightened
region (with a small margin) in page-percentage coordinates.

This makes manual selection feel "smart": a sloppy drag becomes a clean,
content-hugging crop that matches the look of the auto-detected ones.
"""

from __future__ import annotations

import logging

import fitz

logger = logging.getLogger(__name__)

# Render DPI for the analysis pass. Low enough to be fast, high enough to
# resolve thin strokes when locating ink.
_SNAP_DPI = 110

# A pixel counts as "ink" when its darkest channel is below this (catches black
# text and mid-tone coloured rules/figures).
_INK_LEVEL = 235

# A row/column counts as occupied when at least this fraction of it is ink.
# Small but non-zero so a stray speck/scan noise doesn't anchor the bound.
_ROW_INK_FRAC = 0.004
_COL_INK_FRAC = 0.004

# Margin (in % of the page) added back around the tightened content so the crop
# isn't flush against the glyphs.
_MARGIN_PCT = 0.8


def snap_region(
    pdf_bytes: bytes,
    page_index: int,
    *,
    x_start_pct: float,
    x_end_pct: float,
    y_start_pct: float,
    y_end_pct: float,
) -> dict:
    """Return a content-tightened region for a drawn box.

    Input/output coordinates are percentages of the page (0..100). If anything
    goes wrong, or the region is blank, the original box is returned unchanged so
    snapping can never make a selection worse.
    """

    original = {
        "x_start_pct": _clamp(x_start_pct),
        "x_end_pct": _clamp(x_end_pct),
        "y_start_pct": _clamp(y_start_pct),
        "y_end_pct": _clamp(y_end_pct),
    }

    try:
        import numpy as np
    except Exception:
        return original

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            if page_index < 0 or page_index >= doc.page_count:
                return original
            page = doc.load_page(page_index)
            rect = page.rect
            pw, ph = float(rect.width), float(rect.height)
            if pw <= 0 or ph <= 0:
                return original

            x0 = rect.x0 + (original["x_start_pct"] / 100.0) * pw
            x1 = rect.x0 + (original["x_end_pct"] / 100.0) * pw
            y0 = rect.y0 + (original["y_start_pct"] / 100.0) * ph
            y1 = rect.y0 + (original["y_end_pct"] / 100.0) * ph
            if x1 - x0 < 1.0 or y1 - y0 < 1.0:
                return original

            zoom = _SNAP_DPI / 72.0
            clip = fitz.Rect(x0, y0, x1, y1)
            pix = page.get_pixmap(
                matrix=fitz.Matrix(zoom, zoom), clip=clip, colorspace=fitz.csRGB, alpha=False
            )
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    except Exception as exc:
        logger.debug("snap_render_failed page=%s error=%s", page_index, str(exc))
        return original

    h, w = arr.shape[0], arr.shape[1]
    if h < 4 or w < 4:
        return original

    ink = arr.min(axis=2) < _INK_LEVEL  # (h, w) bool
    row_occ = ink.mean(axis=1) >= _ROW_INK_FRAC
    col_occ = ink.mean(axis=0) >= _COL_INK_FRAC

    rows = _np_true_range(row_occ)
    cols = _np_true_range(col_occ)
    if rows is None or cols is None:
        # Blank region — keep the user's box rather than collapsing it.
        return original

    r_top, r_bot = rows
    c_left, c_right = cols

    # Map the tightened pixel bounds back to page percentages, relative to the
    # rendered clip (which started at the drawn box's top-left).
    box_w_pct = original["x_end_pct"] - original["x_start_pct"]
    box_h_pct = original["y_end_pct"] - original["y_start_pct"]

    # Vertical: tighten to the actual content (with a small margin).
    new_y_start = original["y_start_pct"] + (r_top / h) * box_h_pct - _MARGIN_PCT
    new_y_end = original["y_start_pct"] + ((r_bot + 1) / h) * box_h_pct + _MARGIN_PCT

    # Horizontal: KEEP the left edge exactly where the user drew it. That drawn
    # left edge is the alignment reference — every part of a column-split
    # question (the stem + "(A)/(B)" box and the spilled "(C)/(D)" box) is drawn
    # at the same left x, so preserving it and stitching flush-left keeps the
    # option indentation identical. Snapping the left edge to ink/column moved it
    # per-part (stem vs "(C)") and made "(C)/(D)" drift off "(A)/(B)".
    new_x_start = original["x_start_pct"]
    # Right edge: tighten to content (with a margin) so the box doesn't balloon
    # out to grab the empty right half.
    new_x_end = original["x_start_pct"] + ((c_right + 1) / w) * box_w_pct + _MARGIN_PCT

    snapped = {
        "x_start_pct": _clamp(min(new_x_start, original["x_end_pct"])),
        "x_end_pct": _clamp(max(new_x_end, original["x_start_pct"])),
        "y_start_pct": _clamp(min(new_y_start, original["y_end_pct"])),
        "y_end_pct": _clamp(max(new_y_end, original["y_start_pct"])),
    }

    if snapped["x_end_pct"] - snapped["x_start_pct"] < 1.0:
        snapped["x_start_pct"], snapped["x_end_pct"] = original["x_start_pct"], original["x_end_pct"]
    if snapped["y_end_pct"] - snapped["y_start_pct"] < 1.0:
        snapped["y_start_pct"], snapped["y_end_pct"] = original["y_start_pct"], original["y_end_pct"]

    return snapped


def _np_true_range(mask) -> "tuple[int, int] | None":
    import numpy as np

    idx = np.where(mask)[0]
    if idx.size == 0:
        return None
    return int(idx.min()), int(idx.max())


def _clamp(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 100.0:
        return 100.0
    return float(v)
