"""Image helper utilities."""

from __future__ import annotations

import base64
import io

from PIL import Image


def ensure_rgb(img: Image.Image) -> Image.Image:
    """Convert to RGB if not already (handles RGBA, L, P modes)."""

    if img.mode == "RGB":
        return img
    return img.convert("RGB")


# --- Edge-rule trimming ------------------------------------------------------
#
# Some exam papers (e.g. PW solution cards) draw a thin decorative accent rule
# down the side of a question/solution. Because the rule is a *vector graphic*
# with no text, the text/OCR detectors never see it — they only measure the
# horizontal extent of words — so the crop's content bounds + padding can reach
# past the whitespace gutter and pull that bar into the image. The result is the
# stray vertical line the user sees on the side of a crop.
#
# We remove it as a post-render step: scan inward from each edge and, if the
# first thing we meet is a thin *solid* strip (a ruled line) that is separated
# from the real content by a whitespace gap, cut the strip off. The whitespace
# gap is kept so the crop retains a natural margin.

# A row/column counts as "inked" when at least this fraction of its pixels are
# non-white.
_RULE_ACTIVE_COVERAGE = 0.03

# Pixel is "non-white" when its darkest channel falls below this. Catches both
# black text and faint coloured rules (the PW accent bar is ~RGB(194,179,180)).
_RULE_NONWHITE_LEVEL = 235

# A leading/trailing inked strip is treated as a ruled line only when it is at
# least this solid (fraction of its length that is inked) ...
_RULE_SOLID_COVERAGE = 0.45

# ... and no thicker than this fraction of the image dimension (rules are thin;
# real content blocks are much wider/taller).
_RULE_MAX_THICKNESS_FRAC = 0.05

# A genuine separated rule has a whitespace gap of at least this fraction of the
# dimension between it and the real content. Requiring the gap prevents us from
# eating the leading stroke of an actual glyph.
_RULE_MIN_GAP_FRAC = 0.004


def _leading_rule_cut(coverage: list[float], length: int) -> int:
    """Return how many leading units (cols/rows) to drop from an edge.

    ``coverage[i]`` is the inked fraction of column/row ``i``. We trim only when
    the edge begins with optional whitespace, then a thin *solid* strip (the
    rule), then a whitespace gap before the content. Otherwise we trim nothing.
    The returned cut keeps the whitespace gap as a margin.
    """

    strip = _leading_rule_strip(coverage, length)
    return strip[1] if strip is not None else 0


def _leading_rule_strip(
    coverage: list[float], length: int
) -> "tuple[int, int] | None":
    """Return ``(run_start, run_end)`` of a leading edge rule, or None.

    Same detection as :func:`_leading_rule_cut` (skip leading whitespace, find a
    thin solid strip, require a whitespace gap before the content) but exposes
    the strip's column range so the caller can tell a lone decorative accent bar
    from the side border of a content box (which has horizontal corner
    connectors).
    """

    if length <= 0:
        return None

    max_thickness = max(2, int(_RULE_MAX_THICKNESS_FRAC * length))
    min_gap = max(2, int(_RULE_MIN_GAP_FRAC * length))

    i = 0
    # Skip any leading whitespace (e.g. left padding before the rule).
    while i < length and coverage[i] < _RULE_ACTIVE_COVERAGE:
        i += 1
    if i >= length:
        return None  # blank strip — nothing to do

    run_start = i
    while i < length and coverage[i] >= _RULE_ACTIVE_COVERAGE:
        i += 1
    run_end = i  # exclusive

    run_width = run_end - run_start
    if run_width > max_thickness:
        return None  # too thick to be a rule — this is real content

    run = coverage[run_start:run_end]
    if (sum(run) / float(run_width)) < _RULE_SOLID_COVERAGE:
        return None  # not solid enough to be a ruled line

    # Require a whitespace gap between the rule and the content.
    gap = 0
    j = run_end
    while j < length and coverage[j] < _RULE_ACTIVE_COVERAGE:
        gap += 1
        j += 1
    if gap < min_gap or j >= length:
        return None

    return (run_start, run_end)


# A horizontal corner connector (the top/bottom border of a content box meeting
# its side border) must extend inward from the side strip by at least this
# fraction of the crop width to count. A lone decorative accent bar has no such
# connector; a framed note-box does, at both ends.
_BOX_CORNER_CONNECTOR_FRAC = 0.04


def _strip_is_box_side(nonwhite, run_start: int, run_end: int, *, from_left: bool) -> bool:
    """True if a leading vertical strip is the side border of a content box.

    A content box ("Additional Information", "PW SUPER HINT") frames text in a
    rectangle, so its side border meets a horizontal top/bottom border at the
    corners. We confirm this by checking that, at the topmost or bottommost
    inked row of the strip, ink continues *inward* (away from the edge) in an
    unbroken horizontal run — a corner.

    A corner at *either* end is enough. A box split by a page break leaves a
    half-frame with a corner at only one end (the other end is the cut), and
    that half must still be recognised as a box side so the side border of a
    cross-page note-box survives stitching. A lone decorative accent bar has no
    corner at either end and is therefore still trimmed.
    """

    import numpy as np

    h, w, *_ = nonwhite.shape
    # Map the strip to absolute columns (the right side scans a reversed view).
    # ``inward_rows`` is the region just past the strip, column-ordered so that
    # index 0 is the column adjacent to the strip and increasing index moves
    # *inward* (away from the edge). Building it once lets each corner check be a
    # vectorized leading-run on a row instead of a per-pixel Python loop.
    if from_left:
        a, b = run_start, run_end  # [a, b) absolute columns
        inward_rows = nonwhite[:, b:w]
    else:
        a, b = w - run_end, w - run_start
        inward_rows = nonwhite[:, :a][:, ::-1]

    strip = nonwhite[:, a:b]
    rows = np.where(strip.any(axis=1))[0]
    if rows.size == 0:
        return False
    # The strip must span a tall region (a true border, not a stray mark).
    if (rows.max() - rows.min() + 1) < 0.30 * h:
        return False

    min_run = max(3, int(_BOX_CORNER_CONNECTOR_FRAC * w))

    def _has_corner(row: int) -> bool:
        # A corner is an unbroken inward run of ink (the box's horizontal
        # border) starting at the strip edge, on any row within ±2 of the strip
        # end. The leading-run length is the index of the first non-ink column,
        # which reproduces the original "count until the first gap" scan.
        for r in range(max(0, row - 2), min(h, row + 3)):
            line = inward_rows[r]
            if line.size == 0:
                continue
            false_idx = np.flatnonzero(~line)
            run = int(false_idx[0]) if false_idx.size else int(line.size)
            if run >= min_run:
                return True
        return False

    return _has_corner(int(rows.min())) or _has_corner(int(rows.max()))


def trim_edge_rules(img: Image.Image) -> Image.Image:
    """Strip thin decorative ruled lines hugging any edge of a crop.

    Only an isolated thin+solid strip separated from the content by whitespace
    is removed; whitespace margins and the content itself are left untouched, so
    a crop with no edge rule is returned unchanged. The side border of a framed
    content box is preserved — it meets the box's horizontal borders at the
    corners, which a lone decorative accent bar never does.
    """

    try:
        import numpy as np
    except Exception:
        return img

    rgb = ensure_rgb(img)
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[0] < 4 or arr.shape[1] < 4:
        return img

    h, w, _ = arr.shape
    nonwhite = arr.min(axis=2) < _RULE_NONWHITE_LEVEL  # (h, w) bool

    col_cov = nonwhite.mean(axis=0).tolist()  # per-column inked fraction

    # Only side (vertical) rules are trimmed. A horizontal thin+solid strip is
    # indistinguishable from a line of body text (both are thin, solid, and
    # followed by a whitespace gap), so trimming top/bottom would eat real
    # content. A vertical accent rule, by contrast, is a single column inked for
    # most of the page height — something text never produces — so left/right
    # detection is safe.
    left_strip = _leading_rule_strip(col_cov, w)
    right_strip = _leading_rule_strip(col_cov[::-1], w)

    left = 0
    if left_strip is not None and not _strip_is_box_side(
        nonwhite, left_strip[0], left_strip[1], from_left=True
    ):
        left = left_strip[1]

    right = 0
    if right_strip is not None and not _strip_is_box_side(
        nonwhite, right_strip[0], right_strip[1], from_left=False
    ):
        right = right_strip[1]

    x0 = left
    x1 = w - right

    if left <= 0 and right <= 0:
        return img
    if x1 - x0 < 2:
        return img  # would collapse the crop — bail out

    return rgb.crop((x0, 0, x1, h))


def img_to_base64_png(img: Image.Image) -> str:
    """Convert PIL Image to base64 PNG string (for Anthropic API)."""

    buff = io.BytesIO()
    ensure_rgb(img).save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")


def resize_for_api(img: Image.Image, max_px: int = 1568) -> Image.Image:
    """Resize image so the longest side is <= max_px without upscaling."""

    width, height = img.size
    longest = max(width, height)
    if longest <= max_px:
        return img

    scale = max_px / float(longest)
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    return img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
