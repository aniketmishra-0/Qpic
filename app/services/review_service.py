"""Heuristics that decide what a human should double-check after detection.

The smart pipeline does the heavy lifting, but on unusual PDFs it can still:

  * crop only *part* of a question (a half/cut-off crop that stops before the
    options, or that continues onto the next page but wasn't stitched),
  * emit two items that are really the same question (a duplicate), or
  * miss a question entirely, leaving a *gap* in the numbering.

Rather than silently shipping those, we surface them as review *notes* and
*flag* the suspect items so the UI can highlight them. The user can then
**re-select** the correct full region for a cut-off item (or draw a missed one)
and the app re-crops from the corrected box — nothing here mutates the
detection, it only annotates it.
"""

from __future__ import annotations

import logging
import re
from statistics import median
from typing import Iterable

from ..models.schemas import AnalyzedItem, DetectedQuestion, QuestionSegment, ReviewNote

logger = logging.getLogger(__name__)


# --- Phantom-number filtering ------------------------------------------------
#
# A lone question/solution number that sits far above the rest of its run is
# almost never a real question — it is an *inline* number that a detector (most
# often the AI vision tier, which doesn't go through the marker matcher) mistook
# for a marker: an equation constant ("E3 = 5E0 cos(wt + 53)"), a year, a
# physical quantity. The give-away is that the side's real numbers form a dense
# run (1,2,3,4,5,…) and then there is one isolated value separated by a big gap
# (…,5, 53). We peel those lone top outliers off so the bogus item (e.g. a
# spurious "Q53" spanning two pages) is never cropped, shown, or counted.

# Minimum gap between the outlier and the next-highest number for it to count as
# isolated. A real paper that simply starts high or has a cluster of high
# numbers keeps consecutive values (gap ~1), so this only catches true spikes.
_PHANTOM_GAP = 10

# We only trust the run enough to drop an outlier when at least this many
# distinct numbers sit *below* the gap. Without a solid run beneath it, a high
# number might legitimately be the paper's real numbering, so we leave it.
_PHANTOM_MIN_RUN_BELOW = 4


# A crop whose total vertical extent is below this fraction of one page is
# suspiciously small for a real question/solution (likely a bare marker that
# grabbed no body text). Absolute floor, used when there aren't enough siblings
# to compare against.
_TINY_EXTENT_PCT = 4.0

# A crop shorter than this fraction of the *median* crop on its side
# (questions vs solutions) is probably cut off — its neighbours are all much
# taller, so it likely lost its options or its lower half.
_SHORT_VS_MEDIAN_FRAC = 0.45

# A crop *taller* than this multiple of the median on its side is the opposite
# problem: it probably swallowed its neighbour (two questions merged into one
# box, or a crop that ran past its own end into the next item). Tall outliers
# are just as suspicious as short ones — they were simply never checked before,
# which is why an over-grown item could slip through unflagged.
_TALL_VS_MEDIAN_FRAC = 1.9

# ...but only once the crop is genuinely large in absolute terms. In a paper of
# small crops (median ~6%) a normal 12% question is 2x the median yet perfectly
# fine, so we don't want to cry "merged" on it. A real two-question merge is big.
_TALL_MIN_EXTENT_PCT = 45.0

# A single-segment crop whose bottom sits at/below this % of the page height
# very likely continues onto the next page (it was cut at the page edge and not
# stitched). Treated as a "may continue" cut-off signal.
_PAGE_BOTTOM_PCT = 92.0

# ... but only when the crop is also not clearly a full, tall question. A short
# crop ending at the page bottom is the strong cut-off case.
_PAGE_BOTTOM_MAX_EXTENT_PCT = 35.0


def _q_number(q_num: str) -> int | None:
    digits = re.findall(r"\d+", q_num or "")
    return int(digits[0]) if digits else None


def _extent_pct(item: DetectedQuestion | AnalyzedItem) -> float:
    """Total vertical coverage of an item across all its segments (in % of a page)."""

    return sum(max(0.0, s.y_end_pct - s.y_start_pct) for s in item.segments)


def _primary_page(item: DetectedQuestion | AnalyzedItem) -> int:
    if not item.segments:
        return 0
    return min(s.page for s in item.segments)


def _ends_at_page_bottom(item: DetectedQuestion | AnalyzedItem) -> bool:
    """True if a single-segment crop stops right at the page bottom edge."""

    if len(item.segments) != 1:
        return False
    return item.segments[0].y_end_pct >= _PAGE_BOTTOM_PCT


def _median_extent(detected: Iterable[DetectedQuestion], is_solution: bool) -> float:
    vals = [_extent_pct(q) for q in detected if bool(q.is_solution) == is_solution]
    return float(median(vals)) if vals else 0.0


# Two crops on the same side that physically overlap on the page are a strong,
# content-free sign something went wrong: one box ran into its neighbour, so the
# shared strip belongs to two items at once. Either crop can look perfectly
# normal in height (so the short/tall checks miss it) yet still be wrong — this
# is the "looks fine but isn't" case. We only count a *meaningful* overlap to
# avoid flagging crops that merely touch at a 1px boundary.
_OVERLAP_MIN_PCT = 6.0


def _segments_overlap(
    a: QuestionSegment, b: QuestionSegment, min_pct: float = _OVERLAP_MIN_PCT
) -> bool:
    """True if two segments share the same page and a real 2-D overlap."""

    if a.page != b.page:
        return False
    # Horizontal overlap (columns): needed so a 2-up layout's left/right
    # questions aren't treated as overlapping just because their rows line up.
    x_overlap = min(a.x_end_pct, b.x_end_pct) - max(a.x_start_pct, b.x_start_pct)
    if x_overlap <= 0:
        return False
    y_overlap = min(a.y_end_pct, b.y_end_pct) - max(a.y_start_pct, b.y_start_pct)
    return y_overlap >= min_pct


def find_overlapping_q_nums(
    detected: Iterable[DetectedQuestion],
) -> set[tuple[bool, str]]:
    """Return ``(is_solution, q_num)`` keys for items that overlap a sibling.

    Questions and solutions are checked independently — a question crop sitting
    over a solution crop on a mixed page is expected and not an error.
    """

    items = list(detected)
    overlapping: set[tuple[bool, str]] = set()
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            if bool(a.is_solution) != bool(b.is_solution):
                continue
            if any(_segments_overlap(sa, sb) for sa in a.segments for sb in b.segments):
                overlapping.add((bool(a.is_solution), a.q_num))
                overlapping.add((bool(b.is_solution), b.q_num))
    return overlapping


# --- Content-coverage check --------------------------------------------------
#
# The strongest "looks fine but isn't" signal is content-based, not shape-based:
# a crop can be a perfectly normal-looking box yet still stop short, leaving a
# block of the item's own body text *below* it that no crop covers at all (the
# classic half-cropped solution). Shape heuristics (too-short, too-tall,
# overlap) all miss this because the box itself looks ordinary. So we look at the
# page's actual text lines and check every line is covered by some crop; a tall
# band of uncovered lines sitting directly beneath a crop means that crop was
# cut off and should be re-selected.

# A ``ContentLine`` for coverage purposes: vertical + horizontal extent of one
# text line on a page, all in page-percentage units.
#   (y_top_pct, y_bottom_pct, x_left_pct, x_right_pct)
PageLines = dict  # type alias hint: dict[int, list[tuple[float, float, float, float]]]

# Vertical slack (in % of page height) when deciding whether a crop already
# covers a line — a line whose centre sits within this much of a crop edge
# counts as covered, absorbing tiny rounding between detection and rendering.
_COVER_Y_PAD = 1.5

# Total height of uncovered body text directly below a crop before we call that
# crop cut-off. A couple of stray lines (a footer the crop intentionally
# excluded) stay quiet; a real lost half is a tall run.
_UNCOVERED_MIN_BAND_PCT = 8.0

# Ignore lines in the top/bottom margins entirely — running headers/footers and
# page numbers live there and are deliberately never cropped.
_BODY_TOP_PCT = 6.0
_BODY_BOTTOM_PCT = 94.0

# If the gap between a crop's bottom and the first uncovered line is larger than
# this, the uncovered text is more likely a separately *missed* item (a gap,
# handled by numbering) than the crop's own lost tail, so we don't blame the
# crop above.
_UNCOVERED_MAX_GAP_PCT = 14.0

# Total height of uncovered body text directly ABOVE a column's topmost crop
# before we call that crop's head cut off. This is the mirror of the below-only
# band, but deliberately lower: when a question spilled across a page break the
# stranded head is often a single statement or a lone option line (e.g.
# "(D) Neither 1 nor 2") that is only ~2-3% of the page tall, yet still wrong.
# The topmost-in-column + close-gap constraints keep this from firing on stray
# lines, so a low threshold is safe here.
_HEAD_MIN_BAND_PCT = 2.0

# A page carrying at least this much body text that NO crop covers (and that
# isn't merely the head/tail of an existing crop) almost certainly lost a whole
# question or solution. Kept high so a single stranded footer/branding line
# ("Android App | iOS App | PW Website") never trips it — only a real block of
# missed content does.
_ORPHAN_MIN_BAND_PCT = 6.0


def _line_center_x(line: tuple[float, float, float, float]) -> float:
    return (line[2] + line[3]) / 2.0


def _seg_covers_line(
    seg: QuestionSegment, line: tuple[float, float, float, float]
) -> bool:
    """True if a crop segment vertically + horizontally contains a text line."""

    cx = _line_center_x(line)
    if not (seg.x_start_pct <= cx <= seg.x_end_pct):
        return False
    cy = (line[0] + line[1]) / 2.0
    return (seg.y_start_pct - _COVER_Y_PAD) <= cy <= (seg.y_end_pct + _COVER_Y_PAD)


def find_undercovered_items(
    detected: Iterable[DetectedQuestion],
    page_lines: "dict[int, list[tuple[float, float, float, float]]] | None",
) -> set[tuple[bool, str]]:
    """Flag crops with a tall band of their own body text left uncovered below.

    For each crop segment we look at the strip directly beneath it in its own
    column, down to wherever the next crop in that column begins (or the body
    bottom). Any text lines in that strip that no crop covers are the crop's lost
    tail. When that uncovered run starts close under the crop (so it isn't a
    separately *missed* item sitting lower down) and is taller than
    :data:`_UNCOVERED_MIN_BAND_PCT`, the crop was almost certainly cut off and we
    return its ``(is_solution, q_num)``.

    ``page_lines`` being None/empty (e.g. a scanned PDF whose lines we didn't
    extract) disables the check entirely.
    """

    if not page_lines:
        return set()

    items = list(detected)

    # All segments per page, tagged with the owning item key.
    segs_by_page: dict[int, list[tuple[tuple[bool, str], QuestionSegment]]] = {}
    for q in items:
        key = (bool(q.is_solution), q.q_num)
        for seg in q.segments:
            segs_by_page.setdefault(seg.page, []).append((key, seg))

    flagged: set[tuple[bool, str]] = set()

    for page, segs in segs_by_page.items():
        lines = page_lines.get(page)
        if not lines:
            continue

        for owner_key, owner in segs:
            cx_lo, cx_hi = owner.x_start_pct, owner.x_end_pct

            # Where the owner's column ends below it: the nearest crop in the
            # same column that starts beneath the owner, else the body bottom.
            region_bottom = _BODY_BOTTOM_PCT
            for k, s in segs:
                if k == owner_key:
                    continue
                scx = (s.x_start_pct + s.x_end_pct) / 2.0
                if cx_lo <= scx <= cx_hi and s.y_start_pct >= owner.y_end_pct:
                    region_bottom = min(region_bottom, s.y_start_pct)

            # Uncovered lines in the strip below the owner, within its column.
            tail: list[tuple[float, float]] = []
            for line in lines:
                top, bottom = line[0], line[1]
                cx = _line_center_x(line)
                cy = (top + bottom) / 2.0
                if not (cx_lo <= cx <= cx_hi):
                    continue
                if cy <= owner.y_end_pct or cy < _BODY_TOP_PCT:
                    continue
                if cy > region_bottom:
                    continue
                if any(_seg_covers_line(s, line) for _, s in segs):
                    continue
                tail.append((top, bottom))

            if not tail:
                continue

            tail.sort()
            # The lost tail must begin close under the crop; text starting far
            # below is a separately missed item (a numbering gap), not this
            # crop's cut-off.
            if (tail[0][0] - owner.y_end_pct) > _UNCOVERED_MAX_GAP_PCT:
                continue

            band = sum(b - t for t, b in tail)
            if band >= _UNCOVERED_MIN_BAND_PCT:
                flagged.add(owner_key)

    return flagged


def find_uncovered_head_items(
    detected: Iterable[DetectedQuestion],
    page_lines: "dict[int, list[tuple[float, float, float, float]]] | None",
) -> set[tuple[bool, str]]:
    """Flag crops with a band of their own body text left uncovered ABOVE them.

    This is the mirror of :func:`find_undercovered_items`. The below-only check
    misses the cross-page-spill case: when a question continues onto the next
    page, its page-2 segment is sometimes detected starting *too low*, stranding
    the question's opening line(s) at the very top of the column with no box over
    them (e.g. Q19's "2. The right to health also includes access to essential
    medicines." or Q18's spilled "(D) Neither 1 nor 2" sitting above the Q20
    crop). Because that lost text is *above* the crop, not below, nothing flagged
    it and no Fix button appeared.

    For each column we take the topmost crop in it and look at the strip from the
    body top down to that crop. Any uncovered text lines there that sit close
    above the crop are its lost head; a run taller than :data:`_HEAD_MIN_BAND_PCT`
    flags the crop. ``page_lines`` being None/empty disables the check.
    """

    if not page_lines:
        return set()

    items = list(detected)

    segs_by_page: dict[int, list[tuple[tuple[bool, str], QuestionSegment]]] = {}
    for q in items:
        key = (bool(q.is_solution), q.q_num)
        for seg in q.segments:
            segs_by_page.setdefault(seg.page, []).append((key, seg))

    flagged: set[tuple[bool, str]] = set()

    for page, segs in segs_by_page.items():
        lines = page_lines.get(page)
        if not lines:
            continue

        for owner_key, owner in segs:
            cx_lo, cx_hi = owner.x_start_pct, owner.x_end_pct
            cx_mid = (cx_lo + cx_hi) / 2.0

            # Only the topmost crop in this column can have a lost head: if any
            # crop in the same column starts above this one, the strip above
            # belongs to that crop (or the gap between them), not to a missed
            # head of this one.
            region_top = _BODY_TOP_PCT
            has_crop_above = False
            for k, s in segs:
                if k == owner_key:
                    continue
                scx = (s.x_start_pct + s.x_end_pct) / 2.0
                if not (cx_lo <= scx <= cx_hi):
                    continue
                if s.y_start_pct < owner.y_start_pct:
                    has_crop_above = True
                    break
            if has_crop_above:
                continue

            # Uncovered lines in the strip above the owner, within its column.
            head: list[tuple[float, float]] = []
            for line in lines:
                top, bottom = line[0], line[1]
                cx = _line_center_x(line)
                cy = (top + bottom) / 2.0
                if not (cx_lo <= cx <= cx_hi):
                    continue
                if cy >= owner.y_start_pct or cy > _BODY_BOTTOM_PCT:
                    continue
                if cy < region_top:
                    continue
                if any(_seg_covers_line(s, line) for _, s in segs):
                    continue
                head.append((top, bottom))

            if not head:
                continue

            head.sort()
            # The lost head must end close above the crop; text far above is a
            # separately missed item, not this crop's own clipped opening.
            if (owner.y_start_pct - head[-1][1]) > _UNCOVERED_MAX_GAP_PCT:
                continue

            band = sum(b - t for t, b in head)
            if band >= _HEAD_MIN_BAND_PCT:
                flagged.add(owner_key)

    return flagged


def find_orphan_content_pages(
    detected: Iterable[DetectedQuestion],
    page_lines: "dict[int, list[tuple[float, float, float, float]]] | None",
) -> list[int]:
    """Return pages carrying a real block of body text that NO crop covers.

    The per-item head/tail checks blame an *adjacent* crop for nearby uncovered
    text. But a whole question or solution can be missed outright with no crop
    near it at all — there's nothing to flag, so it stays silent. This scans
    every page for uncovered body text that isn't merely the head/tail of an
    existing crop (i.e. it's separated from any crop by more than
    :data:`_UNCOVERED_MAX_GAP_PCT`) and reports the page when that orphan text is
    taller than :data:`_ORPHAN_MIN_BAND_PCT`.

    ``page_lines`` being None/empty disables the check.
    """

    if not page_lines:
        return []

    items = list(detected)
    segs_by_page: dict[int, list[QuestionSegment]] = {}
    for q in items:
        for seg in q.segments:
            segs_by_page.setdefault(seg.page, []).append(seg)

    orphan_pages: list[int] = []

    for page, lines in page_lines.items():
        if not lines:
            continue
        segs = segs_by_page.get(page, [])

        orphan_band = 0.0
        for line in lines:
            top, bottom = line[0], line[1]
            cy = (top + bottom) / 2.0
            if cy < _BODY_TOP_PCT or cy > _BODY_BOTTOM_PCT:
                continue
            if any(_seg_covers_line(s, line) for s in segs):
                continue
            # Skip lines that sit close to a crop in the same column — those are
            # the head/tail of that crop and are handled by the per-item checks.
            cx = _line_center_x(line)
            near_crop = False
            for s in segs:
                if not (s.x_start_pct <= cx <= s.x_end_pct):
                    continue
                gap_below = top - s.y_end_pct
                gap_above = s.y_start_pct - bottom
                if -_COVER_Y_PAD <= gap_below <= _UNCOVERED_MAX_GAP_PCT:
                    near_crop = True
                    break
                if -_COVER_Y_PAD <= gap_above <= _UNCOVERED_MAX_GAP_PCT:
                    near_crop = True
                    break
            if near_crop:
                continue
            orphan_band += bottom - top

        if orphan_band >= _ORPHAN_MIN_BAND_PCT:
            orphan_pages.append(page)

    return sorted(orphan_pages)


def drop_phantom_numbers(
    detected: Iterable[DetectedQuestion],
) -> list[DetectedQuestion]:
    """Remove lone question numbers that sit far above their run.

    An inline number a detector mistook for a marker — an angle/constant inside
    an equation ("E3 = 5E0 cos(wt + 53)"), a year, a quantity — surfaces as a
    single item whose number is wildly out of sequence with the rest of its side
    (e.g. a paper numbered 1..6 plus a stray "53"). We drop such an item only
    when ALL of these hold, so a genuine high-numbered question is never lost:

      * its number is the maximum on its side, and
      * a solid run of distinct lower numbers sits beneath it
        (``_PHANTOM_MIN_RUN_BELOW``), and
      * it is separated from the next-highest number by a big gap
        (``_PHANTOM_GAP``).

    Questions and solutions are judged independently (each has its own run). The
    peel repeats so two stacked spikes ("53" then "108") are both removed.
    Items without a parseable number are always kept.
    """

    items = list(detected)
    kept: list[DetectedQuestion] = []
    dropped: list[str] = []

    for is_solution in (False, True):
        side = [q for q in items if bool(q.is_solution) == is_solution]
        nums = sorted({n for q in side if (n := _q_number(q.q_num)) is not None})

        # Numbers that look phantom: peel isolated top outliers off the run.
        phantom: set[int] = set()
        run = list(nums)
        while len(run) >= _PHANTOM_MIN_RUN_BELOW + 1:
            top = run[-1]
            second = run[-2]
            if (top - second) >= _PHANTOM_GAP:
                phantom.add(top)
                run = run[:-1]
            else:
                break

        for q in side:
            num = _q_number(q.q_num)
            if num is not None and num in phantom:
                dropped.append(q.q_num)
                continue
            kept.append(q)

    if dropped:
        logger.info("dropped_phantom_numbers items=%s", ",".join(dropped))

    return kept


def _cutoff_reason(
    item: DetectedQuestion | AnalyzedItem, median_extent: float
) -> str | None:
    """Return a human reason if this crop looks cut off / half, else None."""

    extent = _extent_pct(item)

    # Stops at the page bottom and is short -> almost certainly continues.
    if _ends_at_page_bottom(item) and extent <= _PAGE_BOTTOM_MAX_EXTENT_PCT:
        return "Crop stops at the page edge — the rest may be on the next page. Re-select the full question."

    # Much shorter than its neighbours -> likely lost its options/lower half.
    if (
        median_extent > 0
        and extent < _SHORT_VS_MEDIAN_FRAC * median_extent
        and extent < 50.0
    ):
        return "Looks shorter than the other items — it may be only half the question. Re-select the full region."

    # Much taller than its neighbours AND large in absolute terms -> likely two
    # items merged into one box (or a crop that overran into the next question).
    if (
        median_extent > 0
        and extent > _TALL_VS_MEDIAN_FRAC * median_extent
        and extent >= _TALL_MIN_EXTENT_PCT
    ):
        return "Looks taller than the other items — it may have merged with the next question. Re-select just this one."

    # Absolute floor for the lonely-item case.
    if extent < _TINY_EXTENT_PCT:
        return "Crop is very small — it may be missing the question body. Re-select the full region."

    return None


def _missing_options_reason(item: DetectedQuestion | AnalyzedItem) -> str | None:
    """Return a reason if an MCQ crop saw some options but not all four.

    Standard MCQs have options (A)-(D). When detection captured at least two
    option labels but fewer than four — e.g. only the left column "(A)/(C)" of a
    2-up grid survived — the right-hand options were probably clipped. We only
    flag the *partial* case (2 or 3 of 4): zero/one label means the body wasn't a
    clean option list (no reliable signal), and all four means it's complete.

    ``option_labels`` is only populated by the text/OCR tiers; AI/manual items
    leave it empty and are never flagged here.
    """

    labels = getattr(item, "option_labels", "") or ""
    seen = {c for c in labels if c in "ABCD"}
    if 2 <= len(seen) < 4:
        missing = [c for c in "ABCD" if c not in seen]
        pretty = ", ".join(f"({c})" for c in missing)
        return (
            f"Options {pretty} look missing — the crop may have lost its "
            "right-hand options. Re-select the full question."
        )
    return None


def build_analyzed_items(
    detected: Iterable[DetectedQuestion],
    page_lines: "dict[int, list[tuple[float, float, float, float]]] | None" = None,
) -> list[AnalyzedItem]:
    """Wrap raw detections as review items, flagging the suspicious ones.

    Flags applied here (per-item):
      * ``cutoff``     — crop looks like only part of the question (half /
        continues onto the next page / lost its options / has a tall band of its
        own body text left uncovered below it).
      * ``overlap``    — crop physically overlaps another item on the page.
      * ``duplicate``  — same (is_solution, number) as an earlier item.

    ``page_lines`` maps a page number to the text lines on it
    ``(y_top, y_bottom, x_left, x_right)`` in page-percent units. When supplied
    it powers the content-coverage check (the strongest cut-off signal); when
    omitted only the shape-based checks run.
    """

    detected = list(detected)
    median_q = _median_extent(detected, is_solution=False)
    median_s = _median_extent(detected, is_solution=True)
    overlapping = find_overlapping_q_nums(detected)
    undercovered = find_undercovered_items(detected, page_lines)
    uncovered_head = find_uncovered_head_items(detected, page_lines)

    items: list[AnalyzedItem] = []
    seen: set[tuple[bool, int | None]] = set()

    for q in detected:
        num = _q_number(q.q_num)
        key = (bool(q.is_solution), num)
        flagged = False
        reason: str | None = None

        if key in seen and num is not None:
            flagged = True
            reason = "Looks like a duplicate of another item with the same number."
        else:
            median_extent = median_s if q.is_solution else median_q
            cut = _cutoff_reason(q, median_extent)
            if cut is not None:
                flagged = True
                reason = cut
            elif (bool(q.is_solution), q.q_num) in undercovered:
                flagged = True
                reason = (
                    "Text below this crop isn't covered by any box — the crop "
                    "may have stopped early. Re-select the full region."
                )
            elif (bool(q.is_solution), q.q_num) in uncovered_head:
                flagged = True
                reason = (
                    "Text above this crop isn't covered by any box — the crop "
                    "may have started too low (its opening line was cut off). "
                    "Re-select the full region."
                )
            elif (bool(q.is_solution), q.q_num) in overlapping:
                flagged = True
                reason = (
                    "Overlaps another item on the page — the crops may share "
                    "content. Re-select so each box covers only its own item."
                )
            else:
                # MCQ-aware check: some but not all four options captured.
                opt = _missing_options_reason(q)
                if opt is not None:
                    flagged = True
                    reason = opt

        seen.add(key)
        items.append(
            AnalyzedItem(
                q_num=q.q_num,
                is_solution=q.is_solution,
                segments=list(q.segments),
                source="auto",
                flagged=flagged,
                flag_reason=reason,
            )
        )

    return items


def build_review_notes(
    detected: list[DetectedQuestion],
    method_used: str,
    expected_question_numbers: "set[int] | None" = None,
    page_lines: "dict[int, list[tuple[float, float, float, float]]] | None" = None,
) -> list[ReviewNote]:
    """Produce human-readable review notes (cut-off crops, duplicates, gaps).

    These drive the popup: any note means "have a look before downloading".

    ``expected_question_numbers`` (when supplied, parsed from the paper's own
    answer key) is the authoritative set of question numbers the paper contains.
    Any of those numbers missing from the detected questions is reported as a
    high-confidence gap — the key *proves* the question exists — which is
    stronger than the sequence-based guess below.

    ``page_lines`` (page -> text-line extents in page-percent) powers the
    content-coverage check that catches a normal-looking crop which stopped
    short, leaving its own body text uncovered below it.
    """

    notes: list[ReviewNote] = []

    median_q = _median_extent(detected, is_solution=False)
    median_s = _median_extent(detected, is_solution=True)
    overlapping = find_overlapping_q_nums(detected)
    undercovered = find_undercovered_items(detected, page_lines)
    uncovered_head = find_uncovered_head_items(detected, page_lines)

    # Split questions vs solutions; numbering continuity is judged per side.
    for is_solution in (False, True):
        group = [q for q in detected if bool(q.is_solution) == is_solution]
        median_extent = median_s if is_solution else median_q
        seen_nums: set[int] = set()
        nums: list[int] = []
        label_one = "Solution" if is_solution else "Question"

        for q in group:
            num = _q_number(q.q_num)

            cut = _cutoff_reason(q, median_extent)
            if cut is not None:
                notes.append(
                    ReviewNote(
                        kind="incomplete",
                        message=f"{label_one} {q.q_num}: {cut}",
                        q_num=q.q_num,
                        page=_primary_page(q),
                        is_solution=is_solution,
                    )
                )
            elif (is_solution, q.q_num) in undercovered:
                notes.append(
                    ReviewNote(
                        kind="incomplete",
                        message=(
                            f"{label_one} {q.q_num}: Text below this crop isn't "
                            "covered by any box — the crop may have stopped early. "
                            "Re-select the full region."
                        ),
                        q_num=q.q_num,
                        page=_primary_page(q),
                        is_solution=is_solution,
                    )
                )
            elif (is_solution, q.q_num) in uncovered_head:
                notes.append(
                    ReviewNote(
                        kind="incomplete",
                        message=(
                            f"{label_one} {q.q_num}: Text above this crop isn't "
                            "covered by any box — the crop may have started too "
                            "low (its opening line was cut off). Re-select the "
                            "full region."
                        ),
                        q_num=q.q_num,
                        page=_primary_page(q),
                        is_solution=is_solution,
                    )
                )
            elif (is_solution, q.q_num) in overlapping:
                notes.append(
                    ReviewNote(
                        kind="incomplete",
                        message=(
                            f"{label_one} {q.q_num}: Overlaps another item on the "
                            "page — the crops may share content. Re-select so each "
                            "box covers only its own item."
                        ),
                        q_num=q.q_num,
                        page=_primary_page(q),
                        is_solution=is_solution,
                    )
                )
            else:
                opt = _missing_options_reason(q)
                if opt is not None:
                    notes.append(
                        ReviewNote(
                            kind="incomplete",
                            message=f"{label_one} {q.q_num}: {opt}",
                            q_num=q.q_num,
                            page=_primary_page(q),
                            is_solution=is_solution,
                        )
                    )

            if num is None:
                continue
            if num in seen_nums:
                notes.append(
                    ReviewNote(
                        kind="duplicate",
                        message=f"{label_one} {q.q_num} appears more than once.",
                        q_num=q.q_num,
                        page=_primary_page(q),
                        is_solution=is_solution,
                    )
                )
            seen_nums.add(num)
            nums.append(num)

        # Numbering gaps: a missing number in an otherwise sequential run is a
        # strong sign a question was skipped and should be cropped by hand.
        if len(nums) >= 3:
            lo, hi = min(nums), max(nums)
            missing = [n for n in range(lo, hi + 1) if n not in seen_nums]
            if missing and len(missing) <= max(5, (hi - lo) // 2):
                label = "solutions" if is_solution else "questions"
                pretty = ", ".join(str(m) for m in missing[:8])
                more = "" if len(missing) <= 8 else f" (+{len(missing) - 8} more)"
                notes.append(
                    ReviewNote(
                        kind="gap",
                        message=(
                            f"Possible missing {label}: {pretty}{more}. "
                            "Add them manually if they exist."
                        ),
                        is_solution=is_solution,
                    )
                )

    # Answer-key cross-check (questions only): the key lists every question
    # number, so anything it contains that we didn't detect is a confident miss.
    if expected_question_numbers:
        detected_q_nums = {
            _q_number(q.q_num)
            for q in detected
            if not q.is_solution and _q_number(q.q_num) is not None
        }
        missing_vs_key = sorted(expected_question_numbers - detected_q_nums)  # type: ignore[operator]
        if missing_vs_key:
            pretty = ", ".join(str(m) for m in missing_vs_key[:10])
            more = "" if len(missing_vs_key) <= 10 else f" (+{len(missing_vs_key) - 10} more)"
            notes.append(
                ReviewNote(
                    kind="gap",
                    message=(
                        f"The answer key lists {len(expected_question_numbers)} questions "
                        f"but these weren't detected: {pretty}{more}. Add them manually."
                    ),
                )
            )

    # Whole-item miss: a page carrying a real block of body text that no crop
    # covers (and that isn't the head/tail of an existing crop) means a question
    # or solution was missed outright. There's no item to flag, so this is the
    # only signal — surface it as a page-level note so the user knows to look.
    orphan_pages = find_orphan_content_pages(detected, page_lines)
    if orphan_pages:
        pretty = ", ".join(str(p) for p in orphan_pages[:8])
        more = "" if len(orphan_pages) <= 8 else f" (+{len(orphan_pages) - 8} more)"
        notes.append(
            ReviewNote(
                kind="incomplete",
                message=(
                    f"Some text on page {pretty}{more} isn't covered by any crop "
                    "— a question or solution may have been missed. Check the "
                    "page and add it by hand if so."
                ),
                page=orphan_pages[0],
            )
        )

    if method_used != "ai" and not detected:
        notes.append(
            ReviewNote(
                kind="low_confidence",
                message="Nothing was detected automatically. Crop the items you need by hand.",
            )
        )

    return notes
