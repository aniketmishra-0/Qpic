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

from ..models.schemas import AnalyzedItem, DetectedQuestion, ReviewNote

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


def build_analyzed_items(detected: Iterable[DetectedQuestion]) -> list[AnalyzedItem]:
    """Wrap raw detections as review items, flagging the suspicious ones.

    Flags applied here (per-item):
      * ``cutoff``     — crop looks like only part of the question (half /
        continues onto the next page / lost its options).
      * ``duplicate``  — same (is_solution, number) as an earlier item.
    """

    detected = list(detected)
    median_q = _median_extent(detected, is_solution=False)
    median_s = _median_extent(detected, is_solution=True)

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
) -> list[ReviewNote]:
    """Produce human-readable review notes (cut-off crops, duplicates, gaps).

    These drive the popup: any note means "have a look before downloading".

    ``expected_question_numbers`` (when supplied, parsed from the paper's own
    answer key) is the authoritative set of question numbers the paper contains.
    Any of those numbers missing from the detected questions is reported as a
    high-confidence gap — the key *proves* the question exists — which is
    stronger than the sequence-based guess below.
    """

    notes: list[ReviewNote] = []

    median_q = _median_extent(detected, is_solution=False)
    median_s = _median_extent(detected, is_solution=True)

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

    if method_used != "ai" and not detected:
        notes.append(
            ReviewNote(
                kind="low_confidence",
                message="Nothing was detected automatically. Crop the items you need by hand.",
            )
        )

    return notes
