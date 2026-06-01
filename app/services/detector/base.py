"""Shared detector primitives.

This module centralizes question-start matching and the question region
building logic used by multiple detection tiers.

The core idea: a question's crop region is bounded by its *actual content
lines* (stem + options), not by the next question marker or the page bottom.
This yields tight crops with no trailing blank space, and stitches across
pages only when real content actually flows onto the next page.
"""

from __future__ import annotations

import re
from typing import NamedTuple
from typing import Optional

from ...models.schemas import DetectedQuestion, QuestionSegment
from .answer_key import extract_answer_key, expected_question_numbers


# Patterns to match question start markers.
#
# We distinguish two strengths of marker:
#   * STRONG  — an explicit "Q" prefix ("Q1", "Q.1", "Q1.", "Q 1)"). These are
#     unambiguous question markers.
#   * WEAK    — a bare leading number ("1.", "2)", "3. ("). These are real
#     questions in papers that have no "Q" prefix, but in papers that *do* use
#     "Q" markers the same bare numbers are sub-statements ("Consider the
#     following statements: 1. ... 2. ...") and must NOT start a new question.
#
# When a document contains any STRONG markers we keep only those and treat weak
# numbers as ordinary content. Otherwise we fall back to weak markers so simple
# "1. / 2." papers still work exactly as before.

STRONG_QUESTION_PATTERN: re.Pattern[str] = re.compile(
    # "Q1", "Q.1", "Q 1)", "Ques 1", "Question 1", and also the "Q.No 1" /
    # "Q No. 1" / "Question No 1" forms common in Indian exam papers. The
    # optional ``(?:no[\.\s:]*)`` consumes a "No"/"No."/"No:" label between the
    # Q-word and the number so those still register as STRONG markers.
    r"^\s*[Qq](?:ues(?:tion)?)?[\.\s]?\s*(?:no[\.\s:]*)?\s*[-:]?\s*(\d{1,3})\b",
    re.IGNORECASE,
)

WEAK_QUESTION_PATTERNS: list[re.Pattern[str]] = [
    # "1. text", "1) text", and also a bare "1." / "1)" that sits alone on its
    # own line/block (common when a paper renders the question number in a
    # separate frame from the stem). The trailing `(?:\s|$)` accepts both the
    # "marker + space + text" case and the "marker is the whole line" case,
    # while still rejecting decimals ("2.5") and years ("2024") because those
    # have a digit — not whitespace or end-of-line — right after the separator.
    re.compile(r"^\s*(\d{1,3})\s*[\.\)](?:\s|$)"),  # "1." / "1)" (alone or followed by text)
    re.compile(r"^\s*(\d{1,3})\s*\.\s*\("),  # "1. ("
]

# Backward-compatible alias used by older call sites/tests.
QUESTION_PATTERNS: list[re.Pattern[str]] = [STRONG_QUESTION_PATTERN, *WEAK_QUESTION_PATTERNS]


# Headers that mark the start of a solutions / answer-explanation section.
# Everything numbered after one of these is treated as a solution (S001, ...)
# rather than a question (Q001, ...).
SOLUTION_HEADERS: frozenset[str] = frozenset(
    {
        "solution",
        "solutions",
        "answer",
        "answers",
        "answerkey",
        "answerskey",
        "answersolution",
        "answersolutions",
        "answerssolutions",
        "answerandsolution",
        "answerandsolutions",
        "answersandsolutions",
        "hintssolution",
        "hintssolutions",
        "hintandsolution",
        "hintsandsolutions",
        "detailedsolution",
        "detailedsolutions",
        "explanation",
        "explanations",
        "solutionsexplanations",
        "answerexplanation",
        "answerexplanations",
        "key",
        "keysolution",
        "keysolutions",
        "solutionkey",
        "hints",
        "hint",
        "hintsolution",
        "hintandsolution",
        "answerwithsolution",
        "answerswithsolutions",
        "answerwithexplanation",
        "answerswithexplanations",
        "solutionandanswer",
        "solutionsandanswers",
    }
)


def match_solution_header(text: str) -> bool:
    """Return True if a line is a standalone solutions/answer-key section header.

    The line is normalized to lowercase alphanumerics only so decorations like
    ":", "&", "-" or surrounding whitespace don't matter. We only match short
    standalone headers (not the word "solution" buried inside a sentence).
    """

    candidate = (text or "").strip()
    if not candidate:
        return False

    # Ignore long lines: real section headers are short.
    if len(candidate) > 40:
        return False

    normalized = re.sub(r"[^a-z0-9]", "", candidate.lower())
    if not normalized:
        return False

    return normalized in SOLUTION_HEADERS


# Section dividers separate groups of questions in a paper ("PART-II
# (CHEMISTRY)", "SECTION-1", "This section contains FOUR (04) questions").
# These lines, and the marking-scheme instructions that follow them, sit
# *between* two questions but belong to neither. Left untreated, the preceding
# question's crop grows past its own options to swallow the whole divider block
# (and sometimes the first line of the next question). Recognising the divider
# lets the crop stop at the question's real content.
SECTION_HEADER_PATTERNS: list[re.Pattern[str]] = [
    # "PART-II", "PART II", "PART - A", "PART 1", "PART-II (CHEMISTRY)".
    re.compile(r"^\s*PART\s*[-–—:.]?\s*([IVXLCDM]+|[A-Z]|\d{1,2})\b", re.IGNORECASE),
    # "SECTION-1", "SECTION 2", "SECTION-A", "SECTION-I (Maximum Marks: 12)".
    re.compile(r"^\s*SECTION\s*[-–—:.]?\s*([IVXLCDM]+|[A-Z]|\d{1,2})\b", re.IGNORECASE),
]

# A standalone instruction line that opens a section's rubric. Used as a backup
# divider trigger when the "PART/SECTION" title is absent or was already removed
# as a repeating running header.
_SECTION_INSTRUCTION_PATTERN: re.Pattern[str] = re.compile(
    r"^this\s+section\s+contains\b", re.IGNORECASE
)


def match_section_header(text: str) -> bool:
    """Return True if a line begins a new section/part divider block.

    Matches short ``PART …`` / ``SECTION …`` titles and the "This section
    contains …" rubric opener. The length guard keeps a question stem that
    merely mentions the words "part" or "section" from being mistaken for a
    divider — real dividers are short standalone headers.
    """

    candidate = (text or "").strip()
    if not candidate or len(candidate) > 60:
        return False

    for pattern in SECTION_HEADER_PATTERNS:
        if pattern.match(candidate):
            return True

    return bool(_SECTION_INSTRUCTION_PATTERN.match(candidate))


class QuestionStart(NamedTuple):
    page_num: int  # 1-indexed
    y_top: float
    q_num: str
    is_solution: bool = False
    x_left: float = 0.0
    x_right: float = 0.0
    is_strong: bool = True


class ContentLine(NamedTuple):
    """A single line of content with its horizontal + vertical extent on a page."""

    page_num: int  # 1-indexed
    y_top: float
    y_bottom: float
    x_left: float = 0.0
    x_right: float = 0.0
    text: str = ""


class FigureRegion(NamedTuple):
    """A non-text graphic region on a page (diagram, chart, embedded image).

    Exam questions frequently include a figure between the stem and the options
    (a circuit, a cone, a velocity-time graph). The text/OCR detectors only see
    *words*, so without explicitly tracking figures a question's crop bounds —
    built purely from text extents — clip a figure that is wider than the text
    or drop one that sits below the last text line. A ``FigureRegion`` carries
    the figure's full pixel extent so the question it belongs to can grow its
    crop to contain the whole diagram.
    """

    page_num: int  # 1-indexed
    y_top: float
    y_bottom: float
    x_left: float
    x_right: float


# Valid question-numbering styles the caller can force. "auto" keeps the
# original behaviour (prefer "Q" markers, fall back to bare numbers). "q" only
# accepts explicit Q-prefixed markers ("Q1", "Q.1", "Question 1"). "numbered"
# only accepts bare leading numbers ("1.", "2)"). Forcing a style is how a user
# stops sub-statements / option labels / equation numbers from being mistaken
# for questions in a paper whose real markers are all one style.
MARKER_STYLE_AUTO = "auto"
MARKER_STYLE_Q = "q"
MARKER_STYLE_NUMBERED = "numbered"
VALID_MARKER_STYLES = frozenset({MARKER_STYLE_AUTO, MARKER_STYLE_Q, MARKER_STYLE_NUMBERED})


def match_question_start(text: str) -> Optional[str]:
    """Return the matched question number (e.g. "1", "12") or None.

    Matches both strong ("Q1") and weak ("1.") markers. Use
    :func:`match_question_start_ex` when you need to know which strength matched.
    """

    result = match_question_start_ex(text)
    return result[0] if result is not None else None


def match_question_start_ex(
    text: str, style: str = MARKER_STYLE_AUTO
) -> Optional[tuple[str, bool]]:
    """Return ``(q_num, is_strong)`` for a question marker, or None.

    ``is_strong`` is True for explicit "Q"-prefixed markers and False for bare
    leading numbers.

    ``style`` restricts which marker kinds are accepted:
      * ``"auto"``     — both (default).
      * ``"q"``        — only "Q"-prefixed markers.
      * ``"numbered"`` — only bare leading numbers.
    """

    candidate = (text or "").strip()
    if not candidate:
        return None

    if style != MARKER_STYLE_NUMBERED:
        strong = STRONG_QUESTION_PATTERN.match(candidate)
        if strong:
            q_num = strong.group(1).lstrip("0") or "0"
            return (q_num, True)
        if style == MARKER_STYLE_Q:
            return None

    if style != MARKER_STYLE_Q:
        for pattern in WEAK_QUESTION_PATTERNS:
            match = pattern.match(candidate)
            if match:
                q_num = match.group(1).lstrip("0") or "0"
                return (q_num, False)
    return None


# Common Tesseract digit confusions. A misread question number ("20." -> "2O.",
# "2C.", "Z0.") slips past the strict marker patterns, so the number is never
# recognised and the question is dropped — leaving a hole in the numbering
# (the "Q20 aaya hi nahi" case). The gap-recovery pass uses these to re-read a
# line's leading token as a number when we already know, from the surrounding
# sequence, which number *should* be there.
_OCR_DIGIT_FIXUPS: dict[str, str] = {
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "I": "1", "l": "1", "i": "1", "|": "1", "!": "1",
    "Z": "2", "z": "2",
    "E": "3",
    "A": "4",
    "S": "5", "s": "5",
    "G": "6", "b": "6",
    "T": "7",
    "B": "8",
    "g": "9", "q": "9",
}


def _ocr_token_to_int(token: str) -> Optional[int]:
    """Best-effort read of a marker token as an integer, fixing OCR confusions.

    ``token`` is the leading chunk of a line up to the separator (e.g. "2O",
    "2C.", "Z0)"). Each character is mapped through :data:`_OCR_DIGIT_FIXUPS`
    when it isn't already a digit; if the whole token then reads as 1-3 digits we
    return its value. Returns None when the token can't be coerced to a number,
    so a genuine word ("This") is never mistaken for a marker.
    """

    cleaned = (token or "").strip().strip(".)-:")
    if not cleaned or len(cleaned) > 3:
        return None
    digits = ""
    for ch in cleaned:
        if ch.isdigit():
            digits += ch
        elif ch in _OCR_DIGIT_FIXUPS:
            digits += _OCR_DIGIT_FIXUPS[ch]
        else:
            return None
    if not digits or len(digits) > 3:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


# Option labels inside a question body ("(A)", "B)", "(C)", "D."). Used to tell
# whether a detected question captured all four MCQ choices or only some of them
# (e.g. only the left column "(A)/(C)" of a 2-up option grid), which the review
# step surfaces as a "may have lost its options" flag.
_OPTION_LABEL_PATTERN: re.Pattern[str] = re.compile(
    r"(?<![A-Za-z0-9])\(?\s*([A-Da-d])\s*[\.\)]"
)


def _option_letters_in(text: str) -> set[str]:
    """Return the uppercase MCQ option letters present in a line.

    Matches "(A)", "A)", "A.", "(b)" etc., but requires the option-label
    punctuation so a stray capital inside a word ("Acceleration") isn't counted.
    Only A-D are considered (standard four-option MCQ).
    """

    out: set[str] = set()
    for m in _OPTION_LABEL_PATTERN.finditer(text or ""):
        out.add(m.group(1).upper())
    return out


def line_matches_expected_number(text: str, expected: int) -> bool:
    """True if a line's leading token plausibly *is* the expected marker number.

    Used only by the gap-recovery pass, which already knows the missing number
    from the surrounding sequence. The leading token (before the first space) is
    read with OCR digit-confusion fixups; we accept the line when that token
    resolves to ``expected``. This is deliberately narrow — it must agree with a
    number we independently expect — so it can't invent spurious markers.
    """

    candidate = (text or "").strip()
    if not candidate or expected <= 0:
        return False
    # The marker token is the first whitespace-delimited chunk, but a glued
    # "20.The" needs the leading separator-run too; take chars up to the first
    # space OR the first separator followed by a letter/space.
    head = candidate.split(None, 1)[0]
    value = _ocr_token_to_int(head)
    if value == expected:
        return True
    # Also try the chunk up to an explicit separator inside the head ("20.2.5").
    for sep in (".", ")"):
        if sep in head:
            value = _ocr_token_to_int(head.split(sep, 1)[0])
            if value == expected:
                return True
    return False


def _clamp_pct(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 100.0:
        return 100.0
    return float(value)


# Fraction of page width that a low-density valley must span to count as a
# gutter separating two columns.
_GUTTER_FRACTION = 0.03

# A bin counts as "inside a column" when the number of lines covering it exceeds
# this fraction of the densest bin. Real column bodies are covered by many lines
# (10-20+), while the center gutter is only grazed by the odd centered title or
# footer, so a relative threshold isolates the gutter even when it is never
# completely empty.
_COLUMN_DENSITY_FRACTION = 0.18

# Minimum width (as a fraction of page width) for a density core to count as a
# real text column. A strip narrower than this is not a column but a list-marker
# gutter — e.g. question numbers ("1." "2." "3.") rendered in their own narrow
# left-margin frame, separate from the stem. Such a strip forms a thin
# high-density core that would otherwise be split off as its own column,
# stranding every question's number in one column and its stem/options in
# another. That breaks marker→content association so badly that one question can
# swallow the whole page. Any core below this width is absorbed into its nearest
# neighbour so the numbers stay in the same column as their text.
_MIN_COLUMN_WIDTH_FRACTION = 0.12


def detect_columns(
    intervals: list[tuple[float, float]], page_width: float
) -> list[tuple[float, float]]:
    """Return left→right column x-ranges for a page.

    ``intervals`` are the horizontal ``(x_left, x_right)`` extents of every
    content line on the page. We build a coverage-density profile across the
    page width (how many lines overlap each vertical strip), treat strips below
    a relative density threshold as gutters, and split columns at the center of
    each gutter valley. A single dense block yields one column spanning the full
    page width so single-column documents behave exactly as before.

    Splitting on a density *valley* (rather than a fully empty band) is what
    lets two-column exam papers be separated even though a centered heading or
    footer occasionally reaches across the middle.
    """

    if not intervals or page_width <= 0:
        return [(0.0, page_width)]

    bins = 200
    bin_w = page_width / bins

    # Build the per-bin line-coverage density. The vectorized path uses a
    # difference array (mark +1 at each interval's first bin, -1 just past its
    # last, then cumulative-sum) so a page with hundreds of OCR lines costs one
    # numpy pass instead of an O(lines x bins) Python loop. The pure-Python loop
    # is kept as a fallback so column detection still works if numpy can't load
    # (matching the rest of the codebase's optional-numpy pattern).
    density: list[int]
    try:
        import numpy as np

        arr = np.asarray(intervals, dtype=float)
        lo = np.minimum(arr[:, 0], arr[:, 1])
        hi = np.maximum(arr[:, 0], arr[:, 1])
        # Match the scalar loop's asymmetric clamping exactly: b0 is only
        # lower-bounded at 0 and b1 only upper-bounded at bins-1, so an interval
        # lying entirely off the page (or negative) yields b0 > b1 and is
        # dropped rather than piling onto an edge bin.
        b0 = np.maximum(0, (lo / bin_w).astype(np.int64))
        b1 = np.minimum(bins - 1, (hi / bin_w).astype(np.int64))
        valid = b0 <= b1
        diff = np.zeros(bins + 1, dtype=np.int64)
        np.add.at(diff, b0[valid], 1)
        np.add.at(diff, b1[valid] + 1, -1)
        density = np.cumsum(diff[:bins]).tolist()
    except Exception:
        density = [0] * bins
        for x0, x1 in intervals:
            if x1 < x0:
                x0, x1 = x1, x0
            b0_i = max(0, int(x0 / bin_w))
            b1_i = min(bins - 1, int(x1 / bin_w))
            for b in range(b0_i, b1_i + 1):
                density[b] += 1

    peak = max(density)
    if peak <= 0:
        return [(0.0, page_width)]

    threshold = max(1.0, peak * _COLUMN_DENSITY_FRACTION)
    in_column = [d > threshold for d in density]

    # Contiguous runs of column bins are candidate column cores.
    runs: list[list[int]] = []
    i = 0
    while i < bins:
        if in_column[i]:
            j = i
            while j < bins and in_column[j]:
                j += 1
            runs.append([i, j - 1])
            i = j
        else:
            i += 1

    if len(runs) <= 1:
        return [(0.0, page_width)]

    # Merge cores separated only by a narrow valley (not a real gutter).
    min_gap_bins = max(1, int(_GUTTER_FRACTION * bins))
    merged: list[list[int]] = [runs[0]]
    for s, e in runs[1:]:
        prev = merged[-1]
        if s - prev[1] - 1 < min_gap_bins:
            prev[1] = e
        else:
            merged.append([s, e])

    if len(merged) <= 1:
        return [(0.0, page_width)]

    # Absorb cores too narrow to be a real text column (a list-marker gutter:
    # standalone question numbers in their own left-margin frame). Merging such
    # a thin core into its neighbour keeps each question's number in the same
    # column as its stem, instead of splitting the page into a "numbers" column
    # and a "text" column. Iterate until stable so several thin strips collapse.
    min_core_bins = max(1, int(_MIN_COLUMN_WIDTH_FRACTION * bins))

    def _absorb_narrow(cores: list[list[int]]) -> list[list[int]]:
        if len(cores) <= 1:
            return cores
        widths_bins = [e - s + 1 for s, e in cores]
        # Find the narrowest sub-threshold core and merge it into the adjacent
        # core it sits closest to (smallest gap), then repeat.
        narrow_idx = min(
            range(len(cores)),
            key=lambda i: widths_bins[i],
        )
        if widths_bins[narrow_idx] >= min_core_bins:
            return cores
        if narrow_idx == 0:
            target = 1
        elif narrow_idx == len(cores) - 1:
            target = len(cores) - 2
        else:
            gap_left = cores[narrow_idx][0] - cores[narrow_idx - 1][1]
            gap_right = cores[narrow_idx + 1][0] - cores[narrow_idx][1]
            target = narrow_idx - 1 if gap_left <= gap_right else narrow_idx + 1
        lo = min(cores[narrow_idx][0], cores[target][0])
        hi = max(cores[narrow_idx][1], cores[target][1])
        new_cores = [
            c for i, c in enumerate(cores) if i not in (narrow_idx, target)
        ]
        new_cores.append([lo, hi])
        new_cores.sort()
        return new_cores

    prev_len = -1
    while len(merged) != prev_len:
        prev_len = len(merged)
        merged = _absorb_narrow(merged)

    if len(merged) <= 1:
        return [(0.0, page_width)]

    # Tile the page into columns, splitting at the center of each gutter so each
    # column captures its full visual extent (including any centered heading).
    columns: list[tuple[float, float]] = []
    n = len(merged)
    for idx, (s, e) in enumerate(merged):
        if idx == 0:
            left_x = 0.0
        else:
            prev_e = merged[idx - 1][1]
            left_x = ((prev_e + 1 + s) / 2.0) * bin_w
        if idx == n - 1:
            right_x = page_width
        else:
            next_s = merged[idx + 1][0]
            right_x = ((e + 1 + next_s) / 2.0) * bin_w
        columns.append((left_x, right_x))

    return columns


# A real multi-column page numbers each column independently, so a genuine
# column split has question markers landing in at least two of the candidate
# columns. The lookalike that this guards against is the single-column MCQ paper
# whose options are laid out in a 2-up grid ("(A) … (B) …" over "(C) … (D) …"):
# that grid opens a tall whitespace gutter down the page middle, which
# :func:`detect_columns` reads as two page columns. Every question *marker*,
# though, still sits in the left column — the right "column" holds only option
# continuations — so confining each crop to its marker's column slices the (B)
# and (D) options off the right half (and balloons the last question to swallow
# the orphaned right strip). Requiring markers in ≥2 columns collapses such a
# page back to a single full-width column, while leaving true two-column papers
# (markers on both sides) untouched.
# A line is an MCQ option label when it *starts* with an option marker
# ("(A)", "B)", "C."). Used to tell an option-grid's right column (only option
# labels) from a real second text column (independent prose).
_OPTION_START_RE: re.Pattern[str] = re.compile(r"^\s*\(?\s*[A-Da-d]\s*[\.\)]")

# A non-marker column must carry at least this many content lines, of which at
# least this fraction are non-option prose, before its split is trusted as a
# real second text column rather than the option-grid false gutter.
_MIN_SECOND_COLUMN_LINES = 4
_SECOND_COLUMN_PROSE_FRACTION = 0.5


def _line_starts_with_option(text: str) -> bool:
    """True if a line begins with an MCQ option label ("(B)", "C.", "d)")."""

    return bool(_OPTION_START_RE.match((text or "").strip()))


# An answer-key grid cell: a number then its correct option, the option being a
# single bracketed/plain digit or letter — "1. (1)", "10. (3)", "1-B", "2 (a)".
# Such cells are packed many-to-a-row in a compact key table, e.g.
# "1. (1)  2. (2)  3. (3) …". They are NOT croppable solutions — they are the
# answer key — yet their leading "1." reads as a question/solution marker and
# the row repeats the whole 1..N numbering, duplicating every real solution. A
# line that *is* a single such cell (short, nothing after the option) is matched
# here so its marker can be dropped.
_ANSWER_KEY_CELL_RE: re.Pattern[str] = re.compile(
    r"^\s*(\d{1,3})\s*[\.\)\-:]?\s*\(?\s*([A-Ea-e1-5])\s*\)?\s*$"
)


def _is_answer_key_cell(text: str) -> bool:
    """True if a line is a lone answer-key cell ("1. (1)", "10. (3)", "2-B")."""

    return bool(_ANSWER_KEY_CELL_RE.match((text or "").strip()))


def _other_columns_carry_independent_text(
    columns: list[tuple[float, float]],
    lines: "Optional[list[ContentLine]]",
    page_num: int,
    marker_col: int,
) -> bool:
    """True when columns other than the marker column hold their own body text.

    This is the discriminator between the two layouts that both put every
    question marker in a single column:

      * **Option grid** (single logical column) — the non-marker column holds
        only the spilled "(B)/(D)" option labels, each paired with a left
        "(A)/(C)". It is *not* an independent column, so the page must collapse
        to full width or the options get clipped.
      * **True two-column page** — the non-marker column carries its own prose
        (e.g. a solution's explanation continuing on the right), which must be
        cropped separately, not stitched under the left column.

    We treat the other column as independent only when it contains a meaningful
    number of lines and a substantial share of them are *not* option labels.
    With no line data we report False so the conservative option-grid collapse
    is preserved.
    """

    if not lines:
        return False

    other = [
        ln
        for ln in lines
        if ln.page_num == page_num
        and ln.x_right > ln.x_left
        and _column_index(columns, ln.x_left, ln.x_right) != marker_col
    ]
    if len(other) < _MIN_SECOND_COLUMN_LINES:
        return False

    non_option = sum(1 for ln in other if not _line_starts_with_option(ln.text))
    needed = max(
        _MIN_SECOND_COLUMN_LINES,
        int(len(other) * _SECOND_COLUMN_PROSE_FRACTION),
    )
    return non_option >= needed


def _validate_columns_with_markers(
    columns: list[tuple[float, float]],
    starts: list["QuestionStart"],
    page_num: int,
    page_width: float,
    lines: "Optional[list[ContentLine]]" = None,
) -> list[tuple[float, float]]:
    """Keep a multi-column split only when markers don't betray an option grid.

    The option-grid false split has a clear signature: a page that *starts
    several questions* has every one of those markers in a single column (the
    left one), because the right "column" is only option continuations. A
    genuine multi-column page starts questions in more than one column.

    So we collapse to one full-width column only when this page has **two or
    more** question markers and they all fall in the same column. Pages with no
    markers (a pure cross-page continuation, whose two real columns must be kept
    to stitch correctly) and pages that start a single question (ambiguous —
    left as detected) are returned unchanged.

    Markers clustering in one column is *also* what a real two-column solutions
    page looks like when its "Q1 Text Solution", "Q2 Text Solution" openings all
    fall in the left column and the right column carries their explanation prose.
    To avoid collapsing that genuine layout, when ``lines`` are supplied we keep
    the split if the non-marker column holds its own independent text (see
    :func:`_other_columns_carry_independent_text`); only the option-grid
    signature (the other column is just spilled option labels) collapses.
    """

    if len(columns) <= 1:
        return columns

    cols_with_marker: set[int] = set()
    marker_count = 0
    marker_col = 0
    for s in starts:
        if s.page_num != page_num:
            continue
        marker_count += 1
        ci = _column_index(columns, s.x_left, s.x_right)
        marker_col = ci
        cols_with_marker.add(ci)
        if len(cols_with_marker) >= 2:
            # Questions start in two different columns → real multi-column page.
            return columns

    # Several questions all starting in one column → either an option-grid false
    # split (collapse) or a real two-column page whose markers happen to cluster
    # (keep). The non-marker column's content decides which.
    if marker_count >= 2 and len(cols_with_marker) == 1:
        if _other_columns_carry_independent_text(
            columns, lines, page_num, marker_col
        ):
            return columns
        # Option-grid signature: collapse to one full-width column so each crop
        # spans the whole page and the right-hand options aren't clipped.
        return [(0.0, page_width or 1.0)]

    return columns


def _column_index(columns: list[tuple[float, float]], x_left: float, x_right: float) -> int:
    """Return the index of the column that best contains the given x-extent."""

    if not columns:
        return 0
    center = (x_left + x_right) / 2.0
    for idx, (c0, c1) in enumerate(columns):
        if c0 <= center <= c1:
            return idx
    # Center falls in a gutter — pick the nearest column.
    best_idx = 0
    best_dist = float("inf")
    for idx, (c0, c1) in enumerate(columns):
        dist = min(abs(center - c0), abs(center - c1))
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx


def _pos(page_num: int, col: int, y: float) -> tuple[int, int, float]:
    """A globally comparable document position: page → column → vertical offset."""

    return (page_num, col, y)


def _normalize_text(text: str) -> str:
    """Lowercase alphanumerics only — for comparing repeated furniture lines."""

    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def _normalize_loose(text: str) -> str:
    """Like :func:`_normalize_text` but also drops digits.

    A running header/footer is usually constant except for a per-page page
    number or date ("… | PW Website   12", "… | PW Website   13"). Dropping the
    digits lets those still collapse to one key so the footer is recognised as
    repeating instead of looking unique on every page.
    """

    return re.sub(r"[0-9]", "", _normalize_text(text))


# Standalone page-number / folio lines that live in a page margin: a bare
# number ("5"), an "n/m" folio ("5/5"), a "Page 5 of 10" caption or a dash-
# wrapped number ("- 5 -"). These defeat the repeat-based furniture test: a
# folio's only varying part is its per-page digit, so stripping digits
# (:func:`_normalize_loose`) collapses it to an empty key that is ignored, while
# its exact text ("5/5", "6/5") is unique on every page and never looks like a
# repeat. Left in, a folio pinned to ~97% of the page drags a column's crop
# bound to the page bottom (pulling the footer band into the crop and leaving a
# tall blank gap). We therefore match folios by *shape* and drop them whenever
# they sit in a margin band.
_PAGE_NUMBER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\d{1,4}$"),                                  # "5"
    re.compile(r"^\d{1,4}\s*/\s*\d{1,4}$"),                    # "5/5"
    re.compile(r"^page\s*\d{1,4}(\s*(?:of|/)\s*\d{1,4})?$", re.IGNORECASE),  # "Page 5", "Page 5 of 10"
    re.compile(r"^[-–—]\s*\d{1,4}\s*[-–—]$"),                  # "- 5 -"
]


def _is_page_number_text(text: str) -> bool:
    """True if a line is a standalone page-number/folio (e.g. "5", "5/5")."""

    candidate = (text or "").strip()
    if not candidate or len(candidate) > 20:
        return False
    return any(pattern.match(candidate) for pattern in _PAGE_NUMBER_PATTERNS)


# How tightly the repeated occurrences of a header/footer must cluster
# vertically (as a % of page height) to count as a fixed running element.
# A footer/header sits at the *same* place on every page; real flowing content
# that merely shares a (digit-stripped) key ("Statement 1 …" / "Statement 2 …",
# or repeated "Ans:" labels) lands at different heights page-to-page, so a tight
# position variance is what separates furniture from genuine repeated content.
_FURNITURE_POSITION_TOL_PCT = 3.0

# A position-stable repeat confined to a margin band is running furniture even
# when it only spans a *section* of a long document (e.g. a "PW Website" footer
# that appears solely on the solutions pages of a 30-page paper, which is far
# fewer than half the pages). Requiring it on at least this many pages — while
# also demanding tight position stability and margin confinement — keeps the
# rule from touching genuine repeated body content.
_FURNITURE_MIN_REPEAT = 3

# Repeating headers/footers live in the *outer* margins. The tight band is used
# for a weak (only-two-page) exact repeat where we want to stay conservative and
# not touch anything that isn't clearly in the top/bottom margin.
_FURNITURE_TOP_PCT = 6.0
_FURNITURE_BOTTOM_PCT = 90.0

# Anything at or above this top fraction / at or below this bottom fraction is
# unambiguously in a margin and can never be real question content, so a
# position-stable repeat there is furniture regardless of strength.
_FURNITURE_MARGIN_TOP_PCT = 15.0
_FURNITURE_MARGIN_BOTTOM_PCT = 85.0


def _is_marker_line(text: str) -> bool:
    """True if a line begins a question/solution or is a solutions header.

    Marker lines (e.g. "Q3 Text Solution:", "Hints & Solutions") repeat their
    boilerplate across pages but are *real content* that must open a crop, so
    they are never treated as page furniture.
    """

    return match_question_start_ex(text) is not None or match_solution_header(text)


def _find_furniture(
    lines: list[ContentLine],
    page_heights: dict[int, float],
    total_pages: int,
) -> set[tuple[int, float, float]]:
    """Return the set of (page, y_top, y_bottom) lines that are page furniture.

    A running header/footer (a date strip, a "UPSC" running title, a page
    number, an "Android App | iOS App | PW Website" branding line) is identified
    by two traits that real content never combines:

      1. its text **repeats across pages** — either verbatim, or once a per-page
         page-number/date is stripped (see :func:`_normalize_loose`); and
      2. it sits at a **stable vertical position** on every page it appears on.

    The position-stability test is the key discriminator. Repeated *labels*
    inside the body ("Ans:", "Exp:") also recur verbatim, but they ride along
    with their question and therefore land at a different height on each page, so
    they are kept. A footer pinned to ~88% of every page does not move, so it is
    removed — even when it sits a little above the very bottom margin and would
    otherwise be stitched into the middle of a cross-page crop.

    Question/solution marker lines are always preserved (see
    :func:`_is_marker_line`) so a repeated "Q3 Text Solution:" opening is never
    mistaken for furniture and dropped from the top of a crop.
    """

    if total_pages < 2:
        return set()

    strong_threshold = max(2, round(total_pages * 0.5))

    # Gather per-(normalized text) and per-(loose text) occurrences, recording
    # the page and top-position% of each so we can test position stability.
    def _collect(key_fn) -> dict[str, list[tuple[int, float, float]]]:
        out: dict[str, list[tuple[int, float, float]]] = {}
        for ln in lines:
            if _is_marker_line(ln.text):
                continue
            key = key_fn(ln.text)
            if not key:
                continue
            page_h = float(page_heights.get(ln.page_num) or 0.0)
            if page_h <= 0.0:
                continue
            top_pct = (ln.y_top / page_h) * 100.0
            bottom_pct = (ln.y_bottom / page_h) * 100.0
            out.setdefault(key, []).append((ln.page_num, top_pct, bottom_pct))
        return out

    exact_occ = _collect(_normalize_text)
    loose_occ = _collect(_normalize_loose)

    def _is_position_stable(occ: list[tuple[int, float, float]]) -> bool:
        tops = [t for _, t, _ in occ]
        return (max(tops) - min(tops)) <= _FURNITURE_POSITION_TOL_PCT

    def _pages(occ: list[tuple[int, float, float]]) -> set[int]:
        return {p for p, _, _ in occ}

    def _all_in_margin(occ: list[tuple[int, float, float]]) -> bool:
        return all(
            top <= _FURNITURE_MARGIN_TOP_PCT or bottom >= _FURNITURE_MARGIN_BOTTOM_PCT
            for _, top, bottom in occ
        )

    def _is_furniture_key(occ: list[tuple[int, float, float]]) -> bool:
        """A repeat is running furniture when it is pinned to one vertical
        position across pages and either (a) recurs on a large share of the
        document, or (b) recurs on a few pages but always inside a margin band
        (a section-local header/footer such as a solutions-only PW footer)."""

        if not _is_position_stable(occ):
            return False
        page_count = len(_pages(occ))
        if page_count >= strong_threshold:
            return True
        return page_count >= _FURNITURE_MIN_REPEAT and _all_in_margin(occ)

    # Keys whose repeats are pinned to one vertical position → running furniture.
    furniture_keys_exact: set[str] = {
        key for key, occ in exact_occ.items() if _is_furniture_key(occ)
    }
    furniture_keys_loose: set[str] = {
        key for key, occ in loose_occ.items() if _is_furniture_key(occ)
    }

    # Weak exact repeat (only a couple of pages): conservative — strip only when
    # it's clearly in the tight outer margin band.
    weak_exact: set[str] = {
        key for key, occ in exact_occ.items() if len(_pages(occ)) >= 2
    }

    # Topmost question/solution marker position (% of height) per page. A line
    # in the TOP margin band is only a running header when it sits *above* the
    # first marker on its page. If a marker sits above the line, the line is
    # that question's body content — e.g. the options line ("A) … B) …") of a
    # question whose stem begins at the very top of the page — and must never be
    # stripped as furniture. Without this guard a top-of-page question loses its
    # options (the reported "answers getting cut" bug). Marker lines themselves
    # are skipped by ``_collect``/``_is_marker_line``, so we read their position
    # straight from the line list.
    first_marker_pct: dict[int, float] = {}
    for ln in lines:
        if not _is_marker_line(ln.text):
            continue
        page_h = float(page_heights.get(ln.page_num) or 0.0)
        if page_h <= 0.0:
            continue
        tp = (ln.y_top / page_h) * 100.0
        cur = first_marker_pct.get(ln.page_num)
        if cur is None or tp < cur:
            first_marker_pct[ln.page_num] = tp

    def _marker_above(page_num: int, top_pct: float) -> bool:
        """True if a question/solution marker sits above this y-position."""

        fm = first_marker_pct.get(page_num)
        return fm is not None and fm < top_pct - 0.01

    furniture: set[tuple[int, float, float]] = set()
    for ln in lines:
        if _is_marker_line(ln.text):
            continue
        page_h = float(page_heights.get(ln.page_num) or 0.0)
        if page_h <= 0.0:
            continue
        norm = _normalize_text(ln.text)
        loose = _normalize_loose(ln.text)
        top_pct = (ln.y_top / page_h) * 100.0
        bottom_pct = (ln.y_bottom / page_h) * 100.0

        # A top-margin line is a header only when no marker precedes it on the
        # page; a marker above it means it is body content, not furniture.
        in_top_margin = top_pct <= _FURNITURE_MARGIN_TOP_PCT and not _marker_above(
            ln.page_num, top_pct
        )
        in_margin = in_top_margin or bottom_pct >= _FURNITURE_MARGIN_BOTTOM_PCT

        # A standalone page-number/folio ("5", "5/5", "Page 5 of 10") in a
        # margin is running furniture even though its per-page digit defeats the
        # repeat test (its loose key is empty and its exact text differs every
        # page). Drop it so it can't drag a column's crop bound to the page
        # bottom and pull the footer band into the crop.
        if in_margin and _is_page_number_text(ln.text):
            furniture.add((ln.page_num, ln.y_top, ln.y_bottom))
            continue

        if norm in furniture_keys_exact or loose in furniture_keys_loose:
            # A position-stable cross-page repeat. Strip it anywhere it is not
            # inside the central content region — i.e. throughout the margins
            # and the near-margin band where footers/headers live — but leave a
            # stable repeat that genuinely sits mid-page alone.
            if in_margin:
                furniture.add((ln.page_num, ln.y_top, ln.y_bottom))
            continue

        if norm in weak_exact:
            if (
                top_pct <= _FURNITURE_TOP_PCT
                and not _marker_above(ln.page_num, top_pct)
            ) or bottom_pct >= _FURNITURE_BOTTOM_PCT:
                furniture.add((ln.page_num, ln.y_top, ln.y_bottom))

    return furniture


# Top strip (% of page height) within which banner lines are considered.
_HEADER_BAND_PCT = 15.0

# The first real question marker must sit within this top fraction of the page
# for the lines above it to be considered a header. On a continuation page the
# first marker sits much lower (text flows in from the previous page), so we
# leave that page's top content alone.
_HEADER_MARKER_MAX_PCT = 25.0

# A header region above the first marker is only stripped when it is *sparse*
# (a few short banner lines: running title, "DPP: 50", "Answer Key", "Hints &
# Solutions"). A continuation page has many dense lines above its first marker,
# so a higher count means "real content", not a header.
_HEADER_MAX_LINES = 4


def _find_banner_headers(
    lines: list[ContentLine],
    starts: list[QuestionStart],
    page_heights: dict[int, float],
    page_columns: dict[int, list[tuple[float, float]]],
) -> set[tuple[int, float, float]]:
    """Return per-page banner/title lines that should not bleed into a crop.

    On a page whose first *strong* question marker sits near the top, the few
    short lines above it are page furniture — a running title ("Polity"), a
    banner ("DPP: 50") or a section header ("Answer Key", "Hints & Solutions").
    These are stripped so a question whose content flows to the top of the next
    column doesn't drag the banner into its crop.

    A continuation page (where the previous page's content flows in above the
    first marker) has *many* dense lines above that marker, so the sparsity
    check (``_HEADER_MAX_LINES``) leaves its real content intact. Likewise a
    page whose first marker is far down the page is skipped entirely.
    """

    # Topmost STRONG marker y per page (bare-number sub-statements ignored).
    first_marker_y: dict[int, float] = {}
    for s in starts:
        if not s.is_strong:
            continue
        cur = first_marker_y.get(s.page_num)
        if cur is None or s.y_top < cur:
            first_marker_y[s.page_num] = s.y_top

    lines_by_page: dict[int, list[ContentLine]] = {}
    for ln in lines:
        lines_by_page.setdefault(ln.page_num, []).append(ln)

    headers: set[tuple[int, float, float]] = set()

    for page_num, marker_y in first_marker_y.items():
        page_h = float(page_heights.get(page_num) or 0.0)
        if page_h <= 0.0:
            continue
        if (marker_y / page_h) * 100.0 > _HEADER_MARKER_MAX_PCT:
            # First question is far down the page → continuation page, skip.
            continue

        # Lines above the first marker that fall in the top band.
        pre_marker = [
            ln
            for ln in lines_by_page.get(page_num, [])
            if ln.y_top < marker_y and (ln.y_top / page_h) * 100.0 <= _HEADER_BAND_PCT
        ]
        if not pre_marker or len(pre_marker) > _HEADER_MAX_LINES:
            # Empty → nothing to strip. Too many → dense continuation content.
            continue

        for ln in pre_marker:
            headers.add((ln.page_num, ln.y_top, ln.y_bottom))

    return headers


def _recover_missing_markers(
    starts: list[QuestionStart],
    lines: list[ContentLine],
    expected_numbers: Optional[set[int]] = None,
) -> list[QuestionStart]:
    """Find markers whose number OCR misread, using the numbering sequence.

    Questions and solutions are handled independently (each has its own 1..N
    run). For a side with a mostly-sequential run of numbers, any *single*
    interior number that's missing is a recovery candidate: we scan the content
    lines that sit between the present neighbours (in page→y order) for one
    whose leading token resolves to the missing number under OCR digit fixups,
    and emit a synthesised :class:`QuestionStart` at that line. The new marker
    inherits the weak/strong strength of its side so downstream filtering treats
    it like a normal bare-number marker.

    ``expected_numbers`` (when supplied, e.g. parsed from the answer key) is the
    authoritative set of question numbers the paper should contain. It lets
    recovery target numbers the *sequence* alone wouldn't reveal — a question
    missing from the very end of the run, or several in a row — because the key
    proves they exist. It only augments the question side (solutions aren't in a
    question answer key).

    Conservative by construction: it only fills a number it independently
    expects (from the sequence or the key), only from a line physically between
    plausible neighbours, and never creates a number that already exists.
    """

    if not lines:
        return []

    recovered: list[QuestionStart] = []

    for is_solution in (False, True):
        side_starts = [s for s in starts if bool(s.is_solution) == is_solution]

        # Map present number -> its start (first occurrence wins).
        num_to_start: dict[int, QuestionStart] = {}
        for s in side_starts:
            digits = re.findall(r"\d+", s.q_num)
            if not digits:
                continue
            n = int(digits[0])
            num_to_start.setdefault(n, s)

        present = sorted(num_to_start)

        # Determine which numbers to try to recover. The answer key (question
        # side only) is authoritative when present; otherwise fall back to the
        # interior-gap heuristic which needs a trustworthy sequence.
        use_key = (not is_solution) and bool(expected_numbers)
        if use_key:
            missing = sorted(set(expected_numbers) - set(num_to_start))  # type: ignore[arg-type]
            if not missing:
                continue
        else:
            if len(present) < 3:
                continue
            lo, hi = present[0], present[-1]
            missing = [n for n in range(lo, hi + 1) if n not in num_to_start]
            if not missing:
                continue
            # Only attempt recovery on a mostly-complete run (a couple of holes),
            # not a sparse scatter that isn't really sequential.
            if len(missing) > max(3, (hi - lo) // 3):
                continue

        def _doc_key(page: int, y: float) -> tuple[int, float]:
            return (page, y)

        for n in missing:
            prev_start = num_to_start.get(n - 1)
            next_start = num_to_start.get(n + 1)
            if prev_start is None and next_start is None:
                # With a key we may know n exists but have neither neighbour
                # (a run of misses). Fall back to the nearest known numbers.
                lower_num = max((p for p in num_to_start if p < n), default=None)
                upper_num = min((p for p in num_to_start if p > n), default=None)
                prev_start = num_to_start.get(lower_num) if lower_num is not None else None
                next_start = num_to_start.get(upper_num) if upper_num is not None else None
                if prev_start is None and next_start is None:
                    continue

            lower = (
                _doc_key(prev_start.page_num, prev_start.y_top)
                if prev_start is not None
                else (-1, -1.0)
            )
            upper = (
                _doc_key(next_start.page_num, next_start.y_top)
                if next_start is not None
                else (10**9, 10**9)
            )

            best: Optional[ContentLine] = None
            for ln in lines:
                key = _doc_key(ln.page_num, ln.y_top)
                if key <= lower or key >= upper:
                    continue
                if line_matches_expected_number(ln.text, n):
                    best = ln
                    break

            if best is None:
                continue

            recovered.append(
                QuestionStart(
                    page_num=best.page_num,
                    y_top=best.y_top,
                    q_num=str(n),
                    is_solution=is_solution,
                    x_left=best.x_left,
                    x_right=best.x_right,
                    is_strong=False,
                )
            )
            # Register so a neighbouring gap can chain off the recovered marker.
            num_to_start[n] = recovered[-1]

    return recovered


# --- Numbered-option suppression --------------------------------------------
#
# Some papers (most SSC / many Indian exam papers) label their four MCQ choices
# with *numbers* — "1. … 2. … 3. … 4. …" — instead of letters "(A)-(D)". Those
# option labels are indistinguishable, character-for-character, from a bare
# question marker ("1."), so the weak-marker matcher treats every option as a
# brand-new question. A 100-question paper then explodes into 700+ detected
# items (each question's stem plus its four options), and the review panel fills
# with hundreds of bogus "Question 1 / 2 / 3 / 4" entries.
#
# The give-away is typographic, not textual: a real question number *hangs* at
# its column's left text margin, while every option is *indented* past it (the
# option sits under the stem, not under the number). We learn each page's
# question margin(s) from "anchor" markers whose number is too big to be one of
# the four options (>= _ANCHOR_MIN_NUMBER), since those can only be questions,
# then drop any weak marker indented past every margin as an option.
#
# This only ever runs on weak (bare-number) markers, and only when a confident
# anchor margin exists — exactly the papers that have this ambiguity. Strong-Q
# papers and lettered-option papers carry no weak option markers, so nothing is
# dropped there.

# How far (in pixels) past a page's marker margin a weak marker must start before
# it is treated as an indented option rather than a question. A genuine question
# number sits within a glyph-width of the margin; an option is indented by a
# stem's worth (tens of px). Kept moderate so a real question at the margin is
# never reclassified, while an indented option always is.
_OPTION_INDENT_MIN_PX = 9.0

# A page's question margin is anchored by markers whose number is too large to
# be one of the four options. Only ``>= _ANCHOR_MIN_NUMBER`` markers define a
# margin, since a "5." could still be an option in a 5-option paper but a "6."
# essentially never is.
_ANCHOR_MIN_NUMBER = 6

# Two anchor left-edges within this many px belong to the same margin (one
# column). A two-column page yields two well-separated margin clusters; jitter
# inside one column stays well under this.
_MARGIN_CLUSTER_TOL_PX = 6.0


def _weak_marker_number(start: "QuestionStart") -> Optional[int]:
    digits = re.findall(r"\d+", start.q_num or "")
    return int(digits[0]) if digits else None


def _cluster_margins(values: list[float], tol: float) -> list[float]:
    """Collapse nearby left-edge x's into one representative margin each.

    Sorted values within ``tol`` px of the running cluster are merged; each
    cluster is represented by its minimum (the true hanging margin). A
    single-column page yields one margin, a two-column page two.
    """

    if not values:
        return []
    ordered = sorted(values)
    clusters: list[list[float]] = [[ordered[0]]]
    for v in ordered[1:]:
        if v - clusters[-1][-1] <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [min(c) for c in clusters]


def drop_numbered_options(starts: list["QuestionStart"]) -> list["QuestionStart"]:
    """Drop weak markers that are really *numbered MCQ options*, not questions.

    For each (page, side) we learn the question margin(s) — the left-x of markers
    whose number is too big to be an option (``>= _ANCHOR_MIN_NUMBER``), clustered
    so a two-column page keeps one margin per column — and discard any weak marker
    indented more than :data:`_OPTION_INDENT_MIN_PX` past *every* margin. Strong
    "Q" markers and markers on a (page, side) with no high-numbered anchor are
    always kept, so the filter only acts where it has firm evidence and never
    touches a lettered-option or strong-Q paper.
    """

    if not starts:
        return starts

    # Anchor margins per (page, is_solution), from high-numbered weak markers.
    anchors: dict[tuple[int, bool], list[float]] = {}
    for s in starts:
        if s.is_strong:
            continue
        num = _weak_marker_number(s)
        if num is not None and num >= _ANCHOR_MIN_NUMBER:
            anchors.setdefault((s.page_num, bool(s.is_solution)), []).append(s.x_left)

    margins: dict[tuple[int, bool], list[float]] = {
        key: _cluster_margins(vals, _MARGIN_CLUSTER_TOL_PX)
        for key, vals in anchors.items()
    }

    kept: list["QuestionStart"] = []
    for s in starts:
        if s.is_strong:
            kept.append(s)
            continue
        cols = margins.get((s.page_num, bool(s.is_solution)))
        if not cols:
            # No confident margin on this page/side — leave the markers alone.
            kept.append(s)
            continue
        # The marker's own column is the rightmost margin at or left of it (a
        # small tolerance lets a marker sitting a hair left of its margin still
        # bind to it). Anything further right than that is the column's body.
        own = None
        for m in cols:
            if s.x_left >= m - _OPTION_INDENT_MIN_PX:
                own = m
        if own is None:
            own = cols[0]
        # Hanging at its column margin → a question; indented past it → an
        # option label, drop it.
        if s.x_left <= own + _OPTION_INDENT_MIN_PX:
            kept.append(s)

    return kept


# A grid row must hold at least this many answer-key cells before the row is
# trusted as part of a compact answer-key table (and its markers dropped). A
# couple of "1. (1)"-shaped lines in ordinary prose never reach this; a real key
# packs a dozen-plus per row.
_ANSWER_KEY_MIN_CELLS_PER_ROW = 4

# Rows whose baselines sit within this many px belong to the same grid row.
_ANSWER_KEY_ROW_TOL_PX = 4.0


def drop_answer_key_cells(
    starts: list["QuestionStart"],
    lines: list["ContentLine"],
) -> list["QuestionStart"]:
    """Drop markers that are really *answer-key grid cells*, not solutions.

    Many papers print a compact answer key ("1. (1)  2. (2)  3. (3) …") before
    the detailed solutions. Each cell's leading "1." reads as a marker and the
    grid repeats the whole 1..N numbering, so every real solution is duplicated
    by a key cell. We find rows that pack several answer-key cells
    (``>= _ANSWER_KEY_MIN_CELLS_PER_ROW`` lone "n. (x)" lines at the same
    baseline) and drop any marker that sits on such a row.

    Conservative: only lone-cell lines count, and only when many share a row, so
    a stray "1. (1)" inside a sentence or a normal numbered item is never
    removed.
    """

    if not starts or not lines:
        return starts

    # Group answer-key-cell lines by (page, rounded baseline) into rows.
    rows: dict[tuple[int, int], list["ContentLine"]] = {}
    for ln in lines:
        if not _is_answer_key_cell(getattr(ln, "text", "")):
            continue
        row_key = (ln.page_num, int(round(ln.y_top / _ANSWER_KEY_ROW_TOL_PX)))
        rows.setdefault(row_key, []).append(ln)

    # Vertical bands (per page) covered by a dense answer-key row.
    grid_bands: dict[int, list[tuple[float, float]]] = {}
    for (page, _), cells in rows.items():
        if len(cells) < _ANSWER_KEY_MIN_CELLS_PER_ROW:
            continue
        top = min(c.y_top for c in cells)
        bottom = max(c.y_bottom for c in cells)
        grid_bands.setdefault(page, []).append((top, bottom))

    if not grid_bands:
        return starts

    def _on_grid_row(s: "QuestionStart") -> bool:
        bands = grid_bands.get(s.page_num)
        if not bands:
            return False
        return any(top - 1.0 <= s.y_top <= bottom + 1.0 for top, bottom in bands)

    return [s for s in starts if not _on_grid_row(s)]


# --- Marker-cluster column fallback -----------------------------------------
#
# A tight two-column paper whose *options* are numbered "1.-4." fills the middle
# gutter with option text, so :func:`detect_columns` sees no whitespace valley
# and collapses the page to a single full-width column. Reading order then runs
# straight down the page across both physical columns, so each question is cut
# off at the next marker in y-order and its crop becomes a sliver. The question
# *markers*, however, are an unambiguous column signal: they hang at each
# column's left margin, forming well-separated x-clusters (e.g. x≈72 and x≈312).
# When the geometry-based detector found only one column we rebuild the columns
# from those marker clusters instead.

# Minimum horizontal gap between two marker clusters, as a fraction of page
# width, for them to count as separate columns. Two real columns are separated
# by far more than this; jitter within one column is far less.
_MARKER_COLUMN_GAP_FRAC = 0.12

# A marker cluster must hold at least this many markers to anchor a column, so a
# lone stray marker on the right half never splits a single-column page.
_MIN_MARKERS_PER_COLUMN = 2

# Padding (px) left of a column's marker margin when placing the column's left
# edge, so the marker glyph itself is inside the column.
_MARKER_COLUMN_PAD_PX = 6.0


def columns_from_markers(
    page_starts: list["QuestionStart"], page_width: float
) -> Optional[list[tuple[float, float]]]:
    """Derive column x-ranges from question-marker clusters, or None.

    Markers hang at each column's left margin, so their ``x_left`` values form
    one tight cluster per column. We cluster them by horizontal gap and, when at
    least two clusters each carry ``>= _MIN_MARKERS_PER_COLUMN`` markers, tile
    the page into columns split just left of each cluster's margin. Returns None
    when the markers don't clearly describe a multi-column layout, so the caller
    keeps whatever the geometry detector found.
    """

    if page_width <= 0 or len(page_starts) < 2 * _MIN_MARKERS_PER_COLUMN:
        return None

    xs = sorted(s.x_left for s in page_starts)
    gap = page_width * _MARKER_COLUMN_GAP_FRAC

    clusters: list[list[float]] = [[xs[0]]]
    for x in xs[1:]:
        if x - clusters[-1][-1] > gap:
            clusters.append([x])
        else:
            clusters[-1].append(x)

    big = [c for c in clusters if len(c) >= _MIN_MARKERS_PER_COLUMN]
    if len(big) < 2:
        return None

    # Left margin of each column = its cluster's leftmost marker, minus a small
    # pad so the marker glyph is included. Columns tile left→right, each ending
    # where the next begins.
    margins = [min(c) for c in big]
    columns: list[tuple[float, float]] = []
    for i, margin in enumerate(margins):
        left = 0.0 if i == 0 else max(0.0, margins[i] - _MARKER_COLUMN_PAD_PX)
        right = (
            float(page_width)
            if i == len(margins) - 1
            else max(0.0, margins[i + 1] - _MARKER_COLUMN_PAD_PX)
        )
        columns.append((left, right))

    return columns


def starts_to_questions(
    starts: list[QuestionStart],
    page_heights: dict[int, float],
    total_pages: int,
    content_lines: Optional[list[ContentLine]] = None,
    page_widths: Optional[dict[int, float]] = None,
    figures: Optional[list[FigureRegion]] = None,
) -> list[DetectedQuestion]:
    """Convert question-start markers + content lines into DetectedQuestion objects.

    Reading order is page → column → vertical. For each page we detect the
    column layout from the horizontal extent of its content lines (one column
    for normal documents, two or more for side-by-side layouts). Each question
    collects the content lines that fall between its marker and the next
    marker, builds one segment per (page, column) tightly bounded vertically,
    and records the column's x-range so the crop stays inside that column.

    This means:
      - Two-column pages are read left column top-to-bottom, then right column.
      - A question's crop is confined to its own column's width.
      - A question spans multiple segments only when its content genuinely
        continues into another column or page, and those segments are stitched.

    ``content_lines`` is optional; when omitted the marker itself is used as a
    single-line fallback so a question is never dropped.
    """

    if total_pages <= 0:
        return []

    # If the document uses explicit "Q" markers anywhere in a section, bare
    # numbered lines in that section are sub-statements ("1. ... 2. ...") rather
    # than new questions. Drop the weak markers so they don't fragment a
    # question. Questions and solutions are scoped independently because a paper
    # may "Q"-number its questions but plainly number its answer key (or vice
    # versa). Papers with no "Q" markers at all keep their bare numbering.
    starts = list(starts)
    has_strong_q = any(s.is_strong for s in starts if not s.is_solution)
    has_strong_s = any(s.is_strong for s in starts if s.is_solution)
    filtered_starts: list[QuestionStart] = []
    for s in starts:
        section_has_strong = has_strong_s if s.is_solution else has_strong_q
        if section_has_strong and not s.is_strong:
            continue
        filtered_starts.append(s)
    starts = filtered_starts

    # Drop numbered MCQ option labels ("1. 2. 3. 4." under a stem) that the weak
    # matcher mistook for question starts. Real question numbers hang at the
    # column's left margin; options are indented past it. Done before gap
    # recovery so recovery sees a clean question sequence (otherwise every
    # question's "1.-4." options look like a dense, gap-free run).
    starts = drop_numbered_options(starts)

    # Drop answer-key grid cells ("1. (1)  2. (2) …") that read as markers but
    # are the compact key, not croppable solutions — they otherwise duplicate
    # every real solution number. Needs the content lines to spot the dense grid
    # rows, so it runs once those are in hand below.
    widths = dict(page_widths or {})
    lines = list(content_lines or [])
    figure_list = list(figures or [])

    starts = drop_answer_key_cells(starts, lines)

    # Gap recovery: re-read missed markers whose number OCR mangled. If the
    # detected numbers run 1,2,…,19,21,… the absent 20 is usually present in the
    # text but its number was misread ("20." -> "2O.", "2C", "Z0"), so it never
    # matched a marker pattern and the question silently vanished. We look for a
    # content line between the neighbours of each gap whose leading token reads
    # as the missing number (with OCR digit fixups) and synthesise a marker
    # there, so the question is cropped instead of dropped.
    #
    # When the paper has an answer key, it lists every question number exactly
    # once — authoritative ground truth for how many questions exist. We parse
    # it and feed the implied 1..N run to recovery so it can target numbers the
    # detected *sequence* alone wouldn't reveal (a question missing from the end,
    # or several in a row).
    answer_key = extract_answer_key(lines)
    expected_q_numbers = expected_question_numbers(answer_key) if answer_key else None
    recovered = _recover_missing_markers(starts, lines, expected_q_numbers)
    if recovered:
        starts = starts + recovered

    # Strip repeating header/footer furniture (running titles, page branding) so
    # question crops don't bleed into the page footer or start at a header.
    furniture = _find_furniture(lines, page_heights, total_pages)
    if furniture:
        lines = [
            ln for ln in lines if (ln.page_num, ln.y_top, ln.y_bottom) not in furniture
        ]

    # Per-page column layouts derived from content-line x-extents.
    page_columns: dict[int, list[tuple[float, float]]] = {}
    for page_num, page_width in widths.items():
        intervals = [
            (ln.x_left, ln.x_right)
            for ln in lines
            if ln.page_num == page_num and ln.x_right > ln.x_left
        ]
        cols = detect_columns(intervals, float(page_width))
        cols = _validate_columns_with_markers(
            cols, starts, page_num, float(page_width), lines
        )
        # A tight two-column page whose numbered options fill the gutter defeats
        # the whitespace-based detector, collapsing it to one column and slicing
        # every question into a sliver. When the markers themselves describe a
        # clear multi-column layout, trust them instead.
        if len(cols) <= 1:
            page_starts = [s for s in starts if s.page_num == page_num]
            marker_cols = columns_from_markers(page_starts, float(page_width))
            if marker_cols is not None:
                cols = marker_cols
        page_columns[page_num] = cols

    # Strip isolated banner/title lines above the first question on a page
    # ("Polity", "DPP: 50", "Answer Key", "Hints & Solutions"). These sit in the
    # top strip of one column only, so without this a question whose content
    # flows to the top of the next column would drag the banner into its crop.
    banners = _find_banner_headers(lines, starts, page_heights, page_columns)
    if banners:
        lines = [
            ln for ln in lines if (ln.page_num, ln.y_top, ln.y_bottom) not in banners
        ]

    def _cols_for(page_num: int) -> list[tuple[float, float]]:
        cols = page_columns.get(page_num)
        if cols:
            return cols
        return [(0.0, float(widths.get(page_num) or 0.0) or 1.0)]

    def _col_of(page_num: int, x_left: float, x_right: float) -> int:
        return _column_index(_cols_for(page_num), x_left, x_right)

    # Order markers in true reading order: page → column → vertical.
    ordered = sorted(
        starts,
        key=lambda s: _pos(s.page_num, _col_of(s.page_num, s.x_left, s.x_right), s.y_top),
    )

    # Light de-duplication: same q_num on same page/column very close together.
    deduped: list[QuestionStart] = []
    for start in ordered:
        if (
            deduped
            and start.page_num == deduped[-1].page_num
            and start.q_num == deduped[-1].q_num
            and start.is_solution == deduped[-1].is_solution
            and abs(start.y_top - deduped[-1].y_top) < 2.0
        ):
            continue
        deduped.append(start)

    if not deduped:
        return []

    # Pre-tag content lines with their (page, column) and sort in reading order.
    tagged_lines = [
        (_col_of(ln.page_num, ln.x_left, ln.x_right), ln) for ln in lines
    ]
    tagged_lines.sort(key=lambda t: _pos(t[1].page_num, t[0], t[1].y_top))

    # Tag figures with the column they belong to. A figure is folded into the
    # question whose text vertically surrounds it (the diagram between a stem and
    # its options) or immediately precedes it (a trailing graph), so the crop
    # grows to contain the whole graphic instead of clipping it.
    tagged_figures = [
        (_col_of(fig.page_num, fig.x_left, fig.x_right), fig) for fig in figure_list
    ]

    # Tight horizontal *content* bounds per (page, column). Cropping to these,
    # rather than the gutter-center tile, has two effects the user asked for:
    #   1. The inter-column divider rule lives in the empty gutter (no text), so
    #      it falls outside every column's content bounds and is excluded.
    #   2. Each column is cropped from its own left text margin, so when a
    #      question's segments are stacked the list markers ("1." over "2."/"3.")
    #      line up instead of drifting left/right.
    col_content_bounds: dict[tuple[int, int], tuple[float, float]] = {}
    for col, ln in tagged_lines:
        if ln.x_right <= ln.x_left:
            continue
        key = (ln.page_num, col)
        existing = col_content_bounds.get(key)
        if existing is None:
            col_content_bounds[key] = (ln.x_left, ln.x_right)
        else:
            col_content_bounds[key] = (
                min(existing[0], ln.x_left),
                max(existing[1], ln.x_right),
            )

    def _x_range(page: int, col: int) -> tuple[float, float]:
        """Horizontal crop range for a (page, column): tight content bounds when
        known, otherwise the column tile."""

        cb = col_content_bounds.get((page, col))
        if cb is not None:
            return cb
        cols = _cols_for(page)
        page_width = float(widths.get(page) or 0.0)
        return cols[col] if col < len(cols) else (0.0, page_width)

    # Snap each marker's effective top to the top of its own text row. A question
    # number ("20.") is frequently rendered in its own text block whose y_top
    # sits a hair *below* the first words of the same-row stem ("2.5 mL of 2 …"),
    # because the number and the prose share a baseline but differ slightly in
    # cap height. Ordering content purely by y_top then files that stem fragment
    # under the PREVIOUS question (its y_top is above the marker glyph), so the
    # previous crop swallows the next question's opening line and the next crop
    # loses it — the reported "Q20 ka pehla part Q19 me chala gaya / Q20 missing"
    # bug. Lowering each marker's boundary to its row top fixes both sides at
    # once: the current question keeps its first line, and the previous one stops
    # before it.
    def _marker_row_top(start: "QuestionStart") -> float:
        s_col = _col_of(start.page_num, start.x_left, start.x_right)
        page_h = float(page_heights.get(start.page_num) or 0.0)
        if page_h <= 0.0:
            return start.y_top
        # A genuine same-row stem *straddles* the marker's top edge: it starts at
        # or just above the number glyph and extends meaningfully below the
        # marker's top (they share a baseline). The previous question's last line
        # sits entirely ABOVE the next marker (no vertical overlap), so requiring
        # real overlap — not mere proximity — keeps it from being mistaken for
        # the marker's stem (which would steal that line's options).
        top_tol = page_h * 0.006  # stem may start a hair above the number's cap
        overlap_min = page_h * 0.004  # must dip below the marker top by this much
        row_top = start.y_top
        for col, ln in tagged_lines:
            if ln.page_num != start.page_num or col != s_col:
                continue
            # Only a line at/right of the number may pull the top up.
            if ln.x_left + 1e-6 < start.x_left:
                continue
            if (
                ln.y_top <= start.y_top + top_tol
                and ln.y_bottom >= start.y_top + overlap_min
            ):
                row_top = min(row_top, ln.y_top)
        return row_top

    row_top_by_idx = [_marker_row_top(s) for s in deduped]

    questions: list[DetectedQuestion] = []

    for idx, current in enumerate(deduped):
        next_start = deduped[idx + 1] if idx + 1 < len(deduped) else None

        cur_col = _col_of(current.page_num, current.x_left, current.x_right)
        marker_pos = _pos(current.page_num, cur_col, row_top_by_idx[idx])
        next_pos = (
            _pos(
                next_start.page_num,
                _col_of(next_start.page_num, next_start.x_left, next_start.x_right),
                row_top_by_idx[idx + 1],
            )
            if next_start is not None
            else None
        )

        # Per (page, column), accumulate the tight (min y_top, max y_bottom) of
        # this question's content lines.
        seg_bounds: dict[tuple[int, int], tuple[float, float]] = {}
        seen_options: set[str] = set()
        section_cutoff_pos: Optional[tuple[int, int, float]] = None
        for col, ln in tagged_lines:
            line_pos = _pos(ln.page_num, col, ln.y_top)
            if line_pos < marker_pos:
                continue
            if next_pos is not None and line_pos >= next_pos:
                break
            # A section/part divider ("PART-II (CHEMISTRY)", "SECTION-1",
            # "This section contains FOUR (04) questions") sits *between* two
            # questions and belongs to neither. Stop the current question's
            # content at the divider so its crop ends at its own last option
            # instead of swallowing the divider block — and, because the divider
            # precedes the next marker, the next question's opening lines too.
            if match_section_header(ln.text):
                section_cutoff_pos = line_pos
                break
            seen_options.update(_option_letters_in(ln.text))
            key = (ln.page_num, col)
            existing = seg_bounds.get(key)
            if existing is None:
                seg_bounds[key] = (ln.y_top, ln.y_bottom)
            else:
                seg_bounds[key] = (
                    min(existing[0], ln.y_top),
                    max(existing[1], ln.y_bottom),
                )

        # Fold in any figures (diagrams/graphs/images) that belong to this
        # question — i.e. that fall between this marker and the next one in
        # reading order. A figure owns both a vertical and a horizontal extent,
        # so it can grow the crop downward (a trailing diagram) and outward (a
        # diagram wider than the text). Without this the crop would clip the
        # figure to the text bounds.
        seg_fig_bounds: dict[tuple[int, int], tuple[float, float, float, float]] = {}
        # A divider found above caps the figure search at the same position so a
        # diagram belonging to the next section isn't folded into this question.
        fig_cutoff_pos = section_cutoff_pos if section_cutoff_pos is not None else next_pos
        for col, fig in tagged_figures:
            fig_pos = _pos(fig.page_num, col, fig.y_top)
            if fig_pos < marker_pos:
                continue
            if fig_cutoff_pos is not None and fig_pos >= fig_cutoff_pos:
                continue
            key = (fig.page_num, col)
            existing = seg_fig_bounds.get(key)
            if existing is None:
                seg_fig_bounds[key] = (fig.x_left, fig.x_right, fig.y_top, fig.y_bottom)
            else:
                seg_fig_bounds[key] = (
                    min(existing[0], fig.x_left),
                    max(existing[1], fig.x_right),
                    min(existing[2], fig.y_top),
                    max(existing[3], fig.y_bottom),
                )

        # Fallback: no content lines supplied — anchor to the marker line so we
        # still emit something rather than dropping the question.
        if not seg_bounds and not seg_fig_bounds:
            page_height = float(page_heights.get(current.page_num) or 0.0)
            if page_height <= 0.0:
                continue
            page_width = float(widths.get(current.page_num) or 0.0)
            c0, c1 = _x_range(current.page_num, cur_col)
            x_start_pct = _clamp_pct((c0 / page_width) * 100.0) if page_width > 0 else 0.0
            x_end_pct = _clamp_pct((c1 / page_width) * 100.0) if page_width > 0 else 100.0
            y_start_pct = _clamp_pct((float(current.y_top) / page_height) * 100.0)
            questions.append(
                DetectedQuestion(
                    q_num=current.q_num,
                    is_solution=current.is_solution,
                    segments=[
                        QuestionSegment(
                            page=current.page_num,
                            y_start_pct=y_start_pct,
                            y_end_pct=100.0,
                            x_start_pct=x_start_pct,
                            x_end_pct=x_end_pct,
                        )
                    ],
                )
            )
            continue

        segments: list[QuestionSegment] = []
        seg_keys = sorted(
            set(seg_bounds) | set(seg_fig_bounds), key=lambda k: (k[0], k[1])
        )
        for (page, col) in seg_keys:
            page_height = float(page_heights.get(page) or 0.0)
            if page_height <= 0.0:
                continue
            page_width = float(widths.get(page) or 0.0)
            c0, c1 = _x_range(page, col)

            text_bounds = seg_bounds.get((page, col))
            if text_bounds is not None:
                top, bottom = text_bounds
            else:
                # Figure-only segment (a diagram with no text in this column):
                # seed the vertical extent from the figure itself.
                fx0, fx1, fy0, fy1 = seg_fig_bounds[(page, col)]
                top, bottom = fy0, fy1

            # Grow the segment to contain any figure that belongs here, both
            # vertically (a diagram below the text) and horizontally (a diagram
            # wider than the text column).
            fig_bounds = seg_fig_bounds.get((page, col))
            if fig_bounds is not None:
                fx0, fx1, fy0, fy1 = fig_bounds
                top = min(top, fy0)
                bottom = max(bottom, fy1)
                c0 = min(c0, fx0)
                c1 = max(c1, fx1)

            x_start_pct = _clamp_pct((c0 / page_width) * 100.0) if page_width > 0 else 0.0
            x_end_pct = _clamp_pct((c1 / page_width) * 100.0) if page_width > 0 else 100.0
            y_start_pct = _clamp_pct((float(top) / page_height) * 100.0)
            y_end_pct = _clamp_pct((float(bottom) / page_height) * 100.0)
            if y_end_pct > y_start_pct:
                segments.append(
                    QuestionSegment(
                        page=page,
                        y_start_pct=y_start_pct,
                        y_end_pct=y_end_pct,
                        x_start_pct=x_start_pct,
                        x_end_pct=x_end_pct,
                    )
                )

        if not segments:
            continue

        segments.sort(key=lambda s: (s.page, s.x_start_pct, s.y_start_pct))
        questions.append(
            DetectedQuestion(
                q_num=current.q_num,
                is_solution=current.is_solution,
                segments=segments,
                option_labels="".join(sorted(seen_options)),
            )
        )

    # Sort questions numerically when possible.
    # Questions come first (is_solution=False), then solutions (is_solution=True),
    # each group ordered by its number.
    def _q_key(q: DetectedQuestion) -> tuple[int, int, str]:
        digits = re.findall(r"\d+", q.q_num)
        return (1 if q.is_solution else 0, int(digits[0]) if digits else 10**9, q.q_num)

    questions.sort(key=_q_key)
    return questions
