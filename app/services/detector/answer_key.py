"""Answer-key parsing for self-validation.

Most exam papers carry an *answer key* — a compact grid pairing each question
number with its correct option, e.g.::

    1. (b)   2. (a)   3. (d)   4. (c)
    5. (a)   6. (b)   ...

or "1-B 2-A 3-D", "1 (B) 2 (C)", "Q.1 B  Q.2 D". The key lists *every* question
number exactly once, so parsing it gives the detector **ground truth** for how
many questions the paper has and which numbers exist. The pipeline uses that to:

  * tell the user precisely which numbers are missing from the crops, and
  * drive gap recovery with a known target instead of guessing from a sequence.

The parser is deliberately conservative: it only reports a key when it sees a
*dense, mostly-sequential run* of ``number → A-D`` pairs, so an ordinary
question whose options happen to contain digits is never mistaken for a key.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from .base import ContentLine


# A number immediately followed by a single A-D option letter, with the usual
# separators ("1.", "1)", "1-", "1 :", "Q1"). The letter may be parenthesised.
# A trailing boundary stops "12 A" from also swallowing the next pair's digit.
_PAIR_PATTERN: re.Pattern[str] = re.compile(
    r"(?:^|[\s,;|])(?:Q\.?\s*)?(\d{1,3})\s*[\.\)\-:]?\s*\(?\s*([A-Da-d])\s*\)?(?=$|[\s,;|])",
    re.IGNORECASE,
)

# A line/section is only treated as an answer key when at least this many pairs
# are present across the scanned text. Below this it's almost certainly normal
# body text that happened to contain a "number letter" coincidence.
_MIN_PAIRS = 6

# The matched numbers must be mostly sequential: the count of distinct numbers
# divided by the span (max-min+1) must clear this. A real key covers nearly the
# whole 1..N run; scattered coincidences don't.
_MIN_SEQUENCE_COVERAGE = 0.7


def _pairs_in_text(text: str) -> list[tuple[int, str]]:
    """Return every ``(number, option_letter)`` pair found in a string."""

    out: list[tuple[int, str]] = []
    for m in _PAIR_PATTERN.finditer(text or ""):
        try:
            num = int(m.group(1))
        except (TypeError, ValueError):
            continue
        if num <= 0 or num > 999:
            continue
        out.append((num, m.group(2).upper()))
    return out


def extract_answer_key(lines: "Iterable[ContentLine]") -> dict[int, str]:
    """Return ``{question_number: option_letter}`` parsed from an answer key.

    Scans all content lines, collecting ``number → A-D`` pairs. Returns the
    mapping only when the pairs are numerous (``>= _MIN_PAIRS``) and form a
    mostly-sequential run (``>= _MIN_SEQUENCE_COVERAGE`` coverage of their
    span); otherwise returns an empty dict. When a number appears more than once
    (a key reprinted in two places), the first option seen wins.
    """

    pairs: list[tuple[int, str]] = []
    for ln in lines:
        text = getattr(ln, "text", "") or ""
        pairs.extend(_pairs_in_text(text))

    return _pairs_to_key(pairs)


def extract_answer_key_from_text(text: str) -> dict[int, str]:
    """Like :func:`extract_answer_key` but for a single pre-joined string."""

    return _pairs_to_key(_pairs_in_text(text))


def _pairs_to_key(pairs: list[tuple[int, str]]) -> dict[int, str]:
    if len(pairs) < _MIN_PAIRS:
        return {}

    key: dict[int, str] = {}
    for num, letter in pairs:
        key.setdefault(num, letter)

    nums = sorted(key)
    if len(nums) < _MIN_PAIRS:
        return {}

    span = nums[-1] - nums[0] + 1
    coverage = len(nums) / float(span) if span > 0 else 0.0
    if coverage < _MIN_SEQUENCE_COVERAGE:
        return {}

    return key


def expected_question_numbers(key: dict[int, str]) -> set[int]:
    """Return the full 1..N run implied by a parsed answer key.

    The key may itself have a hole (a number whose answer line was mangled), so
    we return the complete contiguous run from its min to its max — that's the
    set of numbers the paper is expected to contain.
    """

    if not key:
        return set()
    nums = sorted(key)
    return set(range(nums[0], nums[-1] + 1))
