"""User-supplied page-range parsing and relabeling.

Lets the caller explicitly say which pages contain *questions* and which pages
contain *answers/solutions*, e.g. "questions on pages 1-5, answers on 7-10".

When ranges are provided they override the automatic
"Solutions / Answer Key" header detection:

  - Items whose primary page is in the answer range  -> is_solution = True
  - Items whose primary page is in the question range -> is_solution = False

Dropping only happens when BOTH ranges are supplied (an explicit, complete
partition of the document): an item on a page in neither range is dropped.
When only ONE range is supplied, the other side falls back to auto-detection
so the unspecified pages are kept with their detected labels rather than
discarded.

When no ranges are provided, detection results pass through unchanged.
"""

from __future__ import annotations

import re

from ..models.schemas import DetectedQuestion


class PageRangeError(ValueError):
    """Raised when a page-range spec can't be parsed."""


def parse_page_ranges(spec: str | None, max_page: int | None = None) -> set[int]:
    """Parse a page-range spec into a set of 1-indexed page numbers.

    Accepted formats (comma-separated, mix freely):
      - "1-5"        -> {1, 2, 3, 4, 5}
      - "1 to 5"     -> {1, 2, 3, 4, 5}
      - "8"          -> {8}
      - "1-5, 8, 10-12"

    Returns an empty set for empty/None input. Raises ``PageRangeError`` for
    malformed input. Pages below 1 or above ``max_page`` (when given) are
    ignored rather than raising, so a generous range like "1-100" stays valid.
    """

    if not spec or not spec.strip():
        return set()

    # Treat the word "to" as a range separator: "1 to 5" -> "1 - 5".
    normalized = re.sub(r"\bto\b", "-", spec, flags=re.IGNORECASE)

    pages: set[int] = set()
    for chunk in normalized.split(","):
        part = chunk.strip()
        if not part:
            continue

        range_match = re.match(r"^(\d+)\s*-\s*(\d+)$", part)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            if start > end:
                start, end = end, start
            for page in range(start, end + 1):
                if page >= 1 and (max_page is None or page <= max_page):
                    pages.add(page)
            continue

        if part.isdigit():
            page = int(part)
            if page >= 1 and (max_page is None or page <= max_page):
                pages.add(page)
            continue

        raise PageRangeError(f"Invalid page range: '{part}'. Use formats like '1-5', '1 to 5' or '3'.")

    return pages


def _primary_page(question: DetectedQuestion) -> int:
    """The page a question starts on (its first/topmost segment)."""

    if not question.segments:
        return 0
    return min(seg.page for seg in question.segments)


def apply_page_ranges(
    questions: list[DetectedQuestion],
    question_pages: set[int],
    answer_pages: set[int],
    strict: bool = False,
) -> list[DetectedQuestion]:
    """Filter and relabel detected items using explicit page ranges.

    If both sets are empty, ``questions`` is returned unchanged so existing
    automatic behavior is preserved.

    ``strict`` forces "crop only the listed pages": any item whose primary page
    is in neither range is dropped, even when only one range is supplied. This
    backs the UI contract that exactly the pages the user typed get cropped (an
    answer-less PDF lists only question pages and nothing else is produced).
    """

    if not question_pages and not answer_pages:
        return questions

    # When the user fills in only ONE field, the other side is left to
    # auto-detection rather than being discarded — UNLESS ``strict`` is set.
    # Filling only "answer pages" (non-strict) means "treat these pages as
    # solutions and keep everything else as the detector found it". Pages are
    # dropped when BOTH ranges are given (an explicit, complete partition) or
    # when ``strict`` is requested.
    both_given = bool(question_pages) and bool(answer_pages)
    drop_outside = both_given or strict

    result: list[DetectedQuestion] = []
    for question in questions:
        page = _primary_page(question)

        if page in answer_pages:
            is_solution = True
        elif page in question_pages:
            is_solution = False
        elif drop_outside:
            # Page is in neither listed range -> drop it.
            continue
        else:
            # Only one range supplied (non-strict): keep the item with its
            # auto-detected label so the unspecified side still works.
            is_solution = question.is_solution

        result.append(
            DetectedQuestion(
                q_num=question.q_num,
                segments=question.segments,
                is_solution=is_solution,
            )
        )

    return result
