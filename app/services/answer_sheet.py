"""Answer-sheet export: pair each cropped question image with its answer.

The detection pipeline already learns the paper's answer key (question number →
correct option), but until now that knowledge was used only to flag missed
questions and then thrown away. This module turns it into a deliverable: a small
``answers.csv`` + ``answers.json`` written next to the crops so the download
ships an answer sheet keyed by the exact image filenames the user receives.

Why both formats: CSV opens straight into Excel/Sheets for a teacher building a
test, while JSON is the machine-readable form for importing into a quiz/Anki
pipeline.

Nothing here calls the network or the model — it only formats a key the caller
already obtained (from the PDF text, or from the AI vision reader). When no key
is available it writes nothing, so a paper without an answer key is unaffected.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
from pathlib import Path
from typing import Iterable

from ..models.schemas import DetectedQuestion

logger = logging.getLogger(__name__)

CSV_NAME = "answers.csv"
JSON_NAME = "answers.json"


def _q_number(q_num: str) -> int | None:
    digits = re.findall(r"\d+", q_num or "")
    return int(digits[0]) if digits else None


def _output_filename(
    q_num: str,
    *,
    is_solution: bool,
    question_prefix: str,
    solution_prefix: str,
    start_number: int,
    image_format: str,
) -> str:
    """Reproduce the on-disk crop filename (see ``save_question_image``).

    Kept in lock-step with the cropper's naming so the answer sheet references
    the exact files in the ZIP: ``<prefix><number>.<ext>`` zero-padded to three
    digits, with ``start_number`` shifting the detected number.
    """

    detected_number = _q_number(q_num) or 0
    number = detected_number + (start_number - 1)
    if number < 0:
        number = 0
    prefix = solution_prefix if is_solution else question_prefix
    ext = "jpg" if (image_format or "png").strip().lower() in ("jpg", "jpeg") else "png"
    return f"{prefix}{number:03d}.{ext}"


def build_answer_rows(
    detected: Iterable[DetectedQuestion],
    answer_key: dict[int, str],
    *,
    question_prefix: str = "Q",
    solution_prefix: str = "S",
    start_number: int = 1,
    image_format: str = "png",
) -> list[dict[str, str]]:
    """Return one row per cropped *question* with its filename and answer.

    Solutions are skipped (the key answers questions, not solution write-ups).
    A question with no entry in the key gets an empty ``answer`` so the sheet
    still lists every image — making a missing answer obvious rather than hidden.
    Rows are ordered by question number for a tidy sheet.
    """

    rows: list[tuple[int, dict[str, str]]] = []
    seen: set[int] = set()

    for q in detected:
        if q.is_solution:
            continue
        num = _q_number(q.q_num)
        if num is None or num in seen:
            continue
        seen.add(num)

        filename = _output_filename(
            q.q_num,
            is_solution=False,
            question_prefix=question_prefix,
            solution_prefix=solution_prefix,
            start_number=start_number,
            image_format=image_format,
        )
        answer = (answer_key.get(num) or "").upper()
        rows.append(
            (num, {"file": filename, "question": str(num), "answer": answer})
        )

    rows.sort(key=lambda r: r[0])
    return [row for _, row in rows]


def _csv_text(rows: list[dict[str, str]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["file", "question", "answer"])
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def count_answered(
    detected: Iterable[DetectedQuestion],
    answer_key: dict[int, str],
    *,
    start_number: int = 1,
) -> int:
    """Return how many detected questions the key supplies an answer for.

    Mirrors the row-building logic so the API can report the answer count
    without re-reading the written files. ``start_number`` is accepted for
    signature symmetry; it doesn't affect which questions are answered.
    """

    if not answer_key:
        return 0
    answered = 0
    seen: set[int] = set()
    for q in detected:
        if q.is_solution:
            continue
        num = _q_number(q.q_num)
        if num is None or num in seen:
            continue
        seen.add(num)
        if answer_key.get(num):
            answered += 1
    return answered


def write_answer_sheet(
    detected: Iterable[DetectedQuestion],
    answer_key: dict[int, str],
    output_dir: Path,
    *,
    question_prefix: str = "Q",
    solution_prefix: str = "S",
    start_number: int = 1,
    image_format: str = "png",
    always: bool = False,
    empty_reason: str = "",
) -> list[Path]:
    """Write ``answers.csv`` + ``answers.json`` into ``output_dir``.

    Returns the paths written (empty list when there's nothing to write).

    By default a sheet is only written when the answer key actually matched at
    least one detected question. When ``always`` is True the sheet is written
    even with no answers found — every question is listed with a blank answer and
    a ``answers_README.txt`` note explains *why* the answers are blank
    (``empty_reason``). This makes the feature's result visible in the ZIP
    instead of silently producing nothing when no key was detected.
    """

    rows = build_answer_rows(
        detected,
        answer_key,
        question_prefix=question_prefix,
        solution_prefix=solution_prefix,
        start_number=start_number,
        image_format=image_format,
    )
    if not rows:
        return []

    has_answers = any(r["answer"] for r in rows)

    # Default behaviour: only ship a sheet that actually carries answers.
    if not has_answers and not always:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    csv_path = output_dir / CSV_NAME
    csv_path.write_text(_csv_text(rows), encoding="utf-8")
    written.append(csv_path)

    json_path = output_dir / JSON_NAME
    json_payload = {
        "count": len(rows),
        "answered": sum(1 for r in rows if r["answer"]),
        "answers": [
            {"file": r["file"], "question": int(r["question"]), "answer": r["answer"]}
            for r in rows
        ],
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    written.append(json_path)

    # When no answers were found, drop a plain-text note explaining why so the
    # user isn't left guessing at an all-blank sheet.
    if not has_answers:
        note = (
            "No answer key was detected in this PDF, so the 'answer' column is "
            "blank.\n\n"
            "Why this happens:\n"
            "  1. This PDF has no answer-key section in it (the key may be in a "
            "separate file).\n"
            "  2. The PDF is a scan/photo (no selectable text). Turn ON 'Online "
            "mode (AI)' and configure an AI key so the key is read from the page "
            "images.\n"
            "  3. The answer key is laid out in an unusual way the parser didn't "
            "recognise.\n\n"
            + (f"Detail: {empty_reason}\n" if empty_reason else "")
        )
        note_path = output_dir / "answers_README.txt"
        note_path.write_text(note, encoding="utf-8")
        written.append(note_path)

    logger.info(
        "answer_sheet_written questions=%s answered=%s always=%s dir=%s",
        len(rows),
        json_payload["answered"],
        always,
        output_dir.name,
    )
    return written
