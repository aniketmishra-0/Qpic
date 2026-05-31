"""Tests for the answer-sheet export."""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

from app.models.schemas import DetectedQuestion, QuestionSegment
from app.services.answer_sheet import (
    build_answer_rows,
    write_answer_sheet,
)
from app.services.zip_service import create_zip_set


def _q(num: str, is_solution: bool = False) -> DetectedQuestion:
    return DetectedQuestion(
        q_num=num,
        is_solution=is_solution,
        segments=[QuestionSegment(page=1, y_start_pct=0.0, y_end_pct=10.0)],
    )


def test_build_answer_rows_maps_filenames_and_answers():
    detected = [_q("1"), _q("2"), _q("3")]
    key = {1: "B", 2: "A", 3: "D"}

    rows = build_answer_rows(detected, key)

    assert rows == [
        {"file": "Q001.png", "question": "1", "answer": "B"},
        {"file": "Q002.png", "question": "2", "answer": "A"},
        {"file": "Q003.png", "question": "3", "answer": "D"},
    ]


def test_build_answer_rows_honours_prefix_start_and_format():
    detected = [_q("1"), _q("2")]
    key = {1: "C", 2: "D"}

    rows = build_answer_rows(
        detected,
        key,
        question_prefix="P",
        start_number=11,
        image_format="jpg",
    )

    assert rows[0]["file"] == "P011.jpg"
    assert rows[1]["file"] == "P012.jpg"


def test_build_answer_rows_skips_solutions_and_blank_unknowns():
    detected = [_q("1"), _q("2"), _q("1", is_solution=True)]
    key = {1: "A"}  # no answer for Q2

    rows = build_answer_rows(detected, key)

    # Solution item excluded; Q2 present but with empty answer.
    assert [r["question"] for r in rows] == ["1", "2"]
    assert rows[0]["answer"] == "A"
    assert rows[1]["answer"] == ""


def test_write_answer_sheet_creates_csv_and_json(tmp_path: Path):
    detected = [_q("1"), _q("2")]
    key = {1: "B", 2: "C"}

    written = write_answer_sheet(detected, key, tmp_path)

    names = {p.name for p in written}
    assert names == {"answers.csv", "answers.json"}

    payload = json.loads((tmp_path / "answers.json").read_text())
    assert payload["count"] == 2
    assert payload["answered"] == 2
    assert payload["answers"][0] == {"file": "Q001.png", "question": 1, "answer": "B"}

    with (tmp_path / "answers.csv").open() as fh:
        reader = list(csv.DictReader(fh))
    assert reader[1]["answer"] == "C"


def test_write_answer_sheet_noop_without_key(tmp_path: Path):
    assert write_answer_sheet([_q("1")], {}, tmp_path) == []
    assert not (tmp_path / "answers.csv").exists()


def test_write_answer_sheet_noop_when_no_matches(tmp_path: Path):
    # Key exists but references numbers we didn't detect -> all-blank, skip.
    assert write_answer_sheet([_q("1")], {99: "A"}, tmp_path) == []


def test_write_answer_sheet_always_writes_with_note(tmp_path: Path):
    # always=True: write the sheet even with no key, plus an explanatory note.
    written = write_answer_sheet(
        [_q("1"), _q("2")], {}, tmp_path, always=True, empty_reason="scanned PDF"
    )
    names = {p.name for p in written}
    assert names == {"answers.csv", "answers.json", "answers_README.txt"}

    note = (tmp_path / "answers_README.txt").read_text()
    assert "No answer key was detected" in note
    assert "scanned PDF" in note

    # CSV still lists every question, just with blank answers.
    with (tmp_path / "answers.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert [r["question"] for r in rows] == ["1", "2"]
    assert all(r["answer"] == "" for r in rows)


def test_write_answer_sheet_always_omits_note_when_answers_exist(tmp_path: Path):
    # always=True but answers exist -> no note file, normal sheet.
    written = write_answer_sheet([_q("1")], {1: "B"}, tmp_path, always=True)
    names = {p.name for p in written}
    assert names == {"answers.csv", "answers.json"}


def test_create_zip_set_bundles_sheet_into_questions_and_combined(tmp_path: Path):
    # Make dummy image + sheet files.
    q_img = tmp_path / "Q001.png"
    q_img.write_bytes(b"img")
    s_img = tmp_path / "S001.png"
    s_img.write_bytes(b"img")
    sheet = tmp_path / "answers.csv"
    sheet.write_text("file,question,answer\n")

    zips = create_zip_set([q_img], [s_img], "job1", tmp_path, [sheet])

    with zipfile.ZipFile(zips["questions"]) as zf:
        assert "answers.csv" in zf.namelist()
    with zipfile.ZipFile(zips["combined"]) as zf:
        assert "answers.csv" in zf.namelist()
    # Solutions-only archive must NOT carry the question-keyed sheet.
    with zipfile.ZipFile(zips["solutions"]) as zf:
        assert "answers.csv" not in zf.namelist()
