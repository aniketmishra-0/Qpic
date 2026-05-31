"""End-to-end: a searchable PDF with an answer key ships an answer sheet."""

from __future__ import annotations

import io
import zipfile

import fitz
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def _make_pdf_with_key() -> bytes:
    """Build a 1-page searchable PDF: 4 numbered questions + an answer key."""

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4

    y = 60
    for n in range(1, 5):
        page.insert_text((50, y), f"{n}. This is question {n} - pick one.", fontsize=12)
        y += 20
        page.insert_text((70, y), "A) one   B) two   C) three   D) four", fontsize=11)
        y += 40

    # A dense, mostly-sequential answer-key grid so the parser accepts it.
    page.insert_text((50, y + 40), "Answer Key", fontsize=12)
    page.insert_text(
        (50, y + 70),
        "1. (B)   2. (A)   3. (D)   4. (C)   5. (A)   6. (B)   7. (C)",
        fontsize=12,
    )

    out = doc.tobytes()
    doc.close()
    return out


def test_crop_finds_answer_key_outside_question_pages(client: TestClient):
    """The key on a later 'answers' page must still be found (whole-doc scan)."""

    # Page 1: questions only. Page 2: the answer key (not a question page).
    doc = fitz.open()
    p1 = doc.new_page(width=595, height=842)
    y = 60
    for n in range(1, 5):
        p1.insert_text((50, y), f"{n}. Question {n} - pick one.", fontsize=12)
        y += 20
        p1.insert_text((70, y), "A) one   B) two   C) three   D) four", fontsize=11)
        y += 40
    p2 = doc.new_page(width=595, height=842)
    p2.insert_text((50, 60), "Answer Key", fontsize=12)
    p2.insert_text(
        (50, 90),
        "1. (B)   2. (A)   3. (D)   4. (C)   5. (A)   6. (B)   7. (C)",
        fontsize=12,
    )
    pdf_bytes = doc.tobytes()
    doc.close()

    files = {"file": ("paper.pdf", pdf_bytes, "application/pdf")}
    resp = client.post(
        "/api/crop",
        params={
            "has_questions": True,
            "question_pages": "1",  # key lives on page 2, NOT listed here
            "has_answers": False,
            "use_ai": False,
        },
        files=files,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["answer_sheet_included"] is True
    assert body["answers_count"] >= 1


def test_crop_includes_answer_sheet(client: TestClient):
    pdf_bytes = _make_pdf_with_key()
    files = {"file": ("paper.pdf", pdf_bytes, "application/pdf")}

    resp = client.post(
        "/api/crop",
        params={
            "has_questions": True,
            "question_pages": "1",
            "has_answers": False,
            "use_ai": False,
        },
        files=files,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    job_id = body["job_id"]
    assert body["answer_sheet_included"] is True
    assert body["answers_count"] >= 1

    dl = client.get(f"/api/crop/download/{job_id}", params={"kind": "questions"})
    assert dl.status_code == 200

    with zipfile.ZipFile(io.BytesIO(dl.content)) as zf:
        names = zf.namelist()
        assert "answers.csv" in names
        assert "answers.json" in names
        csv_text = zf.read("answers.csv").decode("utf-8")

    # Q1 -> B, as listed in the key.
    assert "Q001.png,1,B" in csv_text.replace("\r\n", "\n")


def test_crop_answer_sheet_toggle_off(client: TestClient):
    pdf_bytes = _make_pdf_with_key()
    files = {"file": ("paper.pdf", pdf_bytes, "application/pdf")}

    resp = client.post(
        "/api/crop",
        params={
            "has_questions": True,
            "question_pages": "1",
            "has_answers": False,
            "use_ai": False,
            "answer_sheet": False,
        },
        files=files,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["answer_sheet_included"] is False

    dl = client.get(f"/api/crop/download/{body['job_id']}", params={"kind": "questions"})
    with zipfile.ZipFile(io.BytesIO(dl.content)) as zf:
        names = zf.namelist()
    assert "answers.csv" not in names
    assert "answers.json" not in names
