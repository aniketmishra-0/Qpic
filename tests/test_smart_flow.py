"""Tests for the smart analyze -> review -> finalize flow."""

from __future__ import annotations

import os

import fitz
import httpx
import pytest

os.environ["ANTHROPIC_API_KEY"] = ""

from app.main import app  # noqa: E402
from app.services.review_service import build_analyzed_items, build_review_notes  # noqa: E402
from app.models.schemas import DetectedQuestion, QuestionSegment  # noqa: E402


def _make_pdf_bytes() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "1. " + ("A" * 120), fontsize=12)
    page.insert_text((72, 200), "2. " + ("B" * 120), fontsize=12)
    page.insert_text((72, 330), "3. " + ("C" * 120), fontsize=12)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.mark.asyncio
async def test_analyze_returns_items_and_pages() -> None:
    pdf = _make_pdf_bytes()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            files = {"file": ("t.pdf", pdf, "application/pdf")}
            r = await client.post("/api/analyze?dpi=120", files=files)
            assert r.status_code == 200, r.text
            data = r.json()
            assert data["total_pages"] == 1
            assert len(data["items"]) >= 3
            assert len(data["pages"]) == 1
            assert data["pages"][0]["preview_url"].endswith("/page/1")

            # Preview image is served.
            pr = await client.get(data["pages"][0]["preview_url"])
            assert pr.status_code == 200
            assert pr.headers["content-type"] == "image/png"


@pytest.mark.asyncio
async def test_analyze_then_finalize_combines_manual_item() -> None:
    pdf = _make_pdf_bytes()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            files = {"file": ("t.pdf", pdf, "application/pdf")}
            r = await client.post("/api/analyze?dpi=120", files=files)
            data = r.json()
            job_id = data["job_id"]

            items = [
                {"q_num": it["q_num"], "is_solution": it["is_solution"], "segments": it["segments"]}
                for it in data["items"]
            ]
            # Add a manual box for a "missed" question.
            items.append(
                {
                    "q_num": "9",
                    "is_solution": False,
                    "segments": [
                        {"page": 1, "x_start_pct": 5, "x_end_pct": 95, "y_start_pct": 60, "y_end_pct": 75}
                    ],
                }
            )
            body = {"job_id": job_id, "items": items, "dpi": 120, "image_format": "png"}
            fr = await client.post("/api/finalize", json=body)
            assert fr.status_code == 200, fr.text
            fd = fr.json()
            assert fd["total_questions"] == len(items)

            dl = await client.get(fd["download_url"])
            assert dl.status_code == 200
            assert len(dl.content) > 0


@pytest.mark.asyncio
async def test_finalize_produces_question_solution_and_combined_zips() -> None:
    pdf = _make_pdf_bytes()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            files = {"file": ("t.pdf", pdf, "application/pdf")}
            r = await client.post("/api/analyze?dpi=120", files=files)
            data = r.json()
            job_id = data["job_id"]

            # Two questions and one solution -> all three archives expected.
            items = [
                {
                    "q_num": "1",
                    "is_solution": False,
                    "segments": [{"page": 1, "x_start_pct": 5, "x_end_pct": 95, "y_start_pct": 5, "y_end_pct": 20}],
                },
                {
                    "q_num": "2",
                    "is_solution": False,
                    "segments": [{"page": 1, "x_start_pct": 5, "x_end_pct": 95, "y_start_pct": 25, "y_end_pct": 40}],
                },
                {
                    "q_num": "1",
                    "is_solution": True,
                    "segments": [{"page": 1, "x_start_pct": 5, "x_end_pct": 95, "y_start_pct": 60, "y_end_pct": 75}],
                },
            ]
            body = {
                "job_id": job_id,
                "items": items,
                "dpi": 120,
                "image_format": "png",
                "question_prefix": "Q",
                "solution_prefix": "S",
            }
            fr = await client.post("/api/finalize", json=body)
            assert fr.status_code == 200, fr.text
            fd = fr.json()
            assert fd["questions_count"] == 2
            assert fd["solutions_count"] == 1
            assert fd["questions_download_url"] and "kind=questions" in fd["questions_download_url"]
            assert fd["solutions_download_url"] and "kind=solutions" in fd["solutions_download_url"]

            # Combined archive (default kind) downloads with the QScombined name.
            dl = await client.get(fd["download_url"])
            assert dl.status_code == 200 and len(dl.content) > 0
            assert "QScombined.zip" in dl.headers.get("content-disposition", "")

            # Questions-only archive.
            qd = await client.get(fd["questions_download_url"] + "&question_prefix=Q&solution_prefix=S")
            assert qd.status_code == 200 and len(qd.content) > 0
            assert "Q.zip" in qd.headers.get("content-disposition", "")

            # Solutions-only archive.
            sd = await client.get(fd["solutions_download_url"] + "&question_prefix=Q&solution_prefix=S")
            assert sd.status_code == 200 and len(sd.content) > 0
            assert "S.zip" in sd.headers.get("content-disposition", "")


@pytest.mark.asyncio
async def test_finalize_unknown_job_404() -> None:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            body = {
                "job_id": "does-not-exist",
                "items": [
                    {"q_num": "1", "is_solution": False, "segments": [{"page": 1, "y_start_pct": 0, "y_end_pct": 10}]}
                ],
            }
            fr = await client.post("/api/finalize", json=body)
            assert fr.status_code == 404


@pytest.mark.asyncio
async def test_snap_tightens_loose_box_to_content() -> None:
    pdf = _make_pdf_bytes()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            files = {"file": ("t.pdf", pdf, "application/pdf")}
            r = await client.post("/api/analyze?dpi=120", files=files)
            job_id = r.json()["job_id"]

            # A loose box covering the whole page should snap inward to the text.
            body = {
                "job_id": job_id,
                "page": 1,
                "x_start_pct": 0.0,
                "x_end_pct": 100.0,
                "y_start_pct": 0.0,
                "y_end_pct": 100.0,
            }
            sr = await client.post("/api/snap", json=body)
            assert sr.status_code == 200
            s = sr.json()
            # Tightened region must be inside the original and non-empty.
            assert s["x_start_pct"] >= 0.0 and s["x_end_pct"] <= 100.0
            assert s["y_start_pct"] >= 0.0 and s["y_end_pct"] <= 100.0
            assert s["y_end_pct"] - s["y_start_pct"] < 100.0
            assert s["x_end_pct"] - s["x_start_pct"] <= 100.0


@pytest.mark.asyncio
async def test_snap_unknown_job_echoes_box() -> None:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            body = {
                "job_id": "nope",
                "page": 1,
                "x_start_pct": 10.0,
                "x_end_pct": 90.0,
                "y_start_pct": 20.0,
                "y_end_pct": 80.0,
            }
            sr = await client.post("/api/snap", json=body)
            assert sr.status_code == 200
            s = sr.json()
            assert s["x_start_pct"] == 10.0 and s["y_end_pct"] == 80.0


def test_review_notes_flag_duplicate_and_gap() -> None:
    detected = [
        DetectedQuestion(q_num="1", segments=[QuestionSegment(page=1, y_start_pct=0, y_end_pct=20)]),
        DetectedQuestion(q_num="2", segments=[QuestionSegment(page=1, y_start_pct=20, y_end_pct=40)]),
        # 3 is missing -> gap
        DetectedQuestion(q_num="4", segments=[QuestionSegment(page=1, y_start_pct=40, y_end_pct=60)]),
        # duplicate of 4
        DetectedQuestion(q_num="4", segments=[QuestionSegment(page=2, y_start_pct=0, y_end_pct=20)]),
    ]
    notes = build_review_notes(detected, "text")
    kinds = {n.kind for n in notes}
    assert "gap" in kinds
    assert "duplicate" in kinds


def test_build_analyzed_items_flags_tiny_crop() -> None:
    detected = [
        DetectedQuestion(q_num="1", segments=[QuestionSegment(page=1, y_start_pct=10.0, y_end_pct=11.0)]),
    ]
    items = build_analyzed_items(detected)
    assert items[0].flagged is True


def test_cutoff_crop_at_page_bottom_is_flagged() -> None:
    # A short crop that stops right at the page bottom should be flagged as
    # likely cut off (continues onto the next page).
    detected = [
        DetectedQuestion(q_num="1", segments=[QuestionSegment(page=1, y_start_pct=80.0, y_end_pct=99.0)]),
        DetectedQuestion(q_num="2", segments=[QuestionSegment(page=2, y_start_pct=5.0, y_end_pct=40.0)]),
    ]
    items = build_analyzed_items(detected)
    q1 = next(it for it in items if it.q_num == "1")
    assert q1.flagged is True
    assert "page" in (q1.flag_reason or "").lower()

    notes = build_review_notes(detected, "text")
    assert any(n.kind == "incomplete" for n in notes)


def test_short_crop_vs_tall_neighbours_is_flagged() -> None:
    # One item is far shorter than its peers -> likely only half the question.
    detected = [
        DetectedQuestion(q_num="1", segments=[QuestionSegment(page=1, y_start_pct=5.0, y_end_pct=45.0)]),
        DetectedQuestion(q_num="2", segments=[QuestionSegment(page=2, y_start_pct=5.0, y_end_pct=45.0)]),
        DetectedQuestion(q_num="3", segments=[QuestionSegment(page=3, y_start_pct=5.0, y_end_pct=12.0)]),
    ]
    items = build_analyzed_items(detected)
    q3 = next(it for it in items if it.q_num == "3")
    assert q3.flagged is True


def test_tall_merged_crop_is_flagged() -> None:
    # Two normal-height items plus one that is far taller and large in absolute
    # terms -> likely two questions merged into one box.
    detected = [
        DetectedQuestion(q_num="1", segments=[QuestionSegment(page=1, y_start_pct=5.0, y_end_pct=25.0)]),
        DetectedQuestion(q_num="2", segments=[QuestionSegment(page=2, y_start_pct=5.0, y_end_pct=25.0)]),
        DetectedQuestion(q_num="3", segments=[QuestionSegment(page=3, y_start_pct=5.0, y_end_pct=75.0)]),
    ]
    items = build_analyzed_items(detected)
    q3 = next(it for it in items if it.q_num == "3")
    assert q3.flagged is True
    assert "taller" in (q3.flag_reason or "").lower()


def test_overlapping_crops_are_flagged() -> None:
    # Two crops that share a vertical strip on the same page/column overlap, even
    # though each has a perfectly normal height -> "looks fine but isn't".
    detected = [
        DetectedQuestion(q_num="1", segments=[QuestionSegment(page=1, y_start_pct=5.0, y_end_pct=40.0)]),
        DetectedQuestion(q_num="2", segments=[QuestionSegment(page=1, y_start_pct=30.0, y_end_pct=65.0)]),
    ]
    items = build_analyzed_items(detected)
    assert all(it.flagged for it in items)
    assert all("overlap" in (it.flag_reason or "").lower() for it in items)

    notes = build_review_notes(detected, "text")
    assert any(n.kind == "incomplete" and "overlap" in n.message.lower() for n in notes)


def test_overlap_ignores_different_columns() -> None:
    # Left/right questions in a 2-up layout share rows but not columns, so they
    # must not be treated as overlapping.
    detected = [
        DetectedQuestion(q_num="1", segments=[QuestionSegment(page=1, y_start_pct=5.0, y_end_pct=40.0, x_start_pct=0.0, x_end_pct=48.0)]),
        DetectedQuestion(q_num="2", segments=[QuestionSegment(page=1, y_start_pct=5.0, y_end_pct=40.0, x_start_pct=52.0, x_end_pct=100.0)]),
    ]
    items = build_analyzed_items(detected)
    assert all(not it.flagged for it in items)


def test_undercovered_crop_is_flagged() -> None:
    # S3's box (a thin strip) stops at 30% but its body text runs down to ~62%
    # with no crop covering it -> the strongest "looks fine but isn't" signal.
    # S4 starts lower, at 70%. The uncovered band (32%..62%) belongs to S3.
    detected = [
        DetectedQuestion(
            q_num="3", is_solution=True,
            segments=[QuestionSegment(page=3, y_start_pct=18.0, y_end_pct=30.0, x_start_pct=50.0, x_end_pct=100.0)],
        ),
        DetectedQuestion(
            q_num="4", is_solution=True,
            segments=[QuestionSegment(page=3, y_start_pct=70.0, y_end_pct=90.0, x_start_pct=50.0, x_end_pct=100.0)],
        ),
    ]
    # Text lines on the right column of page 3: a few inside S3's box, then a
    # tall band below it that no box covers, then S4's lines.
    page_lines = {
        3: [
            (20.0, 23.0, 52.0, 98.0),  # inside S3
            (25.0, 28.0, 52.0, 98.0),  # inside S3
            (34.0, 37.0, 52.0, 98.0),  # uncovered - S3's lost tail
            (40.0, 43.0, 52.0, 98.0),  # uncovered
            (46.0, 49.0, 52.0, 98.0),  # uncovered
            (52.0, 55.0, 52.0, 98.0),  # uncovered
            (58.0, 61.0, 52.0, 98.0),  # uncovered
            (72.0, 75.0, 52.0, 98.0),  # inside S4
        ],
    }
    items = build_analyzed_items(detected, page_lines)
    s3 = next(it for it in items if it.q_num == "3")
    assert s3.flagged is True
    assert "cover" in (s3.flag_reason or "").lower()

    notes = build_review_notes(detected, "text", None, page_lines)
    assert any(n.kind == "incomplete" and n.q_num == "3" for n in notes)


def test_fully_covered_crops_not_flagged_by_coverage() -> None:
    # Every text line sits inside a crop -> no coverage flag.
    detected = [
        DetectedQuestion(
            q_num="1", segments=[QuestionSegment(page=1, y_start_pct=10.0, y_end_pct=45.0)],
        ),
        DetectedQuestion(
            q_num="2", segments=[QuestionSegment(page=1, y_start_pct=50.0, y_end_pct=85.0)],
        ),
    ]
    page_lines = {
        1: [
            (12.0, 15.0, 5.0, 95.0),
            (40.0, 43.0, 5.0, 95.0),
            (52.0, 55.0, 5.0, 95.0),
            (80.0, 83.0, 5.0, 95.0),
        ],
    }
    items = build_analyzed_items(detected, page_lines)
    assert all(not it.flagged for it in items)


def test_uncovered_head_crop_is_flagged() -> None:
    # The cross-page-spill case from the screenshots: Q19b's page-2 crop was
    # detected starting too low (at 28%), stranding the question's opening line
    # "2. The right to health also includes access to essential medicines."
    # (~12%..20%) above it with no box over it. The below-only coverage check
    # can't see text *above* a crop, so without the head check this stays silent.
    detected = [
        DetectedQuestion(
            q_num="19",
            segments=[QuestionSegment(page=4, y_start_pct=28.0, y_end_pct=55.0, x_start_pct=0.0, x_end_pct=48.0)],
        ),
    ]
    page_lines = {
        4: [
            (12.0, 15.0, 5.0, 45.0),   # uncovered - Q19's stranded head
            (16.0, 19.0, 5.0, 45.0),   # uncovered - Q19's stranded head
            (30.0, 33.0, 5.0, 45.0),   # inside Q19's crop
            (40.0, 43.0, 5.0, 45.0),   # inside Q19's crop
        ],
    }
    items = build_analyzed_items(detected, page_lines)
    q19 = next(it for it in items if it.q_num == "19")
    assert q19.flagged is True
    assert "above" in (q19.flag_reason or "").lower()

    notes = build_review_notes(detected, "text", None, page_lines)
    assert any(n.kind == "incomplete" and n.q_num == "19" for n in notes)


def test_crop_with_real_crop_above_not_flagged_as_head() -> None:
    # When another crop sits above in the same column, the strip above belongs
    # to that crop (or the inter-crop gap), not to a lost head of this one.
    detected = [
        DetectedQuestion(
            q_num="1", segments=[QuestionSegment(page=1, y_start_pct=10.0, y_end_pct=40.0)],
        ),
        DetectedQuestion(
            q_num="2", segments=[QuestionSegment(page=1, y_start_pct=45.0, y_end_pct=85.0)],
        ),
    ]
    page_lines = {
        1: [
            (12.0, 15.0, 5.0, 95.0),  # inside Q1
            (36.0, 39.0, 5.0, 95.0),  # inside Q1
            (48.0, 51.0, 5.0, 95.0),  # inside Q2
            (80.0, 83.0, 5.0, 95.0),  # inside Q2
        ],
    }
    items = build_analyzed_items(detected, page_lines)
    assert all(not it.flagged for it in items)


def test_orphan_content_page_is_noted() -> None:
    # A whole question was missed: page 2 has a tall block of body text that no
    # crop covers and that isn't adjacent to any crop. There's no item to flag,
    # so the only signal is a page-level note.
    detected = [
        DetectedQuestion(
            q_num="1", segments=[QuestionSegment(page=1, y_start_pct=10.0, y_end_pct=80.0)],
        ),
        DetectedQuestion(
            q_num="2", segments=[QuestionSegment(page=3, y_start_pct=10.0, y_end_pct=80.0)],
        ),
    ]
    page_lines = {
        1: [(12.0, 15.0, 5.0, 95.0), (70.0, 73.0, 5.0, 95.0)],
        2: [
            (30.0, 34.0, 5.0, 95.0),  # orphan - no crop on page 2 at all
            (38.0, 42.0, 5.0, 95.0),
            (46.0, 50.0, 5.0, 95.0),
        ],
        3: [(12.0, 15.0, 5.0, 95.0), (70.0, 73.0, 5.0, 95.0)],
    }
    notes = build_review_notes(detected, "text", None, page_lines)
    assert any(
        n.kind == "incomplete" and n.page == 2 and "missed" in n.message.lower()
        for n in notes
    )

