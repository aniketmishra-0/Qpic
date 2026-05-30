import os

import pytest
import httpx

os.environ["ANTHROPIC_API_KEY"] = ""

from app.main import app  # noqa: E402


@pytest.mark.asyncio
async def test_health_endpoint() -> None:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert "tesseract_available" in data
            assert "ai_available" in data
            assert "version" in data
            assert isinstance(data["tesseract_available"], bool)
            assert isinstance(data["ai_available"], bool)


@pytest.mark.asyncio
async def test_crop_endpoint_no_file() -> None:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/crop")
            assert resp.status_code == 422


@pytest.mark.asyncio
async def test_crop_endpoint_wrong_file_type() -> None:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            files = {"file": ("test.txt", b"hello", "text/plain")}
            # question_pages is required; provide it so the request reaches the
            # content-type check rather than failing query validation.
            resp = await client.post("/api/crop?question_pages=1-5&has_answers=false", files=files)
            assert resp.status_code == 400
            assert "detail" in resp.json()


@pytest.mark.asyncio
async def test_crop_endpoint_requires_question_pages() -> None:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            files = {"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")}
            resp = await client.post("/api/crop", files=files)
            # has_questions defaults to true, so omitting question_pages is a 400.
            assert resp.status_code == 400
            assert "Question" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_crop_endpoint_rejects_nothing_selected() -> None:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            files = {"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")}
            # Both toggles off -> nothing to crop -> 400.
            resp = await client.post(
                "/api/crop?has_questions=false&has_answers=false", files=files
            )
            assert resp.status_code == 400
            assert "Nothing to crop" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_crop_endpoint_requires_answer_pages_when_solutions_on() -> None:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            files = {"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")}
            # has_answers defaults to true, so omitting answer_pages is a 400.
            resp = await client.post("/api/crop?question_pages=1-5", files=files)
            assert resp.status_code == 400
            assert "Answer" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_download_nonexistent_job() -> None:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/crop/download/doesnotexist")
            assert resp.status_code == 404
