import fitz
import pytest
from fastapi import HTTPException

from app.config import Settings
from app.services.pdf_service import pdf_to_images, validate_pdf


def _make_pdf_bytes(page_count: int = 1) -> bytes:
    doc = fitz.open()
    for i in range(page_count):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1}")
    data = doc.tobytes()
    doc.close()
    return data


def test_validate_pdf_valid() -> None:
    pdf_bytes = _make_pdf_bytes(page_count=2)
    settings = Settings(MAX_PAGES=10, MAX_PDF_SIZE_MB=5)
    validate_pdf(pdf_bytes, settings)


def test_validate_pdf_too_large() -> None:
    pdf_bytes = _make_pdf_bytes(page_count=1)
    settings = Settings(MAX_PDF_SIZE_MB=0)
    with pytest.raises(HTTPException) as exc:
        validate_pdf(pdf_bytes, settings)
    assert exc.value.status_code == 413


def test_validate_pdf_not_pdf() -> None:
    settings = Settings()
    with pytest.raises(HTTPException) as exc:
        validate_pdf(b"not a pdf", settings)
    assert exc.value.status_code == 400


def test_pdf_to_images_returns_correct_count() -> None:
    pdf_bytes = _make_pdf_bytes(page_count=3)
    images = pdf_to_images(pdf_bytes, dpi=100)
    assert len(images) == 3
