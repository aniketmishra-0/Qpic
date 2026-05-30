from __future__ import annotations

from typing import Any

import pytest
from PIL import Image

from app.config import Settings
from app.services.detector.ocr_detector import OCRDetector


def test_is_available_returns_bool() -> None:
    detector = OCRDetector()
    available = detector._is_available()
    assert isinstance(available, bool)


def test_preprocess_returns_grayscale() -> None:
    detector = OCRDetector()
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    out = detector._preprocess_for_ocr(img, render_dpi=200)
    assert out.mode == "L"


def test_detects_on_clear_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = OCRDetector()

    # Avoid requiring a real tesseract binary in test environments.
    monkeypatch.setattr(detector, "_is_available", lambda: True)

    def fake_image_to_data(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "text": ["1.", "Question"],
            "top": [10, 10],
            "left": [10, 60],
            "conf": ["95", "95"],
        }

    import pytesseract

    monkeypatch.setattr(pytesseract, "image_to_data", fake_image_to_data)

    img = Image.new("RGB", (800, 600), (255, 255, 255))
    settings = Settings()
    questions = detector.detect([img], settings)

    assert len(questions) == 1
    assert questions[0].q_num == "1"
    assert len(questions[0].segments) == 1
