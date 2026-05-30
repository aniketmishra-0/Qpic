"""Tests for OCR per-page confidence (drives selective AI escalation)."""

from __future__ import annotations

from app.services.detector.ocr_detector import OCRDetector


def test_mean_confidence_ignores_structural_and_empty() -> None:
    data = {
        "text": ["", "Hello", "world", "  ", "5"],
        "conf": [-1, "90", "80", -1, "70"],
    }
    # Only real words 90, 80, 70 count -> mean 80.
    assert OCRDetector._mean_confidence(data) == 80.0


def test_mean_confidence_blank_page_is_zero() -> None:
    data = {"text": ["", "  "], "conf": [-1, -1]}
    assert OCRDetector._mean_confidence(data) == 0.0


def test_mean_confidence_handles_missing_fields() -> None:
    assert OCRDetector._mean_confidence({}) == 0.0
