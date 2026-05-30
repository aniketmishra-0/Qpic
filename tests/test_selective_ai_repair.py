"""Tests for selective AI repair of low-confidence OCR pages."""

from __future__ import annotations

import asyncio

from app.config import Settings
from app.models.schemas import DetectedQuestion, QuestionSegment
from app.services.detector.pipeline import DetectionPipeline


class _StubOCR:
    """Stand-in OCR detector exposing a fixed per-page confidence map."""

    def __init__(self, confidence):
        self.page_confidence = confidence


class _StubAI:
    """Stand-in AI detector that 'recovers' one question per requested page."""

    def __init__(self):
        self.calls = []

    def is_available(self):
        return True

    async def detect(self, page_images, settings, *, marker_style="auto"):
        # Single-page request -> return a question anchored to page 1 of the
        # request frame (the pipeline remaps it to the real page number).
        self.calls.append(len(page_images))
        return [
            DetectedQuestion(
                q_num="99",
                segments=[QuestionSegment(page=1, y_start_pct=10, y_end_pct=30)],
            )
        ]


def test_low_confidence_pages_selected() -> None:
    pipe = DetectionPipeline(ocr_detector=_StubOCR({1: 92.0, 2: 40.0, 3: 88.0}))
    settings = Settings(OCR_MIN_CONFIDENCE=75.0)
    assert pipe._low_confidence_pages(settings) == [2]


def test_blank_pages_excluded_from_weak_set() -> None:
    pipe = DetectionPipeline(ocr_detector=_StubOCR({1: 0.0, 2: 50.0}))
    settings = Settings(OCR_MIN_CONFIDENCE=75.0)
    # Page 1 (confidence 0 = blank) is excluded; only page 2 is weak.
    assert pipe._low_confidence_pages(settings) == [2]


def test_repair_replaces_weak_page_questions() -> None:
    pipe = DetectionPipeline(ocr_detector=_StubOCR({1: 90.0, 2: 40.0}))
    pipe.ai_detector = _StubAI()
    settings = Settings()

    # OCR found Q1 on the strong page and a (garbled) Q2 on the weak page.
    ocr_questions = [
        DetectedQuestion(q_num="1", segments=[QuestionSegment(page=1, y_start_pct=5, y_end_pct=20)]),
        DetectedQuestion(q_num="2", segments=[QuestionSegment(page=2, y_start_pct=5, y_end_pct=20)]),
    ]
    page_images = [object(), object()]  # not actually rendered by the stub

    merged = asyncio.run(
        pipe._repair_pages_with_ai(
            page_images=page_images,
            ocr_questions=ocr_questions,
            weak_pages=[2],
            settings=settings,
            marker_style="auto",
        )
    )
    assert merged is not None
    pages = {seg.page for q in merged for seg in q.segments}
    # The kept OCR Q1 (page 1) plus the AI-recovered question remapped to page 2.
    assert pages == {1, 2}
    # The weak-page OCR question was replaced (not duplicated): exactly one item
    # touches page 2.
    page2_items = [q for q in merged if any(s.page == 2 for s in q.segments)]
    assert len(page2_items) == 1
