from __future__ import annotations

import asyncio

import fitz
from PIL import Image

from app.config import Settings
from app.models.schemas import DetectedQuestion, QuestionSegment
from app.services.detector.pipeline import DetectionPipeline


def _make_text_pdf_bytes() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "1. " + ("A" * 160), fontsize=12)
    page.insert_text((72, 120), "2. " + ("B" * 160), fontsize=12)
    data = doc.tobytes()
    doc.close()
    return data


def _make_blank_pdf_bytes(pages: int = 1) -> bytes:
    doc = fitz.open()
    for _ in range(pages):
        doc.new_page()
    data = doc.tobytes()
    doc.close()
    return data


def test_uses_text_for_searchable_pdf() -> None:
    pdf_bytes = _make_text_pdf_bytes()
    page_images = [Image.new("RGB", (100, 100), (255, 255, 255))]
    settings = Settings(ANTHROPIC_API_KEY=None)

    pipeline = DetectionPipeline()
    questions, method_used = asyncio.run(pipeline.detect(pdf_bytes, page_images, settings))

    assert method_used == "text"
    assert len(questions) >= 1


def test_falls_back_to_ocr_for_scanned() -> None:
    class DummyTextDetector:
        def detect(self, pdf_bytes: bytes, padding_px: int = 0, marker_style: str = "auto"):
            return []

    class DummyOCRDetector:
        def detect(self, page_images, settings: Settings, render_dpi=None, marker_style: str = "auto"):
            return [
                DetectedQuestion(
                    q_num="1",
                    segments=[QuestionSegment(page=1, y_start_pct=10.0, y_end_pct=50.0)],
                )
            ]

    pdf_bytes = _make_blank_pdf_bytes(pages=2)
    page_images = [Image.new("RGB", (100, 100), (255, 255, 255)) for _ in range(2)]
    settings = Settings(ANTHROPIC_API_KEY=None)

    pipeline = DetectionPipeline(text_detector=DummyTextDetector(), ocr_detector=DummyOCRDetector())
    questions, method_used = asyncio.run(pipeline.detect(pdf_bytes, page_images, settings))

    assert method_used == "ocr"
    assert len(questions) == 1


def test_result_sufficient_logic() -> None:
    settings = Settings(MIN_QUESTIONS_PER_2_PAGES=0.5)
    pipeline = DetectionPipeline()

    assert pipeline._result_is_sufficient([], total_pages=2, settings=settings) is False
    assert pipeline._result_is_sufficient(
        [DetectedQuestion(q_num="1", segments=[QuestionSegment(page=1, y_start_pct=0.0, y_end_pct=10.0)])],
        total_pages=4,
        settings=settings,
    ) is False
    assert pipeline._result_is_sufficient(
        [
            DetectedQuestion(q_num="1", segments=[QuestionSegment(page=1, y_start_pct=0.0, y_end_pct=10.0)]),
            DetectedQuestion(q_num="2", segments=[QuestionSegment(page=2, y_start_pct=0.0, y_end_pct=10.0)]),
        ],
        total_pages=4,
        settings=settings,
    ) is True


class _FakeAIDetector:
    """Stand-in vision detector that records whether it was called."""

    def __init__(self, questions, available: bool = True):
        self._questions = questions
        self._available = available
        self.calls = 0

    def is_available(self) -> bool:
        return self._available

    async def detect(self, page_images, settings, *, marker_style: str = "auto"):
        self.calls += 1
        return list(self._questions)


def _ai_question(q_num: str = "1") -> DetectedQuestion:
    return DetectedQuestion(
        q_num=q_num,
        segments=[QuestionSegment(page=1, y_start_pct=5.0, y_end_pct=40.0)],
    )


def test_prefer_ai_runs_ai_first_on_searchable_pdf() -> None:
    """With the AI toggle on, vision is the primary tier even when text would
    have been 'sufficient' — so toggling AI actually changes the output."""

    pdf_bytes = _make_text_pdf_bytes()
    page_images = [Image.new("RGB", (100, 100), (255, 255, 255))]
    settings = Settings(ANTHROPIC_API_KEY=None)

    ai = _FakeAIDetector([_ai_question("7")])
    pipeline = DetectionPipeline(ai_detector=ai)
    questions, method_used = asyncio.run(
        pipeline.detect(pdf_bytes, page_images, settings, prefer_ai=True)
    )

    assert ai.calls == 1
    assert method_used == "ai"
    assert [q.q_num for q in questions] == ["7"]


def test_prefer_ai_falls_back_when_ai_empty() -> None:
    """If AI returns nothing, the pipeline degrades to the cheap tiers instead
    of failing, and does not re-call AI a second time."""

    pdf_bytes = _make_text_pdf_bytes()
    page_images = [Image.new("RGB", (100, 100), (255, 255, 255))]
    settings = Settings(ANTHROPIC_API_KEY=None)

    ai = _FakeAIDetector([])  # AI yields nothing (e.g. transient failure)
    pipeline = DetectionPipeline(ai_detector=ai)
    questions, method_used = asyncio.run(
        pipeline.detect(pdf_bytes, page_images, settings, prefer_ai=True)
    )

    assert ai.calls == 1  # tried once, not re-called as tier 3
    assert method_used == "text"
    assert len(questions) >= 1


def test_no_prefer_ai_keeps_cheap_first_behaviour() -> None:
    """Default (toggle off): AI is never called when the cheap tiers suffice."""

    pdf_bytes = _make_text_pdf_bytes()
    page_images = [Image.new("RGB", (100, 100), (255, 255, 255))]
    settings = Settings(ANTHROPIC_API_KEY=None)

    ai = _FakeAIDetector([_ai_question("7")])
    pipeline = DetectionPipeline(ai_detector=ai)
    questions, method_used = asyncio.run(
        pipeline.detect(pdf_bytes, page_images, settings, prefer_ai=False)
    )

    assert ai.calls == 0
    assert method_used == "text"
