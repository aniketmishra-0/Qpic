"""Auto-detection orchestrator for the 3-tier pipeline."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

import fitz
from PIL import Image

from ...config import Settings
from ...models.schemas import DetectedQuestion
from .ai_detector import AIDetector
from .ocr_detector import OCRDetector
from .text_detector import TextDetector

logger = logging.getLogger(__name__)


class DetectionPipeline:
    """Tries detection methods in order and falls back when results are insufficient."""

    def __init__(
        self,
        *,
        text_detector: Optional[TextDetector] = None,
        ocr_detector: Optional[OCRDetector] = None,
        ai_detector: Optional[AIDetector] = None,
    ) -> None:
        self.text_detector = text_detector or TextDetector()
        self.ocr_detector = ocr_detector or OCRDetector()
        self.ai_detector = ai_detector

    async def detect(
        self,
        pdf_bytes: bytes,
        page_images: list[Image.Image],
        settings: Settings,
        *,
        render_dpi: Optional[int] = None,
        smart: bool = False,
        prefer_ai: bool = False,
        marker_style: str = "auto",
    ) -> tuple[list[DetectedQuestion], str]:
        """Return (questions, method_used).

        method_used: "text" | "ocr" | "ai"

        When ``prefer_ai`` is True (the user explicitly turned the online/AI
        toggle on) and a vision detector is configured, the AI tier runs *first*
        as the primary detector. Its result is returned whenever it finds
        anything; only when AI yields nothing (rate-limited, unparseable, no
        key) does the pipeline fall back to the cheap text/OCR tiers. This is
        what makes the toggle actually change the output — otherwise the cheap
        tiers satisfy the "sufficient" gate on a normal paper and short-circuit
        before AI is ever called.

        When ``smart`` is True the pipeline runs the cheap tiers first but only
        accepts their result if it looks *confident* (see
        :meth:`_result_is_confident`). Otherwise — odd layouts, broken
        numbering, sparse hits — it escalates to the AI vision tier whenever one
        is configured, so genuinely "any PDF" is handled instead of returning a
        thin regex result. With ``smart`` off the original cheap-first,
        sufficient-enough behaviour is preserved exactly.

        ``marker_style`` (``"auto"`` | ``"q"`` | ``"numbered"``) restricts which
        numbering counts as a question across every tier, so a paper whose real
        markers are all one style doesn't pick up sub-statements / option labels
        / equation numbers as extra questions.
        """

        total_pages = len(page_images)
        if total_pages <= 0:
            return ([], "text")

        best_questions: list[DetectedQuestion] = []
        best_method: str = "text"

        searchable = self._is_searchable_pdf(pdf_bytes)
        ai_ready = self.ai_detector is not None and self.ai_detector.is_available()

        # AI-first: when the user opted into the AI tier, use vision as the
        # PRIMARY detector instead of a last-resort fallback. Run it up front and
        # return its result whenever it found anything. If it comes back empty
        # (transient API failure / unparseable response) we degrade gracefully to
        # the cheap tiers below, and disable the duplicate tier-3 AI call so we
        # don't pay for a second round-trip that already failed.
        if prefer_ai and ai_ready:
            logger.info("ai_primary tier_start=ai pages=%s", total_pages)
            ai_questions = await self.ai_detector.detect(
                page_images, settings, marker_style=marker_style
            )
            if ai_questions:
                return ai_questions, "ai"
            logger.info("ai_primary empty_result falling_back=text/ocr")
            # Treat AI as unavailable for the remainder so the cheap-tier gates
            # use the lenient "sufficient" check and tier 2.5/3 don't re-call it.
            ai_ready = False

        if searchable:
            text_questions = await asyncio.to_thread(
                self.text_detector.detect,
                pdf_bytes,
                settings.QUESTION_PADDING_PX,
                marker_style,
            )
            if len(text_questions) > len(best_questions):
                best_questions, best_method = text_questions, "text"
            # In smart mode, only short-circuit on a *confident* text result; a
            # thin or ragged one falls through so AI can do better.
            accept = (
                self._result_is_confident(text_questions, total_pages, settings)
                if (smart and ai_ready)
                else self._result_is_sufficient(text_questions, total_pages, settings)
            )
            if accept:
                return text_questions, "text"
        else:
            logger.info("pdf_not_searchable tier_start=ocr")

        # Tier 2: OCR
        ocr_questions = await asyncio.to_thread(
            self.ocr_detector.detect,
            page_images,
            settings,
            render_dpi,
            marker_style,
        )
        if len(ocr_questions) > len(best_questions):
            best_questions, best_method = ocr_questions, "ocr"
        accept_ocr = (
            self._result_is_confident(ocr_questions, total_pages, settings)
            if (smart and ai_ready)
            else self._result_is_sufficient(ocr_questions, total_pages, settings)
        )
        if accept_ocr:
            return ocr_questions, "ocr"

        # Tier 2.5: Selective AI repair of weak OCR pages.
        # When OCR produced a usable result but some pages scored low confidence
        # (a few blurry/greyish scans in an otherwise clean document), it's
        # wasteful to re-run the whole document through the AI tier. Instead we
        # send only the low-confidence pages to AI and merge their questions with
        # OCR's, keeping the AI version for any page it covers. This recovers the
        # questions OCR garbled on those pages at a fraction of the AI cost.
        if smart and ai_ready and ocr_questions:
            weak_pages = self._low_confidence_pages(settings)
            if weak_pages and len(weak_pages) < total_pages:
                merged = await self._repair_pages_with_ai(
                    page_images=page_images,
                    ocr_questions=ocr_questions,
                    weak_pages=weak_pages,
                    settings=settings,
                    marker_style=marker_style,
                )
                if merged is not None:
                    if len(merged) > len(best_questions):
                        best_questions, best_method = merged, "ocr"
                    if self._result_is_sufficient(merged, total_pages, settings):
                        return merged, "ocr"

        # Tier 3: AI
        if ai_ready:
            ai_questions = await self.ai_detector.detect(
                page_images, settings, marker_style=marker_style
            )
            if len(ai_questions) > len(best_questions):
                best_questions, best_method = ai_questions, "ai"
            if ai_questions:
                # Even if not "sufficient" by heuristic, return AI output if it produced anything.
                return ai_questions, "ai"

        return best_questions, best_method

    def _low_confidence_pages(self, settings: Settings) -> list[int]:
        """1-indexed pages whose mean OCR confidence fell below the threshold.

        Reads the per-page confidence the OCR detector recorded on its last
        ``detect`` call. Pages that produced no words (confidence 0) are excluded
        — a genuinely blank page has nothing for AI to recover and would only
        waste a call.
        """

        conf = getattr(self.ocr_detector, "page_confidence", None)
        if not conf:
            return []
        threshold = float(getattr(settings, "OCR_MIN_CONFIDENCE", 75.0))
        return [
            page
            for page, value in sorted(conf.items())
            if 0.0 < value < threshold
        ]

    async def _repair_pages_with_ai(
        self,
        *,
        page_images: list[Image.Image],
        ocr_questions: list[DetectedQuestion],
        weak_pages: list[int],
        settings: Settings,
        marker_style: str,
    ) -> Optional[list[DetectedQuestion]]:
        """Re-detect only the weak pages with AI and merge with the OCR result.

        Questions whose segments all lie on weak pages are replaced by the AI
        detections for those pages; questions touching only strong pages are kept
        from OCR untouched. Returns None on any AI failure so the caller falls
        back to the normal flow.
        """

        if self.ai_detector is None or not weak_pages:
            return None

        weak_set = set(weak_pages)
        try:
            # Render a single-page AI request per weak page so coordinates stay
            # in that page's own frame; AIDetector numbers pages from 1, so we
            # remap each returned segment back to the real page number.
            repaired: list[DetectedQuestion] = []
            for page in weak_pages:
                if page < 1 or page > len(page_images):
                    continue
                # Render off the event loop (lazy view rasterises on access).
                page_img = await asyncio.to_thread(lambda p=page: page_images[p - 1])
                single = await self.ai_detector.detect(
                    [page_img], settings, marker_style=marker_style
                )
                for q in single:
                    remapped = [
                        seg.model_copy(update={"page": page}) for seg in q.segments
                    ]
                    if remapped:
                        repaired.append(
                            DetectedQuestion(
                                q_num=q.q_num,
                                is_solution=q.is_solution,
                                segments=remapped,
                            )
                        )
        except Exception as exc:  # pragma: no cover - network/parse failures
            logger.warning("selective_ai_repair_failed pages=%s error=%s", weak_pages, str(exc))
            return None

        if not repaired:
            return None

        # Keep OCR questions that don't touch a weak page; drop those that do
        # (the AI versions replace them), then add the AI detections.
        kept = [
            q for q in ocr_questions
            if not any(seg.page in weak_set for seg in q.segments)
        ]
        merged = kept + repaired
        logger.info(
            "selective_ai_repair weak_pages=%s ocr_kept=%s ai_added=%s",
            weak_pages,
            len(kept),
            len(repaired),
        )
        return merged

    def _is_searchable_pdf(self, pdf_bytes: bytes) -> bool:
        """Return True if first 3 pages contain meaningful extractable text."""

        try:
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                total = 0
                for page_idx in range(min(3, doc.page_count)):
                    page = doc.load_page(page_idx)
                    text = (page.get_text("text") or "").strip()
                    total += len(text)
                return total > 100
        except Exception:
            return False

    def _result_is_sufficient(self, questions: list[DetectedQuestion], total_pages: int, settings: Settings) -> bool:
        """Return True if we got at least 1 question per 2 pages."""

        if total_pages <= 0:
            return False
        return (len(questions) / float(total_pages)) >= float(settings.MIN_QUESTIONS_PER_2_PAGES)

    def _result_is_confident(
        self, questions: list[DetectedQuestion], total_pages: int, settings: Settings
    ) -> bool:
        """Stricter gate used in smart mode before skipping the AI tier.

        A cheap-tier result is trusted (and AI is skipped) only when it is both
        *dense enough* and *internally consistent*. Two cheap signals catch the
        layouts where regex detection silently under-performs:

          1. **Density** — clearly more than the bare ``_result_is_sufficient``
             floor, so a paper that yielded only one or two stray matches always
             escalates to vision.
          2. **Numbering continuity** — detected question numbers should run as
             a mostly-unbroken sequence (1,2,3,…). Large gaps mean markers were
             missed (a question Claude would catch), so we escalate.

        Returning False here never drops the cheap result; it only lets the AI
        tier try to do better, and the caller keeps whichever found more.
        """

        if total_pages <= 0 or not questions:
            return False

        # 1. Density well above the minimum floor.
        floor = float(settings.MIN_QUESTIONS_PER_2_PAGES)
        density = len(questions) / float(total_pages)
        if density < max(floor, 0.75):
            return False

        # 2. Numbering continuity among the *question* items (ignore solutions,
        # which are renumbered/relabelled independently).
        nums: list[int] = []
        for q in questions:
            if q.is_solution:
                continue
            digits = re.findall(r"\d+", q.q_num)
            if digits:
                nums.append(int(digits[0]))
        if len(nums) >= 3:
            nums.sort()
            span = nums[-1] - nums[0] + 1
            # Coverage = how much of the implied 1..N run we actually found.
            coverage = len(set(nums)) / float(span) if span > 0 else 1.0
            if coverage < 0.8:
                return False

        return True
