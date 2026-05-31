"""Tier 3 detector: Anthropic Claude Vision (last resort).

Only used when Tier 1 (text) and Tier 2 (OCR) are insufficient.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Optional

import anthropic
import httpx
from anthropic import APIStatusError, RateLimitError
from fastapi import HTTPException, status
from PIL import Image

from ...config import Settings
from ...models.schemas import DetectedQuestion, QuestionSegment
from ...utils.image_utils import img_to_base64_png, resize_for_api

logger = logging.getLogger(__name__)


DETECTION_PROMPT = """
You are analyzing pages from an MCQ exam paper.
Detect every question and its exact vertical position.

Rules:
- Include question stem + all options (A/B/C/D) in one segment
- If a question spans two pages, add a segment for each page
- y_start_pct=0 is top of page, y_end_pct=100 is bottom
- Many papers have a "Solutions" / "Answer Key" / "Explanations" section after
  the questions. Items in that section are numbered just like questions. Mark
  every such item with "is_solution": true. Mark actual questions with
  "is_solution": false.

Return ONLY valid JSON:
{
  "questions": [
    {
      "q_num": "1",
      "is_solution": false,
      "segments": [
        {"page": PAGE_NUM, "y_start_pct": FLOAT, "y_end_pct": FLOAT}
      ]
    }
  ]
}
""".strip()

ERR_AI_CONFIG = "AI service configuration error"
ERR_AI_JSON = "Claude returned invalid JSON"


def _style_hint(marker_style: str) -> str:
    """Return an extra system-prompt clause for a forced numbering style.

    This stops Claude from promoting sub-statements ("Consider statements: 1. …
    2. …"), option labels ("(1) … (2) …") or equation numbers into questions in
    a paper whose real questions are all one style.
    """

    if marker_style == "q":
        return (
            "\n\nIMPORTANT: In THIS paper, a real question ALWAYS starts with an "
            "explicit 'Q' marker (e.g. 'Q1', 'Q.1', 'Q 1.', 'Question 1'). A bare "
            "leading number like '1.' or '2)' is NOT a new question — it is a "
            "sub-point, option, or equation number. Detect ONLY the Q-marked items."
        )
    if marker_style == "numbered":
        return (
            "\n\nIMPORTANT: In THIS paper, a real question starts with a bare "
            "leading number (e.g. '1.', '2)', '3.'). Treat each such top-level "
            "numbered item as one question. Do NOT split a question at its inner "
            "option labels (A/B/C/D) or sub-points."
        )
    return ""


def _batch_index(page_start: int, settings: Settings) -> int:
    step = settings.AI_BATCH_SIZE - settings.AI_BATCH_OVERLAP
    if step <= 0:
        step = settings.AI_BATCH_SIZE
    return ((page_start - 1) // step) + 1


def _strip_json_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _extract_text(message: object) -> str:
    content = getattr(message, "content", None)
    parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            else:
                if getattr(block, "type", None) == "text":
                    parts.append(str(getattr(block, "text", "")))
    return "\n".join(parts).strip()


def _coerce_questions(data: object, page_start: int, page_end: int) -> list[DetectedQuestion]:
    questions: list[DetectedQuestion] = []

    # Models don't always honour the requested wrapper shape. Accept:
    #   * {"questions": [...]}            — the documented shape
    #   * [ {...}, {...} ]                — a bare array of question objects
    #   * {"q_num": ..., "segments": ...} — a single question object
    # Anything else yields no questions instead of crashing on .get().
    if isinstance(data, list):
        raw_questions = data
    elif isinstance(data, dict):
        raw_questions = data.get("questions")
        if raw_questions is None and "segments" in data:
            raw_questions = [data]
    else:
        return []

    if not isinstance(raw_questions, list):
        return []

    for q in raw_questions:
        if not isinstance(q, dict):
            continue
        q_num = q.get("q_num")
        if not isinstance(q_num, str) or not q_num.strip():
            continue

        segments_in = q.get("segments")
        if not isinstance(segments_in, list):
            continue

        segments: list[QuestionSegment] = []
        for seg in segments_in:
            if not isinstance(seg, dict):
                continue
            try:
                page = int(seg.get("page"))
                y_start = float(seg.get("y_start_pct"))
                y_end = float(seg.get("y_end_pct"))
            except (TypeError, ValueError):
                continue

            if page < page_start or page > page_end:
                continue
            if not (0.0 <= y_start <= 100.0 and 0.0 <= y_end <= 100.0):
                continue
            if y_end <= y_start:
                continue

            segments.append(QuestionSegment(page=page, y_start_pct=y_start, y_end_pct=y_end))

        if segments:
            segments.sort(key=lambda s: (s.page, s.y_start_pct))
            questions.append(
                DetectedQuestion(
                    q_num=q_num.strip(),
                    is_solution=bool(q.get("is_solution", False)),
                    segments=segments,
                )
            )

    return questions


async def _detect_questions_in_batch(
    client: anthropic.AsyncAnthropic,
    images: list[Image.Image],
    page_start: int,
    settings: Settings,
    style_hint: str = "",
) -> list[DetectedQuestion]:
    page_end = page_start + len(images) - 1
    batch_no = _batch_index(page_start, settings)

    content_blocks: list[dict] = []
    for i, img in enumerate(images):
        page_no = page_start + i
        resized = resize_for_api(img, max_px=1568)
        b64 = img_to_base64_png(resized)
        content_blocks.append({"type": "text", "text": f"=== PAGE {page_no} ==="})
        content_blocks.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": b64},
            }
        )

    last_raw: Optional[str] = None

    for attempt in range(1, settings.AI_MAX_RETRIES + 1):
        try:
            logger.info(
                "ai_batch=%s attempt=%s page_range=%s-%s stage=claude_call",
                batch_no,
                attempt,
                page_start,
                page_end,
            )
            message = await client.messages.create(
                model=settings.CLAUDE_MODEL,
                max_tokens=4096,
                temperature=0,
                system=DETECTION_PROMPT + style_hint,
                messages=[{"role": "user", "content": content_blocks}],
            )
            raw = _extract_text(message)
            last_raw = raw

            cleaned = _strip_json_fences(raw)
            data = json.loads(cleaned)
            questions = _coerce_questions(data=data, page_start=page_start, page_end=page_end)
            logger.info(
                "ai_batch=%s page_range=%s-%s detected_questions=%s",
                batch_no,
                page_start,
                page_end,
                len(questions),
            )
            return questions
        except json.JSONDecodeError as exc:
            logger.warning("ai_batch=%s attempt=%s json_error=%s", batch_no, attempt, str(exc))
        except RateLimitError as exc:
            logger.warning("ai_batch=%s attempt=%s rate_limited=%s", batch_no, attempt, str(exc))
        except APIStatusError as exc:
            if exc.status_code in (401, 403):
                logger.error("ai_batch=%s auth_error status_code=%s", batch_no, exc.status_code)
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=ERR_AI_CONFIG) from exc
            if exc.status_code == 429:
                logger.warning("ai_batch=%s attempt=%s rate_limited_status", batch_no, attempt)
            else:
                logger.error("ai_batch=%s api_status_error=%s", batch_no, str(exc))
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

        backoff = 2 ** (attempt - 1)
        await asyncio.sleep(backoff)

    if last_raw is not None:
        logger.error("ai_batch=%s exhausted_retries last_raw=%s", batch_no, last_raw[:2000])
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=ERR_AI_JSON)


def merge_detected_questions(raw: list[DetectedQuestion]) -> list[DetectedQuestion]:
    """Merge duplicate detections from overlapping batches."""

    by_q: dict[tuple[str, bool], list[QuestionSegment]] = {}
    for q in raw:
        if not q.q_num:
            continue
        segments = by_q.setdefault((q.q_num, q.is_solution), [])
        for seg in q.segments:
            is_dup = any(
                (existing.page == seg.page and abs(existing.y_start_pct - seg.y_start_pct) < 3.0)
                for existing in segments
            )
            if not is_dup:
                segments.append(seg)

    merged: list[DetectedQuestion] = []
    for (q_num, is_solution), segments in by_q.items():
        segments_sorted = sorted(segments, key=lambda s: (s.page, s.y_start_pct))
        merged.append(
            DetectedQuestion(q_num=q_num, is_solution=is_solution, segments=segments_sorted)
        )

    def _q_num_key(q_num: str) -> tuple[int, str]:
        digits = re.findall(r"\d+", q_num)
        if not digits:
            return (10**9, q_num)
        return (int(digits[0]), q_num)

    merged.sort(key=lambda q: (q.is_solution, _q_num_key(q.q_num)))
    return merged


class AIDetector:
    def __init__(
        self,
        api_key: Optional[str],
        *,
        client: Optional[anthropic.AsyncAnthropic] = None,
        timeout_seconds: int = 120,
    ) -> None:
        self.available = api_key is not None and api_key.strip() != ""
        self._owns_client = False

        if not self.available:
            self.client = None
            return

        if client is not None:
            self.client = client
            return

        self.client = anthropic.AsyncAnthropic(
            api_key=api_key,
            timeout=httpx.Timeout(timeout_seconds),
        )
        self._owns_client = True

    def is_available(self) -> bool:
        return self.available and self.client is not None

    async def detect(
        self,
        page_images: list[Image.Image],
        settings: Settings,
        *,
        marker_style: str = "auto",
    ) -> list[DetectedQuestion]:
        if not self.is_available():
            return []

        assert self.client is not None

        if not page_images:
            return []

        style_hint = _style_hint(marker_style)

        step = settings.AI_BATCH_SIZE - settings.AI_BATCH_OVERLAP
        if step <= 0:
            step = settings.AI_BATCH_SIZE

        raw: list[DetectedQuestion] = []
        total_pages = len(page_images)

        for start_idx in range(0, total_pages, step):
            # Page access may render on demand (lazy page view), so do it off the
            # event loop to avoid blocking it while rasterising.
            batch_images = await asyncio.to_thread(
                lambda s=start_idx: list(page_images[s : s + settings.AI_BATCH_SIZE])
            )
            if not batch_images:
                break

            page_start = start_idx + 1
            batch_questions = await _detect_questions_in_batch(
                client=self.client,
                images=batch_images,
                page_start=page_start,
                settings=settings,
                style_hint=style_hint,
            )
            raw.extend(batch_questions)

        return merge_detected_questions(raw)

    async def aclose(self) -> None:
        if self._owns_client and self.client is not None:
            try:
                await self.client.aclose()
            except Exception:
                return
