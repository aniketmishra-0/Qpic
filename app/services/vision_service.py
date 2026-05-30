"""Claude Vision question detection and merging."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Awaitable, Callable, Optional

import anthropic
from anthropic import APIStatusError, RateLimitError
from fastapi import HTTPException, status
from PIL import Image

from ..config import Settings
from ..models.schemas import DetectedQuestion, QuestionSegment
from ..utils.image_utils import img_to_base64_png, resize_for_api

logger = logging.getLogger(__name__)


DETECTION_PROMPT = """
You are analyzing pages from an MCQ exam paper.

Detect every individual MCQ question and its exact vertical position on each page.

CRITICAL RULES:
1. A question includes its stem (the question text) AND all its options (A, B, C, D or 1,2,3,4).
2. If a question starts near the bottom of one page and its options continue on the next page — report BOTH pages as segments for that question.
3. y_start_pct = 0 means absolute top of the page. y_end_pct = 100 means absolute bottom.
4. Be precise to within ±2% of the actual position.
5. Do NOT merge two different questions into one.
6. Do NOT split one question into two.
7. q_num must be the EXACT number printed in the paper (e.g., "1", "Q1", "12").

Return ONLY valid JSON. No explanation, no markdown, no preamble:
{
  "questions": [
    {
      "q_num": "1",
      "segments": [
        {"page": PAGE_NUMBER, "y_start_pct": FLOAT, "y_end_pct": FLOAT}
      ]
    }
  ]
}
""".strip()


ERR_AI_CONFIG = "AI service configuration error"
ERR_AI_JSON = "Claude returned invalid JSON"


def _batch_index(page_start: int, settings: Settings) -> int:
    step = settings.AI_BATCH_SIZE - settings.AI_BATCH_OVERLAP
    if step <= 0:
        step = settings.AI_BATCH_SIZE
    return ((page_start - 1) // step) + 1


def _strip_json_fences(text: str) -> str:
    text = text.strip()
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

    # Accept the documented {"questions": [...]} shape plus the bare-array and
    # single-object shapes some models emit, instead of crashing on .get().
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
            questions.append(DetectedQuestion(q_num=q_num.strip(), segments=segments))

    return questions


async def detect_questions_in_batch(
    client: anthropic.AsyncAnthropic,
    images: list[Image.Image],
    page_start: int,
    settings: Settings,
    retries: int = 3,
) -> list[DetectedQuestion]:
    """Send a batch of page images to Claude Vision and parse detections."""

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
    for attempt in range(1, retries + 1):
        try:
            logger.info(
                "batch=%s attempt=%s page_range=%s-%s stage=claude_call",
                batch_no,
                attempt,
                page_start,
                page_end,
            )
            message = await client.messages.create(
                model=settings.CLAUDE_MODEL,
                max_tokens=4096,
                temperature=0,
                system=DETECTION_PROMPT,
                messages=[{"role": "user", "content": content_blocks}],
            )
            raw = _extract_text(message)
            last_raw = raw

            cleaned = _strip_json_fences(raw)
            data = json.loads(cleaned)
            questions = _coerce_questions(data=data, page_start=page_start, page_end=page_end)
            logger.info(
                "batch=%s page_range=%s-%s detected_questions=%s",
                batch_no,
                page_start,
                page_end,
                len(questions),
            )
            return questions
        except json.JSONDecodeError as exc:
            logger.warning(
                "batch=%s attempt=%s json_parse_error=%s",
                batch_no,
                attempt,
                str(exc),
            )
        except RateLimitError as exc:
            logger.warning(
                "batch=%s attempt=%s rate_limited=%s",
                batch_no,
                attempt,
                str(exc),
            )
        except APIStatusError as exc:
            if exc.status_code in (401, 403):
                logger.error("batch=%s auth_error status_code=%s", batch_no, exc.status_code)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=ERR_AI_CONFIG,
                ) from exc
            if exc.status_code == 429:
                logger.warning("batch=%s attempt=%s rate_limited_status", batch_no, attempt)
            else:
                logger.error("batch=%s api_status_error=%s", batch_no, str(exc))
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=str(exc),
                ) from exc
        except Exception as exc:
            logger.error("batch=%s attempt=%s unexpected_error=%s", batch_no, attempt, str(exc))
            raise

        backoff = 2 ** (attempt - 1)
        await asyncio.sleep(backoff)

    if last_raw is not None:
        logger.error("batch=%s exhausted_retries last_raw=%s", batch_no, last_raw[:2000])
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=ERR_AI_JSON)


def _q_num_key(q_num: str) -> tuple[int, str]:
    digits = re.findall(r"\d+", q_num)
    if not digits:
        return (10**9, q_num)
    return (int(digits[0]), q_num)


def merge_detected_questions(raw: list[DetectedQuestion]) -> list[DetectedQuestion]:
    """Merge duplicate detections from overlapping batches."""

    by_q: dict[str, list[QuestionSegment]] = {}
    for q in raw:
        if not q.q_num:
            continue
        segments = by_q.setdefault(q.q_num, [])
        for seg in q.segments:
            is_dup = any(
                (existing.page == seg.page and abs(existing.y_start_pct - seg.y_start_pct) < 3.0)
                for existing in segments
            )
            if not is_dup:
                segments.append(seg)

    merged: list[DetectedQuestion] = []
    for q_num, segments in by_q.items():
        segments_sorted = sorted(segments, key=lambda s: (s.page, s.y_start_pct))
        merged.append(DetectedQuestion(q_num=q_num, segments=segments_sorted))

    merged.sort(key=lambda q: _q_num_key(q.q_num))
    return merged


async def detect_all_questions(
    client: anthropic.AsyncAnthropic,
    page_images: list[Image.Image],
    settings: Settings,
    progress_callback: Optional[Callable[[str], Awaitable[None]]] = None,
) -> list[DetectedQuestion]:
    """Process all pages in overlapping batches and merge results."""

    if not page_images:
        return []

    step = settings.AI_BATCH_SIZE - settings.AI_BATCH_OVERLAP
    if step <= 0:
        step = settings.AI_BATCH_SIZE

    raw: list[DetectedQuestion] = []
    total_pages = len(page_images)

    for start_idx in range(0, total_pages, step):
        batch_images = page_images[start_idx : start_idx + settings.AI_BATCH_SIZE]
        if not batch_images:
            break

        page_start = start_idx + 1
        page_end = page_start + len(batch_images) - 1
        if progress_callback is not None:
            await progress_callback(f"Detecting questions: pages {page_start}-{page_end}")

        batch_questions = await detect_questions_in_batch(
            client=client,
            images=batch_images,
            page_start=page_start,
            settings=settings,
            retries=settings.AI_MAX_RETRIES,
        )
        raw.extend(batch_questions)

    merged = merge_detected_questions(raw)
    if progress_callback is not None:
        await progress_callback(f"Detection complete: {len(merged)} questions")
    return merged
