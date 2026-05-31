"""AI vision reader for a paper's answer key (Opus / OpenRouter).

The text parser in :mod:`answer_key` reads the key for free from a searchable
PDF's text layer. On a *scanned* paper that layer is empty, so the key — which
is exactly what the answer-sheet export needs — can't be parsed at all. This
module fills that gap: when Online mode is on and an AI key is configured, it
shows the page images to the vision model and asks it to transcribe the
answer-key grid into ``{question_number: option_letter}``.

It mirrors the provider split used by the detection tier:
  * Anthropic  → the shared async client (``claude-opus-*`` by default).
  * OpenRouter → the OpenAI-compatible ``/chat/completions`` endpoint.

The function is best-effort: any failure (no key, transient API error,
unparseable response) returns an empty dict so the answer sheet simply omits the
answers instead of breaking the crop/download flow.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

import httpx
from PIL import Image

from ...config import Settings
from ...utils.image_utils import img_to_base64_png, resize_for_api
from .ai_detector import _strip_json_fences

logger = logging.getLogger(__name__)


ANSWER_KEY_PROMPT = """
You are reading the ANSWER KEY of an MCQ exam paper.

An answer key pairs each question number with its correct option letter, e.g.
"1. (b)  2. (a)  3. (d)" or "1-B 2-A 3-D" or a grid/table of number→letter.

Read EVERY pair you can see across the pages. Options are single letters A-D
(occasionally A-E). Ignore the questions themselves, marks, and instructions —
only transcribe the question-number → correct-option pairs from the key.

Return ONLY valid JSON, nothing else:
{"answers": {"1": "B", "2": "A", "3": "D"}}

If no answer key is visible, return {"answers": {}}.
""".strip()

# How many pages to show the model. Answer keys are compact and almost always
# sit at the very end of a paper, so we read the last few pages rather than the
# whole document — bounding cost while still catching the key.
_MAX_KEY_PAGES = 6


def _coerce_answer_map(data: object) -> dict[int, str]:
    """Turn a model's JSON into ``{int question: 'A'-'E'}``, dropping junk."""

    if isinstance(data, dict) and "answers" in data:
        data = data.get("answers")
    if not isinstance(data, dict):
        return {}

    out: dict[int, str] = {}
    for raw_num, raw_letter in data.items():
        digits = re.findall(r"\d+", str(raw_num))
        if not digits:
            continue
        try:
            num = int(digits[0])
        except (TypeError, ValueError):
            continue
        if num <= 0 or num > 999:
            continue
        letter = str(raw_letter or "").strip().upper()
        m = re.search(r"[A-E]", letter)
        if not m:
            continue
        out.setdefault(num, m.group(0))
    return out


def _key_pages(page_images: list[Image.Image]) -> list[tuple[int, Image.Image]]:
    """Return ``(page_no, image)`` for the last few pages (where keys live)."""

    total = len(page_images)
    if total == 0:
        return []
    start = max(0, total - _MAX_KEY_PAGES)
    pages: list[tuple[int, Image.Image]] = []
    for idx in range(start, total):
        # Lazy page views rasterise on access; tolerate a render failure.
        try:
            pages.append((idx + 1, page_images[idx]))
        except Exception:
            continue
    return pages


async def _read_anthropic(
    client, model: str, max_tokens: int, pages: list[tuple[int, Image.Image]]
) -> dict[int, str]:
    content_blocks: list[dict] = []
    for page_no, img in pages:
        b64 = img_to_base64_png(resize_for_api(img, max_px=1568))
        content_blocks.append({"type": "text", "text": f"=== PAGE {page_no} ==="})
        content_blocks.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": b64},
            }
        )

    message = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0,
        system=ANSWER_KEY_PROMPT,
        messages=[{"role": "user", "content": content_blocks}],
    )

    parts: list[str] = []
    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(str(getattr(block, "text", "")))
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    text = "\n".join(parts).strip()
    return _coerce_answer_map(json.loads(_strip_json_fences(text)))


async def _read_openrouter(
    settings: Settings, pages: list[tuple[int, Image.Image]]
) -> dict[int, str]:
    content: list[dict] = [{"type": "text", "text": ANSWER_KEY_PROMPT}]
    for page_no, img in pages:
        b64 = img_to_base64_png(resize_for_api(img, max_px=1568))
        content.append({"type": "text", "text": f"=== PAGE {page_no} ==="})
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        )

    body = {
        "model": settings.OPENROUTER_MODEL,
        "temperature": 0,
        "max_tokens": settings.AI_MAX_TOKENS,
        "messages": [{"role": "user", "content": content}],
    }
    headers = {
        "Authorization": f"Bearer {(settings.OPENROUTER_API_KEY or '').strip()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",
        "X-Title": "Qpic",
    }

    async with httpx.AsyncClient(timeout=settings.API_TIMEOUT_SECONDS) as client:
        resp = await client.post(
            f"{settings.OPENROUTER_BASE_URL.rstrip('/')}/chat/completions",
            headers=headers,
            json=body,
        )
        if resp.status_code != 200:
            logger.warning("ai_answer_key openrouter status=%s", resp.status_code)
            return {}
        choices = (resp.json() or {}).get("choices") or []
        if not choices:
            return {}
        text = str((choices[0].get("message") or {}).get("content") or "")
    return _coerce_answer_map(json.loads(_strip_json_fences(text)))


async def read_answer_key_with_ai(
    settings: Settings,
    page_images: list[Image.Image],
    *,
    anthropic_client=None,
) -> dict[int, str]:
    """Read the paper's answer key from page images with the AI vision tier.

    Returns ``{question_number: option_letter}`` or an empty dict. Best-effort:
    swallows any provider/parse error so the caller (answer-sheet export) can
    degrade gracefully to "no answers" rather than failing the request.

    The provider is resolved from settings (OpenRouter or Anthropic); when no
    usable key is configured this returns ``{}`` immediately.
    """

    provider = settings.resolved_ai_provider()
    if provider is None:
        return {}

    pages = _key_pages(page_images)
    if not pages:
        return {}

    try:
        if provider == "anthropic":
            if anthropic_client is None:
                import anthropic

                anthropic_client = anthropic.AsyncAnthropic(
                    api_key=settings.ANTHROPIC_API_KEY,
                    timeout=httpx.Timeout(settings.API_TIMEOUT_SECONDS),
                )
            key = await _read_anthropic(
                anthropic_client,
                settings.CLAUDE_MODEL,
                # Keys can be long; give the model room without the 4096 cap.
                max_tokens=max(1024, settings.AI_MAX_TOKENS),
                pages=pages,
            )
        else:
            key = await _read_openrouter(settings, pages)
    except Exception as exc:  # pragma: no cover - network/parse failures
        logger.warning("ai_answer_key_failed provider=%s error=%s", provider, str(exc))
        return {}

    if key:
        logger.info("ai_answer_key provider=%s pairs=%s", provider, len(key))
    return key
