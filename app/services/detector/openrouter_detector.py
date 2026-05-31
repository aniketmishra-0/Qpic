"""AI vision detector backed by OpenRouter's OpenAI-compatible API.

Mirrors :class:`AIDetector`'s interface (``is_available`` / ``detect``) so the
detection pipeline can use either backend interchangeably. OpenRouter exposes a
single OpenAI-style ``/chat/completions`` endpoint that fans out to many vision
models (Gemini, Qwen, Llama, plus free tiers), so a user can point the AI tier
at whatever model their key affords by setting ``OPENROUTER_MODEL``.

The request shape differs from Anthropic's (OpenAI ``messages`` with
``image_url`` data-URIs instead of Anthropic image blocks), but the *response*
we want is identical JSON, so detection/coercion is shared with the Anthropic
tier via :func:`_coerce_questions` and :func:`merge_detected_questions`.
"""

from __future__ import annotations

import asyncio
import json
import logging

import httpx
from PIL import Image

from ...config import Settings
from ...models.schemas import DetectedQuestion
from ...utils.image_utils import img_to_base64_png, resize_for_api
from .ai_detector import (
    DETECTION_PROMPT,
    _coerce_questions,
    _strip_json_fences,
    _style_hint,
    merge_detected_questions,
)

logger = logging.getLogger(__name__)


class OpenRouterDetector:
    """Vision question detector using OpenRouter (OpenAI-compatible)."""

    def __init__(
        self,
        api_key: "str | None",
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "nvidia/nemotron-nano-12b-v2-vl:free",
        max_tokens: int = 1500,
        timeout_seconds: int = 120,
        max_concurrency: int = 4,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        # How many page requests may be in flight at once. Pages are independent
        # (one request each), so firing them concurrently turns an N-page run
        # from N sequential round-trips into ~ceil(N / max_concurrency) waves.
        # Kept modest so we stay within provider rate limits.
        self.max_concurrency = max(1, max_concurrency)
        self.available = bool(self.api_key)

    def is_available(self) -> bool:
        return self.available

    async def detect(
        self,
        page_images: list[Image.Image],
        settings: Settings,
        *,
        marker_style: str = "auto",
    ) -> list[DetectedQuestion]:
        if not self.is_available() or not page_images:
            return []

        style_hint = _style_hint(marker_style)

        # Free / smaller vision models on OpenRouter are unreliable when several
        # images are packed into one request (they often return an empty array
        # or only see the first page). Sending one page per request keeps each
        # call simple and dramatically improves detection on those models. We
        # remap each single-page result back to its real page number.
        #
        # Pages are independent, so we fan the per-page requests out
        # concurrently (bounded by ``max_concurrency``) instead of awaiting them
        # one after another — an 8-page paper that used to take 8 serial
        # round-trips now finishes in ~2 waves of 4. A semaphore caps in-flight
        # requests so we don't trip provider rate limits.
        total_pages = len(page_images)
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:

            async def _detect_page(idx: int) -> list[DetectedQuestion]:
                page_no = idx + 1
                async with semaphore:
                    # Lazy page views render on access; do it off the event loop.
                    img = await asyncio.to_thread(lambda i=idx: page_images[i])
                    return await self._detect_batch(
                        client, [img], page_no, settings, style_hint
                    )

            results = await asyncio.gather(
                *(_detect_page(idx) for idx in range(total_pages))
            )

        raw: list[DetectedQuestion] = [q for page in results for q in page]
        return merge_detected_questions(raw)

    async def _detect_batch(
        self,
        client: httpx.AsyncClient,
        images: list[Image.Image],
        page_start: int,
        settings: Settings,
        style_hint: str,
    ) -> list[DetectedQuestion]:
        page_end = page_start + len(images) - 1

        content: list[dict] = [
            {"type": "text", "text": DETECTION_PROMPT + style_hint}
        ]
        for i, img in enumerate(images):
            page_no = page_start + i
            b64 = img_to_base64_png(resize_for_api(img, max_px=1568))
            content.append({"type": "text", "text": f"=== PAGE {page_no} ==="})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )

        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # OpenRouter asks for these for attribution; harmless if omitted.
            "HTTP-Referer": "https://localhost",
            "X-Title": "Qpic",
        }

        for attempt in range(1, settings.AI_MAX_RETRIES + 1):
            try:
                resp = await client.post(
                    f"{self.base_url}/chat/completions", headers=headers, json=body
                )
                if resp.status_code == 200:
                    text = self._extract_content(resp.json())
                    try:
                        data = json.loads(_strip_json_fences(text))
                    except json.JSONDecodeError:
                        # Some models echo prose around / instead of JSON. Treat
                        # as an empty page rather than failing the whole run.
                        logger.warning(
                            "openrouter page_range=%s-%s non_json_response=%s",
                            page_start, page_end, repr(text)[:200],
                        )
                        return []
                    questions = _coerce_questions(
                        data=data, page_start=page_start, page_end=page_end
                    )
                    logger.info(
                        "openrouter page_range=%s-%s model=%s detected=%s",
                        page_start, page_end, self.model, len(questions),
                    )
                    return questions

                # Rate-limited or transient upstream error → back off and retry.
                if resp.status_code in (429, 500, 502, 503, 529):
                    logger.warning(
                        "openrouter attempt=%s status=%s retrying",
                        attempt, resp.status_code,
                    )
                else:
                    # 401/402/4xx config errors won't fix themselves — stop.
                    logger.error(
                        "openrouter non_retryable status=%s body=%s",
                        resp.status_code, resp.text[:300],
                    )
                    return []
            except (json.JSONDecodeError, httpx.HTTPError) as exc:
                logger.warning("openrouter attempt=%s error=%s", attempt, str(exc))

            await asyncio.sleep(2 ** (attempt - 1))

        logger.error("openrouter exhausted_retries page_range=%s-%s", page_start, page_end)
        return []

    @staticmethod
    def _extract_content(data: dict) -> str:
        """Pull the assistant text out of an OpenAI-style chat completion."""

        try:
            choices = data.get("choices") or []
            if not choices:
                return ""
            msg = choices[0].get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                return content
            # Some providers return a content list of parts.
            if isinstance(content, list):
                parts = [
                    str(p.get("text", ""))
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                return "\n".join(parts)
        except Exception:  # pragma: no cover - defensive
            return ""
        return ""

    async def aclose(self) -> None:
        # Stateless (client is created per detect call); nothing to close.
        return
