"""Shared FastAPI dependencies."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

import anthropic
from fastapi import Request

from .config import Settings

logger = logging.getLogger(__name__)


@lru_cache
def _cached_settings() -> Settings:
    """Create Settings once for non-request contexts."""

    return Settings()


def get_settings(request: Request) -> Settings:
    """Return settings from app state when available."""

    settings = getattr(request.app.state, "settings", None)
    if isinstance(settings, Settings):
        return settings
    return _cached_settings()


def get_anthropic_client_optional(request: Request) -> Optional[anthropic.AsyncAnthropic]:
    """Return the shared Anthropic async client from app state when configured."""

    client = getattr(request.app.state, "anthropic_client", None)
    if isinstance(client, anthropic.AsyncAnthropic):
        return client
    return None


def build_ai_detector(
    settings: Settings,
    *,
    use_ai: bool = True,
    anthropic_client: Optional[anthropic.AsyncAnthropic] = None,
):
    """Return a vision detector for the active provider, or None.

    Honours the user's online/offline toggle (``use_ai``): when False the AI
    tier is disabled entirely (fully offline run) and None is returned, so the
    pipeline relies on the text/OCR tiers only. When True, the provider is
    resolved from settings:

      * "openrouter" → :class:`OpenRouterDetector` (works with the user's
        OpenRouter key and any vision model they pick).
      * "anthropic"  → :class:`AIDetector` (Anthropic SDK).

    Returns None when AI is toggled off or no usable key is configured.
    """

    if not use_ai:
        return None

    provider = settings.resolved_ai_provider()
    if provider == "openrouter":
        # Imported lazily to avoid a hard dependency when unused.
        from .services.detector.openrouter_detector import OpenRouterDetector

        return OpenRouterDetector(
            settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL,
            model=settings.OPENROUTER_MODEL,
            max_tokens=settings.AI_MAX_TOKENS,
            timeout_seconds=settings.API_TIMEOUT_SECONDS,
        )
    if provider == "anthropic":
        from .services.detector.ai_detector import AIDetector

        return AIDetector(
            settings.ANTHROPIC_API_KEY,
            client=anthropic_client,
            timeout_seconds=settings.API_TIMEOUT_SECONDS,
        )
    return None
