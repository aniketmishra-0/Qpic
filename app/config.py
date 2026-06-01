"""Application configuration using pydantic-settings."""

from __future__ import annotations

from typing import Optional

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Environment-backed settings for the service."""

    # --- AI vision tier (optional) -----------------------------------------
    # Two providers are supported:
    #   * "anthropic"  — uses ANTHROPIC_API_KEY with the Anthropic SDK.
    #   * "openrouter" — uses OPENROUTER_API_KEY against OpenRouter's
    #     OpenAI-compatible endpoint (lets you use Gemini/Qwen/Llama/free models).
    # AI_PROVIDER selects which; "auto" prefers OpenRouter when its key is set,
    # else Anthropic.
    AI_PROVIDER: str = "auto"

    # Anthropic
    ANTHROPIC_API_KEY: Optional[str] = None
    # Anthropic's most capable vision model — best detection + answer-key reading
    # on hard/scanned papers. Override in .env to trade accuracy for cost/speed.
    CLAUDE_MODEL: str = "claude-opus-4-8"

    # OpenRouter (OpenAI-compatible)
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    # A vision-capable model id. Defaults to a free model so it works out of the
    # box on a no-credit key; override with a paid one (e.g.
    # "google/gemini-3.5-flash") for higher accuracy.
    OPENROUTER_MODEL: str = "nvidia/nemotron-nano-12b-v2-vl:free"
    # Cap response tokens so a low-credit OpenRouter account isn't rejected (a
    # huge default max_tokens triggers a 402 "requires more credits").
    AI_MAX_TOKENS: int = 1500

    # PDF rendering
    PDF_RENDER_DPI: int = 200

    # DPI used when rasterizing the final crops. Rendered straight from the PDF
    # vector source (not upscaled from the detection render) so text stays sharp
    # when zoomed. Higher = crisper but larger PNGs.
    CROP_RENDER_DPI: int = 400

    # Detection pipeline
    MIN_QUESTIONS_PER_2_PAGES: float = 0.5
    QUESTION_PADDING_PX: int = 20

    # Languages passed to Tesseract OCR. Combine multiple with "+" so a paper in
    # either script is read correctly — "eng+hin" reads both English and Hindi
    # (Devanagari). This matters a lot: with the wrong/มissing language Tesseract
    # can't recognise the body text, so a question's crop is built from only the
    # few lines it did read and stops short (the classic half-crop). Languages
    # that aren't actually installed are dropped at runtime, so a missing pack
    # degrades to whatever is available instead of erroring.
    OCR_LANGUAGES: str = "eng+hin"

    # Mean OCR word confidence (0-100) below which a page is considered "weak"
    # and, in smart mode with an AI key configured, re-detected by the AI tier on
    # its own instead of trusting OCR's garbled read. Lower = escalate less.
    OCR_MIN_CONFIDENCE: float = 75.0

    # AI batch settings (Tier 3 only)
    AI_BATCH_SIZE: int = 4
    AI_BATCH_OVERLAP: int = 1
    AI_MAX_RETRIES: int = 3

    # Answer-sheet export. When on, every download also contains an
    # ``answers.csv`` + ``answers.json`` mapping each cropped question to the
    # correct option (A-D) read from the paper's own answer key. The key is read
    # for free from the PDF text first; on a scanned paper where the text layer
    # is empty, the AI vision tier (Opus) reads it from the page images instead
    # — only when Online mode is on and a key is configured.
    ANSWER_SHEET_ENABLED: bool = True

    MAX_PDF_SIZE_MB: int = 50
    MAX_PAGES: int = 100
    # Separate, far higher limits for the standalone PDF tools (Compress / Edit /
    # Preflight). These do plain PyMuPDF work — no per-page AI/OCR — so they can
    # safely chew through big documents the cropper never should. Compress in
    # particular is meant for the multi-hundred-MB scans people actually need to
    # shrink, so the size ceiling is in gigabytes.
    MAX_TOOLS_PDF_SIZE_MB: int = 2048
    MAX_TOOLS_PAGES: int = 2000
    # Hard ceiling for a single streamed rename batch (across all chunks). The
    # batch never lives in memory — files are spooled to disk and the ZIP is
    # built from disk — so this can be large. Bump it if you routinely pack
    # multi-gigabyte batches.
    MAX_RENAME_BATCH_MB: int = 4096
    TEMP_DIR: str = "temp"
    # Job dirs (incl. the source PDF + page previews cached for the smart
    # review/manual-crop flow) are kept this long. It must comfortably exceed
    # how long a user might spend hand-fixing crops in the review popup before
    # hitting "Combine & Download", or finalize would 404 on a cleaned-up job.
    CLEANUP_AFTER_SECONDS: int = 1800
    API_TIMEOUT_SECONDS: int = 120

    model_config = ConfigDict(env_file=".env")

    def resolved_ai_provider(self) -> Optional[str]:
        """Return the active AI provider ("openrouter"/"anthropic") or None.

        Honours ``AI_PROVIDER`` when it names a provider whose key is present.
        In "auto" mode OpenRouter wins when its key is set, otherwise Anthropic.
        Returns None when no usable key is configured (AI tier stays off).
        """

        provider = (self.AI_PROVIDER or "auto").strip().lower()
        has_or = bool(self.OPENROUTER_API_KEY and self.OPENROUTER_API_KEY.strip())
        has_anthropic = bool(self.ANTHROPIC_API_KEY and self.ANTHROPIC_API_KEY.strip())

        if provider == "openrouter":
            return "openrouter" if has_or else None
        if provider == "anthropic":
            return "anthropic" if has_anthropic else None

        # auto
        if has_or:
            return "openrouter"
        if has_anthropic:
            return "anthropic"
        return None

    def ai_is_configured(self) -> bool:
        """True when some AI vision provider has a usable key."""

        return self.resolved_ai_provider() is not None
