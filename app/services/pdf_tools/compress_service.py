"""Compress a PDF — strongest-possible size reduction with quality control.

The heavy lifting is image recompression: in almost every oversized PDF the bulk
is high-resolution embedded images (scanned pages, screenshots, photos). We use
PyMuPDF's ``rewrite_images`` to downsample + re-encode those images to a target
DPI and JPEG quality, then ``subset_fonts`` to drop unused glyphs, then a
garbage-collected, deflated ``save`` to squeeze the object stream.

Two ways to drive it:

* **Level** — ``light`` / ``balanced`` / ``strong`` / ``extreme`` map to a DPI +
  quality preset, from "barely touch quality" to "smallest file".
* **Target size (MB)** — iteratively pushes DPI/quality down until the output is
  at or below the requested size (best-effort; reports the smallest it reached).

Everything runs on in-memory bytes and is CPU-bound, so callers should invoke it
inside ``asyncio.to_thread``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import fitz

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Preset:
    """A single compression preset: target image DPI + JPEG quality."""

    dpi_target: int
    quality: int


# Ordered from gentlest to most aggressive. ``rewrite_images`` only touches
# images whose effective resolution exceeds ``dpi_target`` (via dpi_threshold),
# so text-only PDFs are unaffected and stay sharp.
#
# We deliberately never force colour images to grayscale. It saves little over
# DPI + quality reduction, but on dark-coloured artwork (e.g. a navy slide deck)
# it collapses the low-luminance colour to near-black and the whole page reads
# as a dark smear. Size now comes from downsampling + JPEG quality alone, which
# keeps colours intact.
LEVELS: dict[str, _Preset] = {
    "light": _Preset(dpi_target=200, quality=85),
    "balanced": _Preset(dpi_target=150, quality=72),
    "strong": _Preset(dpi_target=100, quality=55),
    "extreme": _Preset(dpi_target=72, quality=40),
}

DEFAULT_LEVEL = "balanced"

# The ladder of presets the target-size search walks down, gentlest first, so we
# stop at the *least* destructive preset that meets the target. Kept short (5
# rungs) so the worst case is a handful of passes, not a dozen.
_TARGET_LADDER: list[_Preset] = [
    _Preset(dpi_target=180, quality=80),
    _Preset(dpi_target=130, quality=65),
    _Preset(dpi_target=100, quality=50),
    _Preset(dpi_target=80, quality=38),
    _Preset(dpi_target=60, quality=25),
]


@dataclass(frozen=True)
class CompressResult:
    """Outcome of a compression run."""

    data: bytes
    original_size: int
    compressed_size: int
    level: str
    target_met: Optional[bool]  # None when no target was requested
    note: str = ""

    @property
    def ratio(self) -> float:
        """Fraction of the original size removed (0.0-1.0)."""

        if self.original_size <= 0:
            return 0.0
        saved = self.original_size - self.compressed_size
        return max(0.0, saved / self.original_size)


def _save_optimized(doc: "fitz.Document") -> bytes:
    """Serialize a document with maximum lossless object/stream savings.

    ``garbage=4`` removes unreferenced objects and merges duplicates,
    ``deflate=True`` zips streams, ``clean`` sanitizes content streams. This is
    the lossless half of the squeeze that always runs after image recompression.
    """

    return doc.tobytes(
        garbage=4,
        deflate=True,
        deflate_images=True,
        deflate_fonts=True,
        clean=True,
        use_objstms=1,
    )


def _apply_preset(doc: "fitz.Document", preset: _Preset) -> None:
    """Recompress images and subset fonts in-place for one preset."""

    # Only rewrite images whose stored resolution is above the target, so we
    # never *upscale* or needlessly re-encode an already-small image.
    # ``dpi_threshold`` must be strictly greater than ``dpi_target`` (PyMuPDF
    # constraint), so we look one DPI above the target and downsample to it.
    # ``rewrite_images`` is newer PyMuPDF; on an older build we silently skip the
    # image pass and still get the lossless object/stream savings from save().
    if hasattr(doc, "rewrite_images"):
        try:
            doc.rewrite_images(
                dpi_threshold=preset.dpi_target + 1,
                dpi_target=preset.dpi_target,
                quality=preset.quality,
                lossy=True,
                lossless=True,
                bitonal=False,  # leave crisp B/W scans alone — JPEG would smear them
                color=True,
                gray=True,
                set_to_gray=False,  # never desaturate — dark colours collapse to black
            )
        except Exception as exc:  # pragma: no cover - depends on PDF internals
            logger.warning("rewrite_images_failed error=%s", str(exc))

    # Drop unused glyphs from embedded fonts (often a big, free win).
    try:
        doc.subset_fonts(fallback=False)
    except Exception as exc:  # pragma: no cover
        logger.debug("subset_fonts_skipped error=%s", str(exc))


def compress_pdf(
    file_bytes: bytes,
    *,
    level: str = DEFAULT_LEVEL,
    target_mb: Optional[float] = None,
) -> CompressResult:
    """Compress a PDF and return the smaller bytes plus size stats.

    When ``target_mb`` is given the level is ignored and the function walks a
    ladder of increasingly aggressive presets, stopping at the first one that
    brings the file to or below the target. If even the most aggressive preset
    can't reach it, the smallest result produced is returned with
    ``target_met=False``.

    The original bytes are returned unchanged if compression somehow made the
    file *larger* (rare, but possible on an already-optimized PDF) — the caller
    always gets the smaller of the two.
    """

    original_size = len(file_bytes)

    if target_mb is not None and target_mb > 0:
        return _compress_to_target(file_bytes, target_bytes=int(target_mb * 1024 * 1024))

    preset = LEVELS.get(level, LEVELS[DEFAULT_LEVEL])
    out = _compress_once(file_bytes, preset)

    # Never hand back something bigger than the input.
    if len(out) >= original_size:
        return CompressResult(
            data=file_bytes,
            original_size=original_size,
            compressed_size=original_size,
            level=level,
            target_met=None,
            note="Already optimized — couldn't shrink it further without quality loss.",
        )

    return CompressResult(
        data=out,
        original_size=original_size,
        compressed_size=len(out),
        level=level,
        target_met=None,
    )


def _compress_once(file_bytes: bytes, preset: _Preset) -> bytes:
    """Run one full compress pass (image recompress + font subset + save)."""

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        _apply_preset(doc, preset)
        return _save_optimized(doc)
    finally:
        doc.close()


def _compress_to_target(file_bytes: bytes, *, target_bytes: int) -> CompressResult:
    """Walk presets until the output fits ``target_bytes`` (best-effort)."""

    original_size = len(file_bytes)
    best: Optional[bytes] = None
    used_preset: _Preset = _TARGET_LADDER[0]

    for preset in _TARGET_LADDER:
        out = _compress_once(file_bytes, preset)
        if best is None or len(out) < len(best):
            best = out
            used_preset = preset
        if len(out) <= target_bytes:
            return CompressResult(
                data=out,
                original_size=original_size,
                compressed_size=len(out),
                level=f"target {target_bytes / (1024 * 1024):.1f}MB",
                target_met=True,
                note=f"Reached target at ~{preset.dpi_target} DPI, quality {preset.quality}.",
            )

    # Couldn't hit the target; hand back the smallest we managed.
    assert best is not None
    smaller = best if len(best) < original_size else file_bytes
    return CompressResult(
        data=smaller,
        original_size=original_size,
        compressed_size=len(smaller),
        level=f"target {target_bytes / (1024 * 1024):.1f}MB",
        target_met=False,
        note=(
            "Couldn't reach the target without destroying readability — this is "
            f"the smallest safe result (~{used_preset.dpi_target} DPI)."
        ),
    )
