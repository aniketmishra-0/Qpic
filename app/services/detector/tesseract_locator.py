"""Locate the Tesseract OCR binary across dev, Docker and PyInstaller builds.

pytesseract works by shelling out to a ``tesseract`` executable. In a normal
dev or Docker environment that binary is on ``PATH``, but a packaged desktop
build (PyInstaller ``.app`` / ``.exe``) has no PATH guarantees and PyInstaller
never bundles Tesseract automatically. This module finds the binary (and its
``tessdata`` language files) in priority order and points pytesseract at it:

1. ``TESSERACT_CMD`` env var — explicit override, wins over everything.
2. A copy shipped *inside* the frozen app (``<bundle>/tesseract/``). This is
   what the build produces when ``scripts/vendor_tesseract.py`` populated
   ``vendor/tesseract`` before PyInstaller ran.
3. Common per-OS install locations (Homebrew, UB-Mannheim, apt).
4. Whatever ``tesseract`` is already on ``PATH`` (pytesseract's own default).

If nothing is found we leave pytesseract's default in place; the OCR tier then
reports itself unavailable and the pipeline degrades gracefully (text/AI tiers
still run).
"""

from __future__ import annotations

import logging
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import Optional

import pytesseract

logger = logging.getLogger(__name__)

# configure_tesseract() is idempotent; this caches that it already ran so the
# per-page OCR loop doesn't repeat the filesystem probing on every call.
_configured = False


def _exe_name() -> str:
    return "tesseract.exe" if os.name == "nt" else "tesseract"


def _bundle_dirs() -> list[Path]:
    """Directories a PyInstaller build may have extracted/placed resources in."""

    dirs: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        dirs.append(Path(meipass))
    # onedir builds (COLLECT) keep data next to the executable too.
    if getattr(sys, "frozen", False):
        dirs.append(Path(sys.executable).resolve().parent)
    return dirs


def _candidate_paths() -> list[Path]:
    """Ordered list of places to look for the Tesseract binary."""

    name = _exe_name()
    candidates: list[Path] = []

    # 2. Bundled alongside the frozen app (vendored at build time).
    for base in _bundle_dirs():
        candidates.append(base / "tesseract" / name)

    # 3. Common per-OS install locations.
    if sys.platform == "darwin":
        candidates += [Path("/opt/homebrew/bin") / name, Path("/usr/local/bin") / name]
    elif os.name == "nt":
        for env_key in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
            base = os.environ.get(env_key)
            if base:
                candidates.append(Path(base) / "Tesseract-OCR" / name)
    else:  # linux / other unix
        candidates += [Path("/usr/bin") / name, Path("/usr/local/bin") / name]

    return candidates


def _ensure_executable(binary: Path) -> None:
    """Restore the exec bit (PyInstaller's ``datas`` extraction can drop it)."""

    if os.name == "nt":
        return
    try:
        mode = binary.stat().st_mode
        if not (mode & 0o111):
            binary.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except OSError:
        pass


def _configure_tessdata(binary: Path) -> None:
    """Point ``TESSDATA_PREFIX`` at language data shipped with the binary.

    A bundled Tesseract carries its own ``tessdata`` so it never depends on a
    (possibly absent) system copy. We only set this when it isn't already set,
    so an explicit user/Docker value always wins.
    """

    if os.environ.get("TESSDATA_PREFIX"):
        return
    candidates = (
        binary.parent / "tessdata",
        binary.parent.parent / "share" / "tessdata",
    )
    for cand in candidates:
        if cand.is_dir():
            os.environ["TESSDATA_PREFIX"] = str(cand)
            return


def configure_tesseract(force: bool = False) -> str:
    """Locate Tesseract and point pytesseract at it. Returns the chosen command.

    Safe to call repeatedly: the probing runs once and the result is cached
    unless ``force`` is set.
    """

    global _configured
    if _configured and not force:
        return pytesseract.pytesseract.tesseract_cmd

    chosen: Optional[Path] = None

    # 1. Explicit override.
    override = os.environ.get("TESSERACT_CMD")
    if override:
        p = Path(override)
        if p.is_file():
            chosen = p
        else:
            logger.warning("tesseract_cmd_override_missing path=%s", override)

    # 2 + 3. Bundled, then common install locations.
    if chosen is None:
        for cand in _candidate_paths():
            try:
                if cand.is_file():
                    chosen = cand
                    break
            except OSError:
                continue

    # 4. Fall back to a PATH lookup.
    if chosen is None:
        on_path = shutil.which(_exe_name())
        if on_path:
            chosen = Path(on_path)

    if chosen is not None:
        _ensure_executable(chosen)
        pytesseract.pytesseract.tesseract_cmd = str(chosen)
        _configure_tessdata(chosen)
        logger.info(
            "tesseract_configured path=%s tessdata=%s",
            chosen,
            os.environ.get("TESSDATA_PREFIX", "<system default>"),
        )
    else:
        logger.info(
            "tesseract_not_located using_default=%s",
            pytesseract.pytesseract.tesseract_cmd,
        )

    _configured = True
    return pytesseract.pytesseract.tesseract_cmd


def available_languages() -> set[str]:
    """Return the set of Tesseract language codes installed on this machine.

    Used to filter a requested ``eng+hin`` down to the packs that actually
    exist, so asking for a language whose ``traineddata`` is missing degrades to
    whatever is available instead of making Tesseract error out. Returns an
    empty set if the language list can't be read (caller then falls back to the
    requested string unchanged).
    """

    configure_tesseract()
    try:
        langs = pytesseract.get_languages(config="")
        return {str(l).strip() for l in langs if str(l).strip()}
    except Exception as exc:  # pragma: no cover - depends on local install
        logger.warning("tesseract_list_langs_failed error=%s", str(exc))
        return set()


def resolve_languages(requested: str) -> str:
    """Keep only the requested OCR languages that are actually installed.

    ``requested`` is a Tesseract language spec like ``"eng+hin"``. We split on
    "+", drop any pack not present on the machine, and re-join. When none of the
    requested packs are installed (or we couldn't read the list) we return the
    original string so Tesseract makes its own decision rather than getting an
    empty language — i.e. this never makes things worse than before.
    """

    wanted = [l.strip() for l in (requested or "").split("+") if l.strip()]
    if not wanted:
        return requested
    installed = available_languages()
    if not installed:
        return requested
    keep = [l for l in wanted if l in installed]
    if not keep:
        logger.warning(
            "ocr_requested_langs_missing requested=%s installed=%s",
            requested,
            sorted(installed),
        )
        # Prefer English if present, else let Tesseract default.
        return "eng" if "eng" in installed else (sorted(installed)[0] if installed else requested)
    if len(keep) < len(wanted):
        logger.info(
            "ocr_langs_filtered requested=%s using=%s", requested, "+".join(keep)
        )
    return "+".join(keep)
