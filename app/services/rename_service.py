"""Batch image rename utilities.

This powers the standalone "Rename Batch" tool: a user uploads any number of
images, picks a naming pattern, and downloads a ZIP of the renamed files. The
image bytes are normally copied verbatim — only the filename changes — so every
file keeps its exact original format. When the user picks an explicit output
format (PNG/JPG/WEBP) the bytes are re-encoded to match.
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Extensions we accept for renaming. The original extension is always preserved
# on output, so this is purely an input guard.
ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "gif",
    "bmp",
    "webp",
    "tif",
    "tiff",
}

# The placeholder replaced by the running number inside a pattern.
NUMBER_TOKEN = "#"


def split_extension(filename: str) -> tuple[str, str]:
    """Return ``(stem, ext)`` for a filename; ``ext`` is lowercase, no dot.

    ``"Photo.JPG"`` -> ``("Photo", "jpg")``. A name with no extension returns an
    empty ``ext`` so the output simply has no extension either.
    """

    name = Path(filename or "").name
    if "." not in name:
        return name, ""
    stem, _, ext = name.rpartition(".")
    # A leading-dot file like ".env" has an empty stem; treat the whole thing as
    # the stem so we don't drop it.
    if stem == "":
        return name, ""
    return stem, ext.lower()


def build_renamed(
    original_name: str,
    *,
    pattern: str,
    number: int,
    padding: int = 0,
) -> str:
    """Build a single renamed filename, preserving the original extension.

    ``pattern`` may contain one or more ``#`` tokens; each is replaced by
    ``number`` zero-padded to ``padding`` digits. When the pattern has no ``#``
    the number is appended (so a constant pattern still yields unique names).
    The original file's extension is always re-applied, so the format can never
    change via a rename.
    """

    _, ext = split_extension(original_name)

    pad = max(0, int(padding))
    num_str = str(int(number)).zfill(pad) if pad > 0 else str(int(number))

    base = (pattern or "").strip()
    if not base:
        base = NUMBER_TOKEN

    if NUMBER_TOKEN in base:
        new_stem = base.replace(NUMBER_TOKEN, num_str)
    else:
        new_stem = f"{base}{num_str}"

    # Strip characters that are unsafe in filenames across OSes, but keep common
    # punctuation the user is likely to want (parentheses, dashes, spaces, _).
    new_stem = re.sub(r'[\\/:*?"<>|]+', "_", new_stem).strip()
    if not new_stem:
        new_stem = num_str

    return f"{new_stem}.{ext}" if ext else new_stem


def plan_renames(
    original_names: list[str],
    *,
    pattern: str,
    start: int = 1,
    padding: int = 0,
) -> list[str]:
    """Return the list of output names for ``original_names``, in order.

    Numbers run from ``start`` upward. Any collision (two inputs mapping to the
    same output name) is broken by appending ``_2``, ``_3`` … to the later one
    so nothing is silently overwritten inside the ZIP.
    """

    used: dict[str, int] = {}
    out: list[str] = []
    for idx, original in enumerate(original_names):
        name = build_renamed(
            original,
            pattern=pattern,
            number=start + idx,
            padding=padding,
        )
        if name in used:
            used[name] += 1
            stem, ext = split_extension(name)
            suffix = f"_{used[name]}"
            name = f"{stem}{suffix}.{ext}" if ext else f"{stem}{suffix}"
        else:
            used[name] = 1
        out.append(name)
    return out


# Output formats the user may force. "original" keeps each file's own format
# (bytes copied verbatim); the rest re-encode every image to that format.
OUTPUT_FORMATS = {"original", "png", "jpg", "jpeg", "webp"}

# Maps a chosen output format to (file extension, PIL save format).
_FORMAT_SPEC = {
    "png": ("png", "PNG"),
    "jpg": ("jpg", "JPEG"),
    "jpeg": ("jpg", "JPEG"),
    "webp": ("webp", "WEBP"),
}


def _sanitize_stem(stem: str, fallback: str) -> str:
    """Strip filesystem-unsafe characters from a name stem."""

    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", str(stem or "")).strip()
    return cleaned or fallback


def plan_output_names(
    original_names: list[str],
    *,
    pattern: str,
    start: int = 1,
    padding: int = 0,
    explicit_stems: "list[str] | None" = None,
    output_format: str = "original",
) -> list[str]:
    """Return final output filenames, honouring format + explicit stems.

    When ``explicit_stems`` is given (one entry per file, no extension) those
    names are used directly — this is how the front-end's variable tokens
    (``(name)``, ``(width)``, ``(date)`` …) reach the ZIP. Otherwise names are
    built from ``pattern``/``start``/``padding`` exactly as before.

    The extension is decided by ``output_format``: ``"original"`` keeps each
    file's own extension, anything else forces the matching one (so a JPG export
    of ``photo.png`` becomes ``photo.jpg``). Collisions get ``_2``, ``_3`` …
    """

    fmt = (output_format or "original").strip().lower()
    forced_ext = _FORMAT_SPEC.get(fmt, (None, None))[0]

    # Stems either come from the caller (variables) or the pattern planner.
    if explicit_stems is not None and len(explicit_stems) == len(original_names):
        stems = [
            _sanitize_stem(s, str(start + i))
            for i, s in enumerate(explicit_stems)
        ]
    else:
        planned = plan_renames(
            original_names, pattern=pattern, start=start, padding=padding
        )
        stems = [split_extension(n)[0] for n in planned]

    used: dict[str, int] = {}
    out: list[str] = []
    for original, stem in zip(original_names, stems):
        ext = forced_ext if forced_ext else split_extension(original)[1]
        name = f"{stem}.{ext}" if ext else stem
        if name in used:
            used[name] += 1
            name = f"{stem}_{used[name]}.{ext}" if ext else f"{stem}_{used[name]}"
        else:
            used[name] = 1
        out.append(name)
    return out


def _reencode(raw: bytes, save_format: str, *, jpg_quality: int = 90) -> bytes:
    """Re-encode image bytes to ``save_format`` (PNG/JPEG/WEBP)."""

    from PIL import Image

    with Image.open(io.BytesIO(raw)) as img:
        out = io.BytesIO()
        if save_format == "JPEG":
            img = img.convert("RGB")
            img.save(out, format="JPEG", quality=int(jpg_quality))
        elif save_format == "WEBP":
            img.save(out, format="WEBP", quality=int(jpg_quality))
        else:  # PNG
            if img.mode in ("P", "CMYK"):
                img = img.convert("RGBA" if img.mode == "P" else "RGB")
            img.save(out, format="PNG")
        return out.getvalue()


def write_rename_zip(
    zip_path: Path,
    files: list[tuple[str, bytes]],
    *,
    pattern: str,
    start: int = 1,
    padding: int = 0,
    explicit_stems: "list[str] | None" = None,
    output_format: str = "original",
    jpg_quality: int = 90,
) -> int:
    """Write a ZIP of renamed images and return the count written.

    ``files`` is a list of ``(original_name, raw_bytes)``. With
    ``output_format="original"`` the bytes are written verbatim under the new
    name. Any other format re-encodes each image so a mixed batch comes out as a
    single, consistent format.
    """

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = (output_format or "original").strip().lower()
    save_format = _FORMAT_SPEC.get(fmt, (None, None))[1]

    new_names = plan_output_names(
        [f[0] for f in files],
        pattern=pattern,
        start=start,
        padding=padding,
        explicit_stems=explicit_stems,
        output_format=fmt,
    )

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for (original_name, raw), new_name in zip(files, new_names):
            data = raw
            if save_format is not None:
                try:
                    data = _reencode(raw, save_format, jpg_quality=jpg_quality)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "reencode_failed name=%s error=%s — writing original bytes",
                        original_name,
                        str(exc),
                    )
                    data = raw
            zf.writestr(new_name, data)

    logger.info(
        "created_rename_zip=%s files=%s format=%s", zip_path.name, len(files), fmt
    )
    return len(files)


def write_rename_zip_from_paths(
    zip_path: Path,
    entries: "list[tuple[str, Path]]",
    *,
    pattern: str,
    start: int = 1,
    padding: int = 0,
    explicit_stems: "list[str] | None" = None,
    output_format: str = "original",
    jpg_quality: int = 90,
) -> int:
    """Like :func:`write_rename_zip` but reads bytes from disk, not memory.

    ``entries`` is a list of ``(original_name, source_path)``. This is the
    large-batch path: nothing holds every image in memory at once. For the
    ``"original"`` format each file is streamed straight from disk into the ZIP
    (``ZipFile.write`` reads in chunks); when re-encoding, only one image is
    decoded at a time. This lets a multi-gigabyte batch be packed with a flat,
    near-constant memory footprint.
    """

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = (output_format or "original").strip().lower()
    save_format = _FORMAT_SPEC.get(fmt, (None, None))[1]

    new_names = plan_output_names(
        [name for name, _ in entries],
        pattern=pattern,
        start=start,
        padding=padding,
        explicit_stems=explicit_stems,
        output_format=fmt,
    )

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for (original_name, src_path), new_name in zip(entries, new_names):
            if save_format is None:
                # Copy verbatim, streamed from disk — never loads the whole file.
                zf.write(src_path, arcname=new_name)
                continue
            try:
                raw = Path(src_path).read_bytes()
                data = _reencode(raw, save_format, jpg_quality=jpg_quality)
                zf.writestr(new_name, data)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "reencode_failed name=%s error=%s — writing original bytes",
                    original_name,
                    str(exc),
                )
                zf.write(src_path, arcname=new_name)

    logger.info(
        "created_rename_zip=%s files=%s format=%s (streamed)",
        zip_path.name,
        len(entries),
        fmt,
    )
    return len(entries)
