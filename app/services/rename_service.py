"""Batch image rename utilities.

This powers the standalone "Rename Batch" tool: a user uploads any number of
images, picks a naming pattern, and downloads a ZIP of the renamed files. The
image bytes are never re-encoded — only the filename changes — so every file
keeps its exact original format (a ``.png`` stays a PNG, a ``.jpg`` stays a
JPEG, byte-for-byte).
"""

from __future__ import annotations

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


def write_rename_zip(
    zip_path: Path,
    files: list[tuple[str, bytes]],
    *,
    pattern: str,
    start: int = 1,
    padding: int = 0,
) -> int:
    """Write a ZIP of renamed images and return the count written.

    ``files`` is a list of ``(original_name, raw_bytes)``; the bytes are written
    verbatim under the new name, so the encoded format is untouched.
    """

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    new_names = plan_renames(
        [f[0] for f in files], pattern=pattern, start=start, padding=padding
    )

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for (original_name, raw), new_name in zip(files, new_names):
            zf.writestr(new_name, raw)

    logger.info("created_rename_zip=%s files=%s", zip_path.name, len(files))
    return len(files)
