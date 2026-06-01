"""Crop and stitching logic for detected questions."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import fitz
from PIL import Image

from ..models.schemas import DetectedQuestion
from ..services.pdf_service import render_page_region
from ..utils.image_utils import ensure_rgb, trim_edge_rules

logger = logging.getLogger(__name__)

# A pixel counts as content ("ink") when its darkest channel falls below this.
# Matches the level used by the edge-rule trimmer so alignment sees the same
# glyphs.
_ALIGN_INK_LEVEL = 235


def _align_left_by_content(images: list[Image.Image]) -> list[Image.Image]:
    """Left-align stitched parts on their first inked column.

    Retained as a fallback aligner: trims each crop's *extra* leading whitespace
    so every part begins its content at the same x, keeping the smallest existing
    margin. Used only when option-label alignment can't find a reference (e.g. a
    scanned page with no text layer).
    """

    try:
        import numpy as np
    except Exception:
        return images

    lefts: list[int] = []
    for img in images:
        arr = np.asarray(ensure_rgb(img))
        if arr.ndim != 3 or arr.shape[0] < 1 or arr.shape[1] < 1:
            lefts.append(0)
            continue
        nonwhite = arr.min(axis=2) < _ALIGN_INK_LEVEL
        cols = np.where(nonwhite.any(axis=0))[0]
        lefts.append(int(cols.min()) if cols.size else 0)

    if not lefts:
        return images

    margin = min(lefts)
    aligned: list[Image.Image] = []
    for img, left in zip(images, lefts):
        cut = left - margin
        if 0 < cut < img.size[0]:
            aligned.append(img.crop((cut, 0, img.size[0], img.size[1])))
        else:
            aligned.append(img)
    return aligned


def _row_ink(arr) -> "tuple":
    """Return a boolean per-row 'has ink' mask for an RGB numpy image."""

    import numpy as np

    nonwhite = arr.min(axis=2) < _ALIGN_INK_LEVEL
    return nonwhite.any(axis=1)


def _content_rows(row_has) -> "tuple[int, int] | None":
    """First/last inked row index, or None when the strip is blank."""

    import numpy as np

    idx = np.where(row_has)[0]
    if idx.size == 0:
        return None
    return int(idx[0]), int(idx[-1])


def _median_line_gap(row_has) -> "int | None":
    """Median whitespace gap (in rows) between text lines within a part.

    Used as the target gap when butting stitched parts together so the join
    between e.g. "(B)" and "(C)" matches the paper's own line rhythm instead of
    leaving the stacked snap margins.
    """

    import numpy as np

    # Interior blank runs are the gaps strictly between consecutive inked rows.
    # Leading/trailing whitespace is ignored (it has no inked row on one side),
    # matching the original row-by-row scan — but in a single vectorized pass.
    inked = np.flatnonzero(np.asarray(row_has))
    if inked.size < 2:
        return None
    gaps = np.diff(inked) - 1
    gaps = gaps[gaps > 0]
    if gaps.size == 0:
        return None
    return int(np.median(gaps))


def _stitch_with_left_pads(crops: list[Image.Image], pads_px: list[int]) -> Image.Image:
    """Stack crops top-to-bottom, left-padding each by ``pads_px[i]`` white px.

    The per-part left pad shifts a part to the right so a chosen reference column
    (the MCQ option labels) lines up across parts. Pads are never negative, so no
    content is ever cut.

    Internal joins are tightened: the trailing whitespace of every non-last part
    and the leading whitespace of every non-first part are trimmed, then a single
    uniform gap (the median in-part line gap) is inserted between parts. This
    stops the snap margins of two boxes from stacking into an oversized gap (the
    "extra space between (B) and (C)" case) while keeping the question's natural
    line spacing. The outer top/bottom margins are preserved.
    """

    if not crops:
        return Image.new("RGB", (1, 1), (255, 255, 255))

    parts = [ensure_rgb(img) for img in crops]
    pads = [max(0, int(p)) for p in pads_px] + [0] * max(0, len(parts) - len(pads_px))

    try:
        import numpy as np
    except Exception:
        np = None  # type: ignore

    if np is None or len(parts) == 1:
        max_width = max(img.size[0] + pads[i] for i, img in enumerate(parts))
        total_height = sum(img.size[1] for img in parts)
        stitched = Image.new("RGB", (max_width, total_height), (255, 255, 255))
        y = 0
        for i, img in enumerate(parts):
            stitched.paste(img, (pads[i], y))
            y += img.size[1]
        return stitched

    masks = [_row_ink(np.asarray(p)) for p in parts]
    bounds = [_content_rows(m) for m in masks]

    # Target inter-part gap: the median line gap seen inside the parts, so the
    # join reads like a normal line break. Fall back to a small fraction of the
    # first part's height when no in-part gap can be measured.
    gaps = [g for g in (_median_line_gap(m) for m in masks) if g is not None]
    if gaps:
        target_gap = int(np.median(np.asarray(gaps)))
    else:
        target_gap = max(2, int(0.02 * parts[0].size[1]))

    # Per-part vertical crop window: keep the outer margins (top of first part,
    # bottom of last part) but trim the inner-join whitespace to the content.
    windows: list[tuple[int, int]] = []
    for i, (p, b) in enumerate(zip(parts, bounds)):
        ph = p.size[1]
        if b is None:
            windows.append((0, ph))
            continue
        top = 0 if i == 0 else b[0]
        bot = ph if i == len(parts) - 1 else b[1] + 1
        windows.append((max(0, top), min(ph, bot)))

    cropped = [p.crop((0, w0, p.size[0], w1)) for p, (w0, w1) in zip(parts, windows)]

    max_width = max(img.size[0] + pads[i] for i, img in enumerate(cropped))
    total_height = sum(img.size[1] for img in cropped) + target_gap * (len(cropped) - 1)
    stitched = Image.new("RGB", (max_width, total_height), (255, 255, 255))

    y = 0
    for i, img in enumerate(cropped):
        stitched.paste(img, (pads[i], y))
        y += img.size[1]
        if i != len(cropped) - 1:
            y += target_gap

    return stitched


# Match a leading MCQ option label at the start of a line: "(A)", "A)", "A.".
_OPTION_LINE_RE = re.compile(r"^\s*\(?\s*([A-Da-d])\s*[\.\)]")


def _segment_option_x_pts(
    page: "fitz.Page",
    *,
    y0_pts: float,
    y1_pts: float,
    x0_pts: float,
    x1_pts: float,
) -> "float | None":
    """Return the leftmost x (PDF points) of an MCQ option label in a region.

    Scans the page text layer for lines inside the segment's box that begin with
    an option marker ("(A)", "B)", "C.") and returns the smallest line-left x —
    i.e. the x at which the option column starts. Returns ``None`` for scanned
    pages (no text) or segments with no option line, so the caller can fall back
    to a plain flush-left stitch.
    """

    try:
        data = page.get_text("dict") or {}
    except Exception:
        return None

    best: "float | None" = None
    for block in data.get("blocks", []):
        for ln in block.get("lines", []):
            spans = ln.get("spans", [])
            if not spans:
                continue
            text = "".join(str(s.get("text", "")) for s in spans).strip()
            if not _OPTION_LINE_RE.match(text):
                continue
            bx0, by0, bx1, by1 = ln.get("bbox", (0, 0, 0, 0))
            cy = (by0 + by1) / 2.0
            cx = (bx0 + bx1) / 2.0
            # Line must sit inside the segment's region.
            if cy < y0_pts or cy > y1_pts:
                continue
            if cx < x0_pts or cx > x1_pts:
                continue
            if best is None or bx0 < best:
                best = float(bx0)
    return best


def _stitch_vertical(crops: list[Image.Image], *, align_parts: bool = False) -> Image.Image:
    """Stack crops top-to-bottom into one image (plain flush-left).

    ``align_parts`` is accepted for signature compatibility; option-label
    alignment for the manual column-split case is handled in
    :func:`crop_and_stitch_hires` (which has the PDF geometry needed to locate
    the option column). Here we always paste flush-left.
    """

    if not crops:
        return Image.new("RGB", (1, 1), (255, 255, 255))

    if len(crops) == 1:
        return ensure_rgb(crops[0])

    parts = [ensure_rgb(img) for img in crops]
    max_width = max(img.size[0] for img in parts)
    total_height = sum(img.size[1] for img in parts)
    stitched = Image.new("RGB", (max_width, total_height), (255, 255, 255))

    y_offset = 0
    for img in parts:
        stitched.paste(img, (0, y_offset))
        y_offset += img.size[1]

    return stitched


def crop_question(
    page_images: list[Image.Image],
    question: DetectedQuestion,
    padding_px: int,
) -> Image.Image:
    """Backward-compatible alias for crop_and_stitch."""

    return crop_and_stitch(page_images=page_images, question=question, padding_px=padding_px)


def crop_and_stitch(
    page_images: list[Image.Image],
    question: DetectedQuestion,
    padding_px: int,
) -> Image.Image:
    """Crop and stitch a single question from one or more pages.

    For each segment in question.segments:
      - Map y_start_pct / y_end_pct → pixel coords on that page
      - Clamp with padding_px (do not exceed image bounds)
      - Crop full width (0 to page_width)

    If single segment → return that crop directly.

    If multiple segments (cross-page): stitch all crops vertically.
    """

    crops: list[Image.Image] = []
    seg_count = len(question.segments)
    for idx, seg in enumerate(question.segments):
        page_index = seg.page - 1
        if page_index < 0 or page_index >= len(page_images):
            continue

        page_img = ensure_rgb(page_images[page_index])
        width, height = page_img.size

        y0 = int((seg.y_start_pct / 100.0) * height)
        y1 = int((seg.y_end_pct / 100.0) * height)
        if y1 <= y0:
            continue

        # Horizontal extent of the column this segment belongs to. Defaults
        # (0..100) reproduce the previous full-width behavior for single-column
        # pages; two-column pages confine the crop to the correct column.
        x0 = int((getattr(seg, "x_start_pct", 0.0) / 100.0) * width)
        x1 = int((getattr(seg, "x_end_pct", 100.0) / 100.0) * width)
        if x1 <= x0:
            x0, x1 = 0, width

        # Apply padding only on the OUTER edges of the question so cross-page
        # segments join seamlessly (no double margin at the page break).
        pad_top = padding_px if idx == 0 else 0
        pad_bottom = padding_px if idx == seg_count - 1 else 0

        y0 = max(0, y0 - pad_top)
        y1 = min(height, y1 + pad_bottom)

        # Pad the column horizontally too, clamped to the page bounds, so glyphs
        # at the column edge aren't clipped.
        x0 = max(0, x0 - padding_px)
        x1 = min(width, x1 + padding_px)
        crops.append(trim_edge_rules(page_img.crop((x0, y0, x1, y1))))

    if not crops:
        return Image.new("RGB", (1, 1), (255, 255, 255))

    if len(crops) == 1:
        return ensure_rgb(crops[0])

    max_width = max(img.size[0] for img in crops)
    total_height = sum(img.size[1] for img in crops)
    stitched = Image.new("RGB", (max_width, total_height), (255, 255, 255))

    y_offset = 0
    for img in crops:
        rgb = ensure_rgb(img)
        stitched.paste(rgb, (0, y_offset))
        y_offset += rgb.size[1]

    return stitched


def crop_and_stitch_hires(
    doc: "fitz.Document",
    question: DetectedQuestion,
    padding_px: int,
    *,
    detection_dpi: int,
    crop_dpi: int,
    furniture_by_page: "dict | None" = None,
    align_parts: bool = False,
) -> Image.Image:
    """Render a question's crop straight from the PDF vector source.

    This mirrors :func:`crop_and_stitch` (same segments, same padding, same
    vertical stitching for cross-page questions) but renders each region from
    the PDF at ``crop_dpi`` instead of cropping the already-rasterized detection
    images. The result stays crisp when zoomed because the text is rasterized at
    a high DPI rather than upscaled from the detection render.

    ``padding_px`` is expressed in detection-render pixels (the same unit the
    raster cropper uses); it is converted to a percentage of each page so the
    padded region matches regardless of the DPI we ultimately render at.

    ``furniture_by_page`` maps a 1-indexed page number to a list of furniture
    rectangles (``FurnitureRect`` or ``(x0, y0, x1, y1)`` in PDF points). Any
    furniture intersecting a rendered region is painted white, so a branding
    footer, logo or decorative rule that falls inside a crop (including the
    middle of a cross-page stitch) is removed from the output pixels.

    ``align_parts`` left-aligns the stitched parts on their content edge. It is
    off by default so the fully-automatic ``/crop`` flow is unchanged; the
    manual review/``finalize`` flow turns it on so a question rebuilt from
    column-split boxes (stem + spilled options) lines up cleanly instead of
    looking shifted.
    """

    page_count = doc.page_count
    crops: list[Image.Image] = []
    seg_count = len(question.segments)
    furniture_by_page = furniture_by_page or {}
    zoom = float(crop_dpi) / 72.0

    # Per-segment data for option-label alignment (manual column-split case).
    opt_pixel_offsets: list["float | None"] = []

    for idx, seg in enumerate(question.segments):
        page_index = seg.page - 1
        if page_index < 0 or page_index >= page_count:
            continue

        page = doc.load_page(page_index)
        rect = page.rect
        page_w_pts = float(rect.width)
        page_h_pts = float(rect.height)
        if page_w_pts <= 0 or page_h_pts <= 0:
            continue

        # Detection pixels -> points -> percentage of the page, so the padding
        # matches what the raster cropper would have produced.
        pad_pts = padding_px * 72.0 / float(detection_dpi or 72)
        pad_x_pct = (pad_pts / page_w_pts) * 100.0
        pad_y_pct = (pad_pts / page_h_pts) * 100.0

        # Padding only on the OUTER edges so cross-page segments join seamlessly.
        pad_top = pad_y_pct if idx == 0 else 0.0
        pad_bottom = pad_y_pct if idx == seg_count - 1 else 0.0

        y_start = max(0.0, seg.y_start_pct - pad_top)
        y_end = min(100.0, seg.y_end_pct + pad_bottom)
        if y_end <= y_start:
            continue

        x_start = getattr(seg, "x_start_pct", 0.0)
        x_end = getattr(seg, "x_end_pct", 100.0)
        if x_end <= x_start:
            x_start, x_end = 0.0, 100.0
        x_start = max(0.0, x_start - pad_x_pct)
        x_end = min(100.0, x_end + pad_x_pct)

        page_furniture = [
            (fr[1], fr[2], fr[3], fr[4]) if len(fr) == 5 else tuple(fr)
            for fr in furniture_by_page.get(seg.page, [])
        ]

        # Region bounds in PDF points (the crop's pixel x=0 maps to x0_pts).
        x0_pts = rect.x0 + (x_start / 100.0) * page_w_pts
        x1_pts = rect.x0 + (x_end / 100.0) * page_w_pts
        y0_pts = rect.y0 + (y_start / 100.0) * page_h_pts
        y1_pts = rect.y0 + (y_end / 100.0) * page_h_pts

        rendered = render_page_region(
            doc,
            page_index,
            x_start_pct=x_start,
            x_end_pct=x_end,
            y_start_pct=y_start,
            y_end_pct=y_end,
            dpi=crop_dpi,
            furniture_rects=page_furniture,
        )

        # Pixel x of this part's MCQ option column (relative to the rendered
        # crop's left edge). Used only by the aligned manual path.
        opt_x_pts = _segment_option_x_pts(
            page, y0_pts=y0_pts, y1_pts=y1_pts, x0_pts=x0_pts, x1_pts=x1_pts
        )
        opt_pixel_offsets.append(
            (opt_x_pts - x0_pts) * zoom if opt_x_pts is not None else None
        )

        if align_parts:
            # Defer edge-rule trimming to the final stitched image so the option
            # pixel offsets above stay valid (a per-part left trim would shift
            # them out of sync).
            crops.append(ensure_rgb(rendered))
        else:
            crops.append(trim_edge_rules(rendered))

    if not align_parts or len(crops) <= 1:
        return _stitch_vertical(crops, align_parts=False)

    # Align the parts so every MCQ option label ("(A)".."(D)") starts at the same
    # x. The reference is the rightmost option column among the parts; each other
    # part is left-padded to push its options under it. Parts without a detected
    # option line (e.g. a pure stem fragment) are left flush-left.
    known = [o for o in opt_pixel_offsets if o is not None]
    if len(known) >= 2:
        ref = max(known)
        pads = [
            int(round(ref - o)) if o is not None else 0 for o in opt_pixel_offsets
        ]
        stitched = _stitch_with_left_pads(crops, pads)
        return trim_edge_rules(stitched)

    # No reliable option columns to align on (e.g. a scanned page) — fall back to
    # the content-edge aligner, then a flush-left stitch.
    trimmed = [trim_edge_rules(c) for c in crops]
    return _stitch_vertical(_align_left_by_content(trimmed), align_parts=False)


def save_question_image(
    image: Image.Image,
    q_num: str,
    output_dir: Path,
    is_solution: bool = False,
    *,
    question_prefix: str = "Q",
    solution_prefix: str = "S",
    start_number: int = 1,
    image_format: str = "png",
    jpg_quality: int = 90,
) -> Path:
    """Save a cropped question (or solution) image in the job directory.

    Filenames are ``<prefix><number>.<ext>`` zero-padded to 3 digits, e.g.
    ``Q001.png`` for questions and ``S001.png`` for solutions, so a question and
    its solution sharing a number don't overwrite each other.

    Naming is configurable:
      - ``question_prefix`` / ``solution_prefix`` set the leading letters.
      - ``start_number`` shifts numbering so the first item can begin at any
        value (e.g. start at 11 -> ``Q011.png``). The detected number is offset
        by ``start_number - 1`` so a paper numbered 1..N is renumbered onto the
        user's chosen starting point while preserving gaps/order.

    Output encoding is configurable:
      - ``image_format`` is ``"png"`` (lossless) or ``"jpg"`` (lossy, smaller).
      - ``jpg_quality`` (1-100) controls JPG compression; ignored for PNG.
        Lower values trade visual quality for a smaller file.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    digits = re.findall(r"\d+", q_num)
    detected_number = int(digits[0]) if digits else 0
    number = detected_number + (start_number - 1)
    if number < 0:
        number = 0
    prefix = solution_prefix if is_solution else question_prefix

    fmt = (image_format or "png").strip().lower()
    if fmt in ("jpg", "jpeg"):
        ext = "jpg"
        filename = f"{prefix}{number:03d}.{ext}"
        out_path = output_dir / filename
        quality = max(1, min(100, int(jpg_quality)))
        ensure_rgb(image).save(out_path, format="JPEG", quality=quality, optimize=True)
    else:
        filename = f"{prefix}{number:03d}.png"
        out_path = output_dir / filename
        ensure_rgb(image).save(out_path, format="PNG")

    logger.info("saved_image=%s q_num=%s is_solution=%s", out_path.name, q_num, is_solution)
    return out_path
