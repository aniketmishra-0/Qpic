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
    """

    page_count = doc.page_count
    crops: list[Image.Image] = []
    seg_count = len(question.segments)
    furniture_by_page = furniture_by_page or {}

    for idx, seg in enumerate(question.segments):
        page_index = seg.page - 1
        if page_index < 0 or page_index >= page_count:
            continue

        rect = doc.load_page(page_index).rect
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

        crops.append(
            trim_edge_rules(
                render_page_region(
                    doc,
                    page_index,
                    x_start_pct=x_start,
                    x_end_pct=x_end,
                    y_start_pct=y_start,
                    y_end_pct=y_end,
                    dpi=crop_dpi,
                    furniture_rects=page_furniture,
                )
            )
        )

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
