"""Tier 2 detector: local OCR using Tesseract via pytesseract.

Used for scanned PDFs where text extraction isn't available.
"""

from __future__ import annotations

import logging
from typing import Any
from typing import Optional

import pytesseract
from PIL import Image
from pytesseract import Output

from ...config import Settings
from ...models.schemas import DetectedQuestion
from .base import (
    ContentLine,
    QuestionStart,
    match_question_start_ex,
    match_solution_header,
    starts_to_questions,
)
from .furniture import is_branding_text
from .tesseract_locator import configure_tesseract, resolve_languages

logger = logging.getLogger(__name__)


class OCRDetector:
    def detect(
        self,
        page_images: list[Image.Image],
        settings: Settings,
        render_dpi: Optional[int] = None,
        marker_style: str = "auto",
    ) -> list[DetectedQuestion]:
        """Detect questions using OCR word boxes grouped into lines.

        ``marker_style`` restricts which numbering counts as a question:
        ``"auto"`` (default), ``"q"`` (only "Q1"/"Question 1"), or ``"numbered"``
        (only bare "1."/"2)").
        """

        if not page_images:
            return []

        if not self._is_available():
            return []

        self._marker_style = marker_style
        effective_dpi = int(render_dpi or settings.PDF_RENDER_DPI)

        # Resolve the configured OCR languages (e.g. "eng+hin") down to the packs
        # actually installed, so a Hindi paper is read with the Hindi model when
        # present and we degrade gracefully when it isn't. Reading the body text
        # correctly is what lets a question's crop reach its true bottom instead
        # of stopping after the few lines an English-only model could recognise.
        ocr_lang = resolve_languages(getattr(settings, "OCR_LANGUAGES", "eng") or "eng")
        self._ocr_lang = ocr_lang
        logger.info("ocr_languages using=%s", ocr_lang)

        starts: list[QuestionStart] = []
        content_lines: list[ContentLine] = []
        page_heights: dict[int, float] = {}
        page_widths: dict[int, float] = {}
        # Mean OCR word confidence per page (0-100). The pipeline reads this
        # after detection to decide which weak pages to escalate to the AI tier.
        page_confidence: dict[int, float] = {}

        in_solutions = False
        for page_index, img in enumerate(page_images, start=1):
            processed = self._preprocess_for_ocr(img, render_dpi=effective_dpi)
            page_heights[page_index] = float(processed.height)
            page_widths[page_index] = float(processed.width)

            try:
                data = pytesseract.image_to_data(
                    processed,
                    output_type=Output.DICT,
                    lang=getattr(self, "_ocr_lang", None) or None,
                    config="--oem 3 --psm 6",
                )
            except Exception as exc:
                logger.warning("ocr_page_failed page=%s error=%s", page_index, str(exc))
                page_confidence[page_index] = 0.0
                continue

            page_confidence[page_index] = self._mean_confidence(data)

            page_starts, page_lines, in_solutions = self._ocr_data_to_starts(
                data=data, page_num=page_index, in_solutions=in_solutions
            )
            starts.extend(page_starts)
            content_lines.extend(page_lines)

        # Expose per-page confidence for the pipeline's selective-escalation step.
        self.page_confidence = page_confidence

        # Expose per-page text-line extents in page-percent units
        # ``(y_top, y_bottom, x_left, x_right)`` so the review step's content-
        # coverage check works on scanned PDFs too (the PDF text layer is empty
        # there, so this is the only source of line geometry). Mirrors how
        # ``page_confidence`` is published for the pipeline to read afterwards.
        page_lines_pct: dict[int, list[tuple[float, float, float, float]]] = {}
        for ln in content_lines:
            ph = page_heights.get(ln.page_num) or 0.0
            pw = page_widths.get(ln.page_num) or 0.0
            if ph <= 0 or pw <= 0:
                continue
            page_lines_pct.setdefault(ln.page_num, []).append(
                (
                    (ln.y_top / ph) * 100.0,
                    (ln.y_bottom / ph) * 100.0,
                    (ln.x_left / pw) * 100.0,
                    (ln.x_right / pw) * 100.0,
                )
            )
        self.page_lines_pct = page_lines_pct

        return starts_to_questions(
            starts=starts,
            page_heights=page_heights,
            total_pages=len(page_images),
            content_lines=content_lines,
            page_widths=page_widths,
        )

    @staticmethod
    def _mean_confidence(data: dict[str, Any]) -> float:
        """Mean confidence of real OCR words on a page (0-100).

        Tesseract reports ``conf = -1`` for structural (non-word) rows; those and
        empty tokens are ignored so the average reflects actual recognised text.
        Returns 0.0 when the page produced no words (blank or unreadable).
        """

        texts = data.get("text") or []
        confs = data.get("conf") or []
        vals: list[float] = []
        for i in range(len(texts)):
            if not str(texts[i] or "").strip():
                continue
            try:
                c = float(confs[i])
            except (TypeError, ValueError, IndexError):
                continue
            if c >= 0:
                vals.append(c)
        return (sum(vals) / len(vals)) if vals else 0.0

    def _is_available(self) -> bool:
        """Return True if the tesseract binary is installed."""

        # Point pytesseract at a bundled / installed binary before probing, so
        # the packaged desktop app finds Tesseract even with no PATH set up.
        configure_tesseract()

        try:
            _ = pytesseract.get_tesseract_version()
            return True
        except pytesseract.TesseractNotFoundError:
            logger.warning("tesseract_not_available")
            return False
        except Exception as exc:
            logger.warning("tesseract_check_failed error=%s", str(exc))
            return False

    def _preprocess_for_ocr(self, img: Image.Image, render_dpi: int) -> Image.Image:
        """Improve OCR accuracy.

        1. Convert to grayscale
        2. Resize towards ~300 DPI if rendered below that
        3. Deskew a tilted scan (small rotation) so text rows are horizontal
        4. Denoise speckle
        5. Apply an Otsu binary threshold (adapts to the page's brightness)

        Deskew matters most: a scan tilted even 1-2° smears Tesseract's line
        grouping, so question numbers and option labels get misread or merged —
        a direct cause of dropped questions. Otsu thresholding beats a fixed cut
        on scans that are greyish or unevenly lit. All steps degrade gracefully
        (any failure falls back to the previous simpler behaviour).
        """

        gray = img.convert("L")

        if render_dpi > 0 and render_dpi < 300:
            scale = 300.0 / float(render_dpi)
            if scale > 1.05:
                w, h = gray.size
                gray = gray.resize(
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    resample=Image.Resampling.LANCZOS,
                )

        try:
            import cv2
            import numpy as np

            arr = np.array(gray)
            arr = self._deskew(arr)
            # Light denoise: median blur removes salt-and-pepper scan speckle
            # without softening glyph edges the way a Gaussian would.
            try:
                arr = cv2.medianBlur(arr, 3)
            except Exception:
                pass
            # Otsu picks the threshold from the page's own histogram.
            _, thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return Image.fromarray(thresh).convert("L")
        except Exception:
            # Fallback to simple PIL thresholding.
            return gray.point(lambda p: 0 if p < 128 else 255, mode="L")

    def _deskew(self, arr: "Any") -> "Any":
        """Rotate a grayscale page array to make its text rows horizontal.

        The skew angle is estimated from the minimum-area rectangle enclosing all
        dark (ink) pixels. Only small angles (< ~10°) are corrected — a real exam
        scan is only slightly tilted, and a large "angle" is noise (or a
        legitimately rotated figure) we must not act on. Returns the input
        unchanged on any failure or when the tilt is negligible (< 0.3°).
        """

        try:
            import cv2
            import numpy as np
        except Exception:
            return arr

        try:
            # Ink mask: invert so text is white on black, then find its extent.
            inv = cv2.bitwise_not(arr)
            _, mask = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            coords = cv2.findNonZero(mask)
            if coords is None or len(coords) < 50:
                return arr

            angle = cv2.minAreaRect(coords)[-1]
            # OpenCV reports the angle in [-90, 0); normalise to a small tilt.
            if angle < -45:
                angle = 90.0 + angle
            # Ignore negligible tilt and implausibly large angles (not real skew).
            if abs(angle) < 0.3 or abs(angle) > 10.0:
                return arr

            h, w = arr.shape[:2]
            center = (w / 2.0, h / 2.0)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(
                arr,
                matrix,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
        except Exception:
            return arr


    def _ocr_data_to_starts(
        self, data: dict[str, Any], page_num: int, in_solutions: bool = False
    ) -> tuple[list[QuestionStart], list[ContentLine], bool]:
        """Reconstruct text lines from Tesseract word boxes and find markers.

        Words are grouped using Tesseract's own structural keys
        (block/paragraph/line) and ordered within a line by ``word_num``.
        Sorting words by pixel ``top`` scrambles words that share a line (their
        tops differ by more than a few pixels at high resolution), which breaks
        marker matching. Using the structural keys keeps word order intact and
        yields an accurate per-line bounding box for tight cropping.
        """

        texts = data.get("text") or []
        tops = data.get("top") or []
        lefts = data.get("left") or []
        heights = data.get("height") or []
        widths = data.get("width") or []
        confs = data.get("conf") or []
        block_nums = data.get("block_num") or []
        par_nums = data.get("par_num") or []
        line_nums = data.get("line_num") or []
        word_nums = data.get("word_num") or []

        n = len(texts)

        def _int_at(seq: Any, i: int, default: int) -> int:
            try:
                return int(seq[i])
            except Exception:
                return default

        # Group words by (block, paragraph, line).
        grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}

        for i in range(n):
            text = str(texts[i] or "").strip()
            if not text:
                continue

            conf_raw = confs[i] if i < len(confs) else None
            try:
                conf = float(conf_raw) if conf_raw is not None else 0.0
            except Exception:
                conf = 0.0
            if conf < 0:
                continue

            try:
                top = int(tops[i])
                left = int(lefts[i])
            except Exception:
                continue

            height = _int_at(heights, i, 0)
            width = _int_at(widths, i, 0)

            key = (
                _int_at(block_nums, i, 0),
                _int_at(par_nums, i, 0),
                _int_at(line_nums, i, 0),
            )
            grouped.setdefault(key, []).append(
                {
                    "text": text,
                    "top": top,
                    "left": left,
                    "height": height,
                    "width": width,
                    "word_num": _int_at(word_nums, i, left),
                }
            )

        starts: list[QuestionStart] = []
        content_lines: list[ContentLine] = []
        solutions_started = in_solutions

        # Emit lines in reading order (top-to-bottom, then left-to-right).
        ordered_keys = sorted(
            grouped.keys(),
            key=lambda k: (min(w["top"] for w in grouped[k]), k),
        )

        for key in ordered_keys:
            words = sorted(grouped[key], key=lambda w: (w["word_num"], w["left"]))
            line_text = " ".join(w["text"] for w in words)

            line_top = min(w["top"] for w in words)
            line_bottom = max(w["top"] + max(w["height"], 0) for w in words)
            if line_bottom <= line_top:
                line_bottom = line_top + 1

            line_left = min(w["left"] for w in words)
            line_right = max(w["left"] + max(w["width"], 0) for w in words)
            if line_right <= line_left:
                line_right = line_left + 1

            # Once we cross a "Solutions"/"Answer Key" header, every numbered
            # item after it is a solution, not a question. The header line
            # itself is not real content, so we skip it (otherwise the preceding
            # question's crop would bleed down into the solutions header).
            if not solutions_started and match_solution_header(line_text):
                solutions_started = True
                continue

            # Drop app/website branding ("Android App | iOS App | PW Website")
            # so the footer never extends a question's content region. On scanned
            # PDFs there are no hyperlinks to key off, so we match the phrase.
            if is_branding_text(line_text):
                continue

            content_lines.append(
                ContentLine(
                    page_num=page_num,
                    y_top=float(line_top),
                    y_bottom=float(line_bottom),
                    x_left=float(line_left),
                    x_right=float(line_right),
                    text=line_text,
                )
            )

            q_info = match_question_start_ex(line_text, getattr(self, "_marker_style", "auto"))
            if q_info is not None:
                q_num, is_strong = q_info
                starts.append(
                    QuestionStart(
                        page_num=page_num,
                        y_top=float(line_top),
                        q_num=q_num,
                        is_solution=solutions_started,
                        x_left=float(line_left),
                        x_right=float(line_right),
                        is_strong=is_strong,
                    )
                )

        return starts, content_lines, solutions_started
