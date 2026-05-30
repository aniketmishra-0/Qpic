"""Tier 1 detector: PyMuPDF text extraction (fast, free).

Designed for searchable/text PDFs.
"""

from __future__ import annotations

import logging
from typing import NamedTuple
from typing import Optional

import fitz

from ...models.schemas import DetectedQuestion
from .base import (
    ContentLine,
    QuestionStart,
    match_question_start_ex,
    match_solution_header,
    starts_to_questions,
)
from .figure_detector import extract_figures_for_page, filter_watermark_figures
from .furniture import (
    branding_link_bands,
    is_branding_text,
    is_margin_furniture_text,
)

logger = logging.getLogger(__name__)


class TextBlock(NamedTuple):
    page_num: int  # 1-indexed
    y_top: float
    y_bottom: float
    page_height: float
    q_num: str


class TextDetector:
    def detect(
        self, pdf_bytes: bytes, padding_px: int = 0, marker_style: str = "auto"
    ) -> list[DetectedQuestion]:
        """Detect question start positions using extracted text.

        padding_px is accepted for API compatibility but is applied at crop-time,
        not during detection.

        ``marker_style`` restricts which numbering counts as a question:
        ``"auto"`` (default), ``"q"`` (only "Q1"/"Question 1"), or ``"numbered"``
        (only bare "1."/"2)"). Forcing a style stops sub-statements, option
        labels and equation numbers from being mistaken for questions.
        """

        self._marker_style = marker_style
        if not pdf_bytes:
            return []

        starts: list[QuestionStart] = []
        content_lines: list[ContentLine] = []
        figures: list = []
        page_heights: dict[int, float] = {}
        page_widths: dict[int, float] = {}

        try:
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                total_pages = doc.page_count
                in_solutions = False
                for page_idx in range(total_pages):
                    page = doc.load_page(page_idx)
                    page_num = page_idx + 1
                    page_heights[page_num] = float(page.rect.height)
                    page_widths[page_num] = float(page.rect.width)

                    # Pull diagrams/graphs/embedded images on this page so a
                    # question's crop grows to contain its figure instead of
                    # clipping a wide diagram or dropping a trailing one.
                    try:
                        figures.extend(extract_figures_for_page(page, page_num))
                    except Exception as exc:
                        logger.debug(
                            "figure_extract_failed page=%s error=%s", page_num, str(exc)
                        )

                    # Vertical bands of branding hyperlinks (app store / PW
                    # website). Any text line overlapping one is page furniture
                    # and must not bound a crop, even mid-page on a short card.
                    link_bands = branding_link_bands(page)

                    blocks = self._get_text_blocks_for_page(page)
                    for block in blocks:
                        y_top = float(block.get("y_top", 0.0))
                        y_bottom = float(block.get("y_bottom", y_top))
                        x_left = float(block.get("x_left", 0.0))
                        x_right = float(block.get("x_right", x_left))
                        text = block.get("text", "")

                        # Drop branding furniture (matched by phrase or by an
                        # overlapping branding hyperlink) so the footer never
                        # extends a question's content region.
                        if is_branding_text(text) or any(
                            y_top < b1 and y_bottom > b0 for b0, b1 in link_bands
                        ):
                            continue

                        # Drop margin furniture: PW print-preview URL, page
                        # numbers ("4/5") and short running titles ("UPSC")
                        # sitting in the top/bottom margin. Position-guarded so
                        # body content is never touched.
                        page_h = page_heights[page_num]
                        if page_h > 0 and is_margin_furniture_text(
                            text,
                            top_pct=(y_top / page_h) * 100.0,
                            bottom_pct=(y_bottom / page_h) * 100.0,
                        ):
                            continue

                        # Once we cross a "Solutions"/"Answer Key" header, every
                        # numbered item after it is a solution, not a question.
                        # The header line itself is not real content, so we skip
                        # it (otherwise the preceding question's crop would bleed
                        # down into the solutions header).
                        if not in_solutions and match_solution_header(text):
                            in_solutions = True
                            continue

                        content_lines.append(
                            ContentLine(
                                page_num=page_num,
                                y_top=y_top,
                                y_bottom=y_bottom,
                                x_left=x_left,
                                x_right=x_right,
                                text=text,
                            )
                        )

                        q_info = self._match_question_start(text)
                        if q_info is None:
                            continue
                        q_num, is_strong = q_info
                        starts.append(
                            QuestionStart(
                                page_num=page_num,
                                y_top=y_top,
                                q_num=q_num,
                                is_solution=in_solutions,
                                x_left=x_left,
                                x_right=x_right,
                                is_strong=is_strong,
                            )
                            )  # Closing the parentheses for starts.append

                return starts_to_questions(
                    starts=starts,
                    page_heights=page_heights,
                    total_pages=total_pages,
                    content_lines=content_lines,
                    page_widths=page_widths,
                    figures=filter_watermark_figures(
                        figures,
                        {
                            p: (page_widths.get(p, 0.0), page_heights.get(p, 0.0))
                            for p in page_heights
                        },
                    ),
                )
        except Exception as exc:
            logger.warning("text_detector_failed error=%s", str(exc))
            return []

    def _get_text_blocks_for_page(self, page: fitz.Page) -> list[dict]:
        """Return list of dicts: {text, y_top, y_bottom, x_left}.

        Uses page.get_text("dict")["blocks"] → iterate lines → spans.
        """

        out: list[dict] = []
        text_dict = page.get_text("dict")

        for block in text_dict.get("blocks", []) or []:
            if not isinstance(block, dict):
                continue
            # PyMuPDF: type==0 is text block.
            if block.get("type") not in (None, 0):
                continue

            for line in block.get("lines", []) or []:
                if not isinstance(line, dict):
                    continue
                spans = line.get("spans", []) or []
                pieces: list[str] = []
                for span in spans:
                    if isinstance(span, dict):
                        pieces.append(str(span.get("text", "")))
                text = "".join(pieces).strip()
                if not text:
                    continue

                bbox = line.get("bbox") or block.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue

                x0, y0, x1, y1 = bbox
                out.append(
                    {
                        "text": text,
                        "y_top": float(y0),
                        "y_bottom": float(y1),
                        "x_left": float(x0),
                        "x_right": float(x1),
                    }
                )

        out.sort(key=lambda b: (b.get("y_top", 0.0), b.get("x_left", 0.0)))
        return out

    def _match_question_start(self, text: str) -> Optional[tuple[str, bool]]:
        return match_question_start_ex(text, getattr(self, "_marker_style", "auto"))
