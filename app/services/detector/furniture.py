"""Structural page-furniture detection for searchable PDFs.

Text-repetition heuristics are fragile: a branding footer that sits *mid-page*
on a short solution card (PW "Android App | iOS App | PW Website"), or a
decorative accent rule drawn as a vector, are easily missed and then get
stitched into the middle of a crop. This module instead reads the PDF's own
*structure* — hyperlink rectangles, embedded logo images and thin vector
rules — which identifies that furniture exactly rather than by guessing.

Two consumers use it:

  * the text detector, to keep furniture out of a question's content bounds so
    a crop is never *sized* to include the branding footer; and
  * the hi-res renderer, to paint over any furniture rectangle that still falls
    inside a crop region (e.g. a footer in the middle of a cross-page stitch),
    so it is physically removed from the output pixels.

Everything here is expressed in PDF points so it can be mapped to either the
detection raster or the hi-res crop render.
"""

from __future__ import annotations

import logging
import re
from typing import NamedTuple

import fitz

logger = logging.getLogger(__name__)


class FurnitureRect(NamedTuple):
    """A furniture region on a page, in PDF points (x0, y0, x1, y1)."""

    page_num: int  # 1-indexed
    x0: float
    y0: float
    x1: float
    y1: float


# Hosts that mark a hyperlink as app/website branding rather than real content.
_BRANDING_URI_HOSTS: tuple[str, ...] = (
    "play.google.com",
    "apps.apple.com",
    "itunes.apple.com",
    "pw.live",
    "penpencil.co",
    "physicswallah",
    "onelink.me",
    "bit.ly",
    "linktr.ee",
)

# Branding phrases that frequently sit next to the link icons.
_BRANDING_TEXT = re.compile(
    r"(android\s*app|ios\s*app|pw\s*website|download\s*app|play\s*store|app\s*store)",
    re.IGNORECASE,
)

# Plain-text furniture (not hyperlinked) that PW prints in the page margins:
#   * the print-preview/admin URL strip, and
#   * any URL on a host we already treat as branding.
# These are matched as *text* because they are usually not real <a> links.
_FURNITURE_TEXT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"penpencil\.co", re.IGNORECASE),
    re.compile(r"qbg-admin", re.IGNORECASE),
    re.compile(r"print-preview", re.IGNORECASE),
    re.compile(r"\bpw\.live\b", re.IGNORECASE),
    re.compile(r"physicswallah", re.IGNORECASE),
)

# A page-number strip such as "4/5" or "Page 4 of 5". Only treated as furniture
# when it also sits in a margin band (see ``is_margin_furniture_text``), so a
# real number in the body is never removed. A *bare* number ("12") is
# deliberately NOT matched here — it is too easily a real answer/option — bare
# page numbers are caught instead by the repeat-position furniture detector.
_PAGE_NUMBER_TEXT = re.compile(
    r"^\s*(?:page\s*)?\d{1,3}\s*(?:/|of)\s*\d{1,3}\s*$",
    re.IGNORECASE,
)

# Margin bands (% of page height) where running headers/footers live. A
# page-number strip here is stripped; the same text mid-page is left alone.
_MARGIN_TOP_PCT = 8.0
_MARGIN_BOTTOM_PCT = 80.0

# A vector drawing counts as a decorative rule when it is thin in one axis and
# long in the other.
_RULE_MAX_THICK_FRAC = 0.04   # <=4% of the page dimension thick
_RULE_MIN_LONG_FRAC = 0.18    # >=18% of the page dimension long

# Padding (in points) added around each furniture rect when painting it out, so
# anti-aliased edges of a rule/branding glyph are fully covered.
_PAINT_PAD_PTS = 2.0

# --- Content note-boxes ------------------------------------------------------
#
# Exam papers frame asides such as "Additional Information", "PW ONLYIAS SUPER
# HINT" or "Extra Edge" in a thin rectangular border that *encloses real body
# text*. Unlike a decorative side rule, this border is part of the content and
# must be KEPT (never painted out), and the crop must grow to contain the whole
# frame so the boxed text is never sliced at the box's bottom edge.
#
# A frame is drawn as four thin strokes (two vertical sides + a top/bottom
# horizontal). We recognise it by a pair of vertical border strokes that share a
# vertical extent, are separated horizontally, and are joined by at least one
# horizontal stroke spanning them at the top or bottom. This is specific enough
# that two unrelated rules (e.g. a column divider + an answer underline) are
# never mistaken for a box.

# A border stroke is at most this thick (points), or this fraction of the page.
_BOX_BORDER_MAX_THICK_PTS = 3.0
_BOX_BORDER_MAX_THICK_FRAC = 0.01

# A frame must be at least this wide / tall (fraction of the page) to count.
_BOX_MIN_WIDTH_FRAC = 0.10
_BOX_MIN_HEIGHT_FRAC = 0.03

# How closely the two sides' corners must align (points) to read as one frame.
_BOX_EDGE_ALIGN_TOL_PTS = 6.0


def _thin_border_segments(
    page: "fitz.Page",
) -> "tuple[list[fitz.Rect], list[fitz.Rect]]":
    """Return ``(horizontal, vertical)`` thin border strokes on a page.

    A horizontal stroke is thin in height and reasonably wide; a vertical stroke
    is thin in width and reasonably tall. These are the candidate edges of a
    content note-box frame.
    """

    rect = page.rect
    page_w = float(rect.width)
    page_h = float(rect.height)
    max_thick_w = max(_BOX_BORDER_MAX_THICK_PTS, _BOX_BORDER_MAX_THICK_FRAC * page_w)
    max_thick_h = max(_BOX_BORDER_MAX_THICK_PTS, _BOX_BORDER_MAX_THICK_FRAC * page_h)
    min_w = _BOX_MIN_WIDTH_FRAC * page_w
    min_h = _BOX_MIN_HEIGHT_FRAC * page_h

    h_segs: list[fitz.Rect] = []
    v_segs: list[fitz.Rect] = []
    try:
        drawings = page.get_drawings()
    except Exception:  # pragma: no cover - defensive
        return h_segs, v_segs

    for d in drawings or []:
        r = d.get("rect") if isinstance(d, dict) else None
        if r is None:
            continue
        r = fitz.Rect(r)
        w = float(r.width)
        h = float(r.height)
        # A border stroke may render as a hairline (one dimension == 0) or as a
        # thin filled rect (a 1-2 pt bar). Both are valid box edges.
        if w < 0 or h < 0:
            continue
        if h <= max_thick_h and w >= min_w:
            h_segs.append(r)
        elif w <= max_thick_w and h >= min_h:
            v_segs.append(r)
    return h_segs, v_segs


def _rect_frame_boxes(
    page: "fitz.Page", page_w: float, page_h: float, min_w: float, min_h: float
) -> "list[fitz.Rect]":
    """Frames drawn as a single rectangle path rather than four strokes.

    Many papers (PW "ONLYIAS SUPER HINT", "Extra Edge") draw the note-box border
    as one stroked rectangle, which PyMuPDF reports as a single drawing whose
    ``items`` hold one ``"re"`` (rectangle) op spanning the whole frame. The
    four-strokes path above never sees two separate vertical segments for these,
    so without this they go undetected and the crop slices the box's bottom.
    """

    out: list[fitz.Rect] = []
    try:
        drawings = page.get_drawings()
    except Exception:  # pragma: no cover - defensive
        return out

    for d in drawings or []:
        if not isinstance(d, dict):
            continue
        # Only a *stroked* rectangle is a frame; a pure fill could be a colour
        # highlight behind an answer, not a border.
        if d.get("type") not in ("s", "fs"):
            continue
        for it in d.get("items") or []:
            if not it or it[0] != "re":
                continue
            try:
                r = fitz.Rect(it[1])
            except Exception:
                continue
            if r.width >= min_w and r.height >= min_h:
                out.append(fitz.Rect(r.x0, r.y0, r.x1, r.y1))
    return out


def detect_content_boxes(page: "fitz.Page") -> "list[fitz.Rect]":
    """Return rectangles (points) of bordered content note-boxes on a page.

    Two framing styles are recognised:
      * four separate strokes — two vertical border strokes of matching vertical
        extent, separated horizontally, joined by a horizontal stroke spanning
        them at the top or the bottom; and
      * a single stroked rectangle path covering the whole frame (how PW
        "ONLYIAS SUPER HINT" / "Extra Edge" boxes are usually drawn).

    Returns the outer frame rectangles, de-duplicated.
    """

    rect = page.rect
    page_w = float(rect.width)
    page_h = float(rect.height)
    if page_w <= 0 or page_h <= 0:
        return []

    tol = _BOX_EDGE_ALIGN_TOL_PTS
    min_w = _BOX_MIN_WIDTH_FRAC * page_w
    min_h = _BOX_MIN_HEIGHT_FRAC * page_h

    boxes: list[fitz.Rect] = []

    # --- Style 1: four separate strokes -----------------------------------
    h_segs, v_segs = _thin_border_segments(page)
    if len(v_segs) >= 2 and h_segs:

        def _has_connector(x0: float, x1: float, target_y: float) -> bool:
            for hs in h_segs:
                hy = (hs.y0 + hs.y1) / 2.0
                if abs(hy - target_y) > tol:
                    continue
                if hs.x0 <= x0 + tol and hs.x1 >= x1 - tol:
                    return True
            return False

        n = len(v_segs)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = v_segs[i], v_segs[j]
                left, right = (a, b) if a.x0 <= b.x0 else (b, a)
                if abs(left.y0 - right.y0) > tol or abs(left.y1 - right.y1) > tol:
                    continue
                x0 = min(left.x0, left.x1)
                x1 = max(right.x0, right.x1)
                y0 = min(left.y0, right.y0)
                y1 = max(left.y1, right.y1)
                if (x1 - x0) < min_w or (y1 - y0) < min_h:
                    continue
                if not (_has_connector(x0, x1, y0) or _has_connector(x0, x1, y1)):
                    continue
                boxes.append(fitz.Rect(x0, y0, x1, y1))

    # --- Style 2: a single rectangle path ---------------------------------
    boxes.extend(_rect_frame_boxes(page, page_w, page_h, min_w, min_h))

    # De-duplicate near-identical frames (each side pair can match more than
    # once) and drop a frame wholly contained in a larger one.
    unique: list[fitz.Rect] = []
    for box in sorted(boxes, key=lambda r: -(r.width * r.height)):
        if any(
            outer.x0 - tol <= box.x0 and outer.y0 - tol <= box.y0
            and outer.x1 + tol >= box.x1 and outer.y1 + tol >= box.y1
            for outer in unique
        ):
            continue
        unique.append(box)
    return unique


# A thin horizontal stroke that rides on a real content line is a text
# *underline* (or strikethrough), not a decorative divider. Painting it out
# would slice the bottom of the glyphs sitting directly above it (the reported
# "words getting cut" on bold "Statement N is correct:" labels). We recognise it
# by horizontal overlap with a content line plus a vertical position inside the
# line band or just below its baseline.

# Fraction of the stroke's own width that must overlap a content line.
_UNDERLINE_MIN_OVERLAP_FRAC = 0.30

# How far below a content line's baseline (as a fraction of the line height) an
# underline stroke may sit and still count as belonging to that line.
_UNDERLINE_BELOW_LINE_FRAC = 0.45

# How far above a content line's top (as a fraction of the line height) a stroke
# may sit and still count as a decoration of that line.
_UNDERLINE_ABOVE_LINE_FRAC = 0.20


def _is_text_underline(
    r: "fitz.Rect", content_line_rects: "list[fitz.Rect]"
) -> bool:
    """True if a thin horizontal stroke underlines/strikes through body text.

    Such a stroke overlaps a real content line horizontally and sits within (or
    just below) that line's vertical band, so it is part of the rendered text
    and must be KEPT — never painted out as a decorative rule.
    """

    rule_w = float(r.x1 - r.x0)
    if rule_w <= 0:
        return False
    rule_mid = (float(r.y0) + float(r.y1)) / 2.0

    for t in content_line_rects:
        overlap_x = min(float(r.x1), float(t.x1)) - max(float(r.x0), float(t.x0))
        if overlap_x <= 0:
            continue
        if overlap_x < _UNDERLINE_MIN_OVERLAP_FRAC * rule_w:
            continue
        line_h = float(t.y1 - t.y0)
        if line_h <= 0:
            continue
        top_lim = float(t.y0) - _UNDERLINE_ABOVE_LINE_FRAC * line_h
        bottom_lim = float(t.y1) + _UNDERLINE_BELOW_LINE_FRAC * line_h
        if top_lim <= rule_mid <= bottom_lim:
            return True
    return False


def _is_box_border(r: "fitz.Rect", boxes: "list[fitz.Rect]", tol: float) -> bool:
    """True if a thin stroke lies on the perimeter of a detected content box.

    Such a stroke is part of the kept box border and must NOT be painted out.
    """

    for box in boxes:
        # Stroke must sit within the box's footprint (allowing the tolerance).
        if r.x0 < box.x0 - tol or r.x1 > box.x1 + tol:
            continue
        if r.y0 < box.y0 - tol or r.y1 > box.y1 + tol:
            continue
        on_left = abs(r.x0 - box.x0) <= tol
        on_right = abs(r.x1 - box.x1) <= tol
        on_top = abs(r.y0 - box.y0) <= tol
        on_bottom = abs(r.y1 - box.y1) <= tol
        if on_left or on_right or on_top or on_bottom:
            return True
    return False


def _uri_is_branding(uri: str) -> bool:
    u = (uri or "").lower()
    return any(host in u for host in _BRANDING_URI_HOSTS)


def is_branding_text(text: str) -> bool:
    """True if a line is app/website branding (e.g. 'Android App | PW Website').

    Also matches the PW print-preview / penpencil admin URL strip, which is
    plain text (not a hyperlink) and must be kept out of crops.
    """

    t = text or ""
    if _BRANDING_TEXT.search(t):
        return True
    return any(p.search(t) for p in _FURNITURE_TEXT_PATTERNS)


def is_margin_furniture_text(text: str, *, top_pct: float, bottom_pct: float) -> bool:
    """True if a line is header/footer furniture given its vertical position.

    Conservative on purpose: only a page-number strip ("4/5", "Page 4 of 5")
    sitting in the top/bottom margin is treated as furniture here. Running
    titles and bare page numbers are left to the repeat-position detector, which
    has the cross-page evidence to identify them without risking real content
    (a short option line like "(A) option" must never be stripped).
    """

    t = (text or "").strip()
    if not t:
        return False

    in_top = top_pct <= _MARGIN_TOP_PCT
    in_bottom = bottom_pct >= _MARGIN_BOTTOM_PCT
    if not (in_top or in_bottom):
        return False

    return bool(_PAGE_NUMBER_TEXT.match(t))


def branding_link_bands(page: "fitz.Page") -> list[tuple[float, float]]:
    """Vertical (y0, y1) bands of branding hyperlinks on a page, in points."""

    bands: list[tuple[float, float]] = []
    try:
        for lk in page.get_links() or []:
            uri = lk.get("uri") or ""
            r = lk.get("from")
            if r is None or not _uri_is_branding(uri):
                continue
            bands.append((float(r.y0), float(r.y1)))
    except Exception:  # pragma: no cover - defensive
        return bands
    return bands


def collect_page_furniture(page: "fitz.Page") -> list[FurnitureRect]:
    """Return furniture rectangles (points) for a single page.

    Detects, from the PDF structure:
      * hyperlink rectangles pointing at app-store / PW branding URLs;
      * the small logo image that rides next to that branding row;
      * thin long vector rules (decorative side bars / card borders);
      * text lines whose words are branding phrases.
    """

    page_num = page.number + 1
    rect = page.rect
    page_w = float(rect.width)
    page_h = float(rect.height)
    if page_w <= 0 or page_h <= 0:
        return []

    out: list[FurnitureRect] = []

    # 1) Branding hyperlinks. Collect their vertical band so we can also drop the
    #    adjacent logo image / icons that share the row.
    branding_bands: list[tuple[float, float]] = []
    try:
        for lk in page.get_links() or []:
            uri = lk.get("uri") or ""
            r = lk.get("from")
            if r is None or not _uri_is_branding(uri):
                continue
            out.append(FurnitureRect(page_num, float(r.x0), float(r.y0), float(r.x1), float(r.y1)))
            branding_bands.append((float(r.y0), float(r.y1)))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("link_scan_failed page=%s err=%s", page_num, exc)

    # Merge link bands that share a row (small vertical overlap/gap).
    merged_bands: list[list[float]] = []
    for y0, y1 in sorted(branding_bands):
        if merged_bands and y0 <= merged_bands[-1][1] + 0.02 * page_h:
            merged_bands[-1][1] = max(merged_bands[-1][1], y1)
        else:
            merged_bands.append([y0, y1])

    # 2) Furniture text lines: branding ("Android App | PW Website"), the PW
    #    print-preview/penpencil URL strip, page numbers and short running
    #    titles in the margins. Covers footers/headers whether or not their
    #    text is also a tagged hyperlink.
    #
    #    We also record every *real* content line's bbox here so step 4 can tell
    #    a decorative divider from a text underline/strikethrough (a thin stroke
    #    riding on a content line is part of the text and must be kept).
    content_line_rects: list["fitz.Rect"] = []
    try:
        text_dict = page.get_text("dict")
        for block in text_dict.get("blocks", []) or []:
            if not isinstance(block, dict) or block.get("type") not in (None, 0):
                continue
            for line in block.get("lines", []) or []:
                spans = line.get("spans", []) or []
                txt = "".join(str(s.get("text", "")) for s in spans).strip()
                if not txt:
                    continue
                bbox = line.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                top_pct = (float(bbox[1]) / page_h) * 100.0
                bottom_pct = (float(bbox[3]) / page_h) * 100.0
                is_furn = is_branding_text(txt) or is_margin_furniture_text(
                    txt, top_pct=top_pct, bottom_pct=bottom_pct
                )
                if not is_furn:
                    # Real body content — remember it so its underline survives.
                    content_line_rects.append(fitz.Rect(bbox))
                    continue
                out.append(FurnitureRect(page_num, float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])))
                if is_branding_text(txt):
                    merged_bands.append([float(bbox[1]), float(bbox[3])])
                    merged_bands.sort()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("text_scan_failed page=%s err=%s", page_num, exc)

    # 3) Logo / icon images that sit inside a branding row.
    try:
        for im in page.get_image_info() or []:
            bb = im.get("bbox")
            if not bb or len(bb) != 4:
                continue
            iy0, iy1 = float(bb[1]), float(bb[3])
            ih = iy1 - iy0
            # A full-page background image is the scan itself, never furniture.
            if ih >= 0.6 * page_h:
                continue
            for by0, by1 in merged_bands:
                if iy0 < by1 + 0.02 * page_h and iy1 > by0 - 0.02 * page_h:
                    out.append(FurnitureRect(page_num, float(bb[0]), iy0, float(bb[2]), iy1))
                    break
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("image_scan_failed page=%s err=%s", page_num, exc)

    # 4) Thin long vector rules (side accent bars, card borders). A stroke that
    #    forms the perimeter of a content note-box ("Additional Information",
    #    "PW SUPER HINT") is part of the kept content, so it is excluded here.
    content_boxes = detect_content_boxes(page)
    try:
        for d in page.get_drawings() or []:
            r = d.get("rect")
            if r is None:
                continue
            r = fitz.Rect(r)
            w = float(r.width)
            h = float(r.height)
            # A mathematically horizontal/vertical rule has zero thickness in one
            # axis. Skip only when BOTH axes are empty (a degenerate point); a
            # zero-height full-width line is exactly the separator/table rule we
            # most want to catch, so it must not be discarded here.
            if w <= 0 and h <= 0:
                continue
            if _is_box_border(r, content_boxes, _BOX_EDGE_ALIGN_TOL_PTS):
                continue
            is_v_rule = w <= _RULE_MAX_THICK_FRAC * page_w and h >= _RULE_MIN_LONG_FRAC * page_h
            is_h_rule = h <= _RULE_MAX_THICK_FRAC * page_h and w >= _RULE_MIN_LONG_FRAC * page_w
            # A thin horizontal stroke riding on a content line is a text
            # underline/strikethrough, not a divider — keep it so the glyphs
            # above it aren't sliced when furniture is painted out.
            if is_h_rule and _is_text_underline(r, content_line_rects):
                continue
            if is_v_rule or is_h_rule:
                out.append(FurnitureRect(page_num, float(r.x0), float(r.y0), float(r.x1), float(r.y1)))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("drawing_scan_failed page=%s err=%s", page_num, exc)

    return out


def collect_document_furniture(doc: "fitz.Document") -> dict[int, list[FurnitureRect]]:
    """Return ``{page_num: [FurnitureRect, ...]}`` for every page in the doc."""

    result: dict[int, list[FurnitureRect]] = {}
    for pno in range(doc.page_count):
        try:
            page = doc.load_page(pno)
        except Exception:  # pragma: no cover - defensive
            continue
        rects = collect_page_furniture(page)
        if rects:
            result[pno + 1] = rects
    return result


def paint_pad_pts() -> float:
    """Padding (points) to grow each furniture rect by when painting it out."""

    return _PAINT_PAD_PTS
