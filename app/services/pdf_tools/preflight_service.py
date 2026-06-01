"""Preflight a PDF — inspect it for print/production readiness.

Preflight is the prepress check publishers run before a file goes to print: it
reports structural facts (page count, sizes), embedded vs. non-embedded fonts,
image resolutions (low-DPI images print blurry), colour spaces (RGB vs CMYK),
encryption/permissions, and other risk flags, then rolls them up into a simple
PASS / WARN / FAIL verdict with actionable messages.

It is read-only: nothing about the PDF is modified. Runs CPU-bound, so call it
inside ``asyncio.to_thread``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import fitz

logger = logging.getLogger(__name__)

# Images below this effective resolution will look soft in print.
_LOW_DPI = 150.0
# ... and below this they're genuinely unacceptable for print.
_CRITICAL_DPI = 72.0

# Severity ranking so we can roll many checks up into one verdict.
_SEVERITY_ORDER = {"info": 0, "ok": 0, "warn": 1, "fail": 2}


@dataclass
class PreflightCheck:
    """A single preflight finding."""

    id: str
    title: str
    status: str  # "ok" | "warn" | "fail" | "info"
    detail: str


@dataclass
class FontInfo:
    name: str
    type: str
    embedded: bool
    subset: bool


@dataclass
class ImageInfo:
    page: int
    width: int
    height: int
    dpi: float
    colorspace: str
    bpc: int


@dataclass
class PreflightReport:
    """The full preflight result."""

    verdict: str  # "pass" | "warn" | "fail"
    page_count: int
    page_sizes: list[str]
    file_size: int
    is_encrypted: bool
    has_text_layer: bool
    checks: list[PreflightCheck] = field(default_factory=list)
    fonts: list[FontInfo] = field(default_factory=list)
    images: list[ImageInfo] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    # Distinct page-size labels across the whole document (not truncated like
    # ``page_sizes``). Lets the UI offer a one-click "fix mixed sizes" action.
    distinct_page_sizes: list[str] = field(default_factory=list)
    mixed_page_sizes: bool = False
    # Per-page detailed geometry for the Preflight Check modal table.
    page_details: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class NormalizeResult:
    """Result of normalizing every page to one uniform size."""

    data: bytes
    target_width: float   # PDF points
    target_height: float  # PDF points
    target_label: str     # human-readable, e.g. "210×297 mm (A4)"
    pages_total: int
    pages_changed: int
    note: str


def _pt_to_mm(pt: float) -> float:
    return pt * 25.4 / 72.0


def _page_size_label(rect: "fitz.Rect") -> str:
    """Human-readable page size, e.g. ``'210×297 mm (A4)'``."""

    w_mm = round(_pt_to_mm(rect.width))
    h_mm = round(_pt_to_mm(rect.height))

    # Recognise a few common sizes (within 2 mm tolerance, either orientation).
    known = {
        (210, 297): "A4",
        (297, 420): "A3",
        (216, 279): "Letter",
        (216, 356): "Legal",
        (148, 210): "A5",
    }
    label = ""
    for (kw, kh), name in known.items():
        if (abs(w_mm - kw) <= 2 and abs(h_mm - kh) <= 2) or (
            abs(w_mm - kh) <= 2 and abs(h_mm - kw) <= 2
        ):
            label = f" ({name})"
            break
    return f"{w_mm}×{h_mm} mm{label}"


def _collect_fonts(doc: "fitz.Document") -> list[FontInfo]:
    """Gather the unique fonts used across the document."""

    seen: dict[str, FontInfo] = {}
    for pno in range(doc.page_count):
        for f in doc.get_page_fonts(pno, full=False):
            # f = (xref, ext, type, basefont, name, encoding)
            xref, ext, ftype, basefont = f[0], f[1], f[2], f[3]
            key = f"{xref}:{basefont}"
            if key in seen:
                continue
            # ``ext != 'n/a'`` means glyph program bytes are embedded.
            embedded = bool(ext) and ext != "n/a"
            # A subset font's basefont name is prefixed like ``ABCDEF+Arial``.
            subset = "+" in (basefont or "")
            seen[key] = FontInfo(
                name=basefont or "(unnamed)",
                type=ftype or "?",
                embedded=embedded,
                subset=subset,
            )
    return list(seen.values())


def _image_dpi(width: int, height: int, bbox: "fitz.Rect") -> float:
    """Effective DPI = pixel size / displayed size on the page."""

    w_in = bbox.width / 72.0
    h_in = bbox.height / 72.0
    if w_in <= 0 or h_in <= 0:
        return 0.0
    dpi_x = width / w_in
    dpi_y = height / h_in
    # The lower of the two axes is what limits perceived sharpness.
    return round(min(dpi_x, dpi_y), 1)


def _collect_images(doc: "fitz.Document", limit: int = 500) -> list[ImageInfo]:
    """Gather embedded raster images and their effective on-page resolution."""

    out: list[ImageInfo] = []
    for pno in range(doc.page_count):
        page = doc.load_page(pno)
        try:
            infos = page.get_image_info(xrefs=True)
        except Exception:
            infos = []
        for info in infos:
            bbox = fitz.Rect(info.get("bbox", (0, 0, 0, 0)))
            width = int(info.get("width", 0))
            height = int(info.get("height", 0))
            if width <= 0 or height <= 0:
                continue
            out.append(
                ImageInfo(
                    page=pno + 1,
                    width=width,
                    height=height,
                    dpi=_image_dpi(width, height, bbox),
                    colorspace=str(info.get("cs-name", "?")),
                    bpc=int(info.get("bpc", 0) or 0),
                )
            )
            if len(out) >= limit:
                return out
    return out


def _has_text_layer(doc: "fitz.Document") -> bool:
    total = 0
    for pno in range(min(5, doc.page_count)):
        total += len((doc.load_page(pno).get_text("text") or "").strip())
        if total > 50:
            return True
    return total > 50


def preflight_pdf(file_bytes: bytes) -> PreflightReport:
    """Run a read-only production-readiness inspection of a PDF."""

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        checks: list[PreflightCheck] = []
        page_count = doc.page_count

        page_sizes: list[str] = []
        size_set: set[str] = set()
        page_details: list[dict[str, Any]] = []
        for pno in range(page_count):
            rect = doc.load_page(pno).rect
            label = _page_size_label(rect)
            if pno < 20:
                page_sizes.append(label)
            size_set.add(label)
            # Per-page detail for the Preflight Check table
            w_mm = round(_pt_to_mm(rect.width), 1)
            h_mm = round(_pt_to_mm(rect.height), 1)
            w_pt = round(rect.width, 2)
            h_pt = round(rect.height, 2)
            # Pixels at 72 DPI (1:1 with points for PDF)
            w_px = round(rect.width * 200 / 72)
            h_px = round(rect.height * 200 / 72)
            orientation = "Landscape" if rect.width > rect.height else "Portrait"
            # Detect known format
            fmt = "Custom"
            known = {
                (210, 297): "A4", (297, 420): "A3", (216, 279): "Letter",
                (216, 356): "Legal", (148, 210): "A5",
            }
            for (kw, kh), name in known.items():
                if (abs(round(w_mm) - kw) <= 2 and abs(round(h_mm) - kh) <= 2) or (
                    abs(round(w_mm) - kh) <= 2 and abs(round(h_mm) - kw) <= 2
                ):
                    fmt = name
                    break
            page_details.append({
                "page": pno + 1,
                "w_mm": w_mm,
                "h_mm": h_mm,
                "w_pt": w_pt,
                "h_pt": h_pt,
                "w_px": w_px,
                "h_px": h_px,
                "format": fmt,
                "orientation": orientation,
            })

        is_encrypted = bool(doc.is_encrypted) or doc.needs_pass
        fonts = _collect_fonts(doc)
        images = _collect_images(doc)
        has_text = _has_text_layer(doc)
        meta = {k: str(v) for k, v in (doc.metadata or {}).items() if v}

        # --- Check: encryption -------------------------------------------
        if is_encrypted:
            checks.append(PreflightCheck(
                id="encryption",
                title="Encryption / password",
                status="fail",
                detail="The PDF is encrypted. Print/RIP workflows usually reject "
                       "encrypted files — remove the password before sending.",
            ))
        else:
            checks.append(PreflightCheck(
                id="encryption", title="Encryption / password", status="ok",
                detail="Not encrypted.",
            ))

        # --- Check: non-embedded fonts -----------------------------------
        non_embedded = [f for f in fonts if not f.embedded]
        if non_embedded:
            names = ", ".join(sorted({f.name for f in non_embedded})[:6])
            checks.append(PreflightCheck(
                id="fonts_embedded",
                title="Font embedding",
                status="fail",
                detail=f"{len(non_embedded)} font(s) are NOT embedded ({names}). "
                       "Text may reflow or substitute on another machine/RIP. "
                       "Embed all fonts before printing.",
            ))
        elif fonts:
            checks.append(PreflightCheck(
                id="fonts_embedded", title="Font embedding", status="ok",
                detail=f"All {len(fonts)} font(s) are embedded.",
            ))
        else:
            checks.append(PreflightCheck(
                id="fonts_embedded", title="Font embedding", status="info",
                detail="No fonts found (image-only PDF).",
            ))

        # --- Check: image resolution -------------------------------------
        critical = [i for i in images if 0 < i.dpi < _CRITICAL_DPI]
        low = [i for i in images if _CRITICAL_DPI <= i.dpi < _LOW_DPI]
        if critical:
            worst = min(i.dpi for i in critical)
            checks.append(PreflightCheck(
                id="image_dpi",
                title="Image resolution",
                status="fail",
                detail=f"{len(critical)} image(s) below {int(_CRITICAL_DPI)} DPI "
                       f"(worst {worst} DPI). These will print clearly blurry.",
            ))
        elif low:
            worst = min(i.dpi for i in low)
            checks.append(PreflightCheck(
                id="image_dpi",
                title="Image resolution",
                status="warn",
                detail=f"{len(low)} image(s) between {int(_CRITICAL_DPI)}–{int(_LOW_DPI)} DPI "
                       f"(lowest {worst}). Acceptable on screen, soft in print.",
            ))
        elif images:
            checks.append(PreflightCheck(
                id="image_dpi", title="Image resolution", status="ok",
                detail=f"All {len(images)} image(s) at {int(_LOW_DPI)} DPI or better.",
            ))
        else:
            checks.append(PreflightCheck(
                id="image_dpi", title="Image resolution", status="info",
                detail="No raster images found (vector/text PDF).",
            ))

        # --- Check: colour space (RGB images for a print target) ---------
        rgb_images = [i for i in images if "rgb" in i.colorspace.lower()]
        if rgb_images:
            checks.append(PreflightCheck(
                id="colorspace",
                title="Colour space",
                status="warn",
                detail=f"{len(rgb_images)} image(s) use RGB. Offset printing "
                       "expects CMYK — colours may shift on conversion.",
            ))
        else:
            checks.append(PreflightCheck(
                id="colorspace", title="Colour space", status="ok",
                detail="No RGB images flagged.",
            ))

        # --- Check: mixed page sizes -------------------------------------
        distinct_sizes = sorted(size_set)
        mixed = len(size_set) > 1
        if mixed:
            checks.append(PreflightCheck(
                id="page_sizes",
                title="Page geometry",
                status="warn",
                detail=f"{len(size_set)} different page sizes in one document "
                       f"({', '.join(distinct_sizes[:4])}"
                       f"{'…' if len(distinct_sizes) > 4 else ''}). "
                       "Mixed sizes break imposition and print trays — use "
                       "“Fix page sizes” to normalize every page to one size.",
            ))
        else:
            checks.append(PreflightCheck(
                id="page_sizes", title="Page geometry", status="ok",
                detail=f"Consistent page size: {next(iter(size_set), 'unknown')}.",
            ))

        # --- Check: searchable text --------------------------------------
        checks.append(PreflightCheck(
            id="text_layer",
            title="Text layer",
            status="ok" if has_text else "info",
            detail="Selectable text layer present." if has_text
                   else "No selectable text (scanned/image PDF) — run OCR in Edit "
                        "if you need searchable text.",
        ))

        # --- Roll up to a single verdict ---------------------------------
        worst_sev = max((_SEVERITY_ORDER.get(c.status, 0) for c in checks), default=0)
        verdict = {0: "pass", 1: "warn", 2: "fail"}[worst_sev]

        return PreflightReport(
            verdict=verdict,
            page_count=page_count,
            page_sizes=page_sizes,
            file_size=len(file_bytes),
            is_encrypted=is_encrypted,
            has_text_layer=has_text,
            checks=checks,
            fonts=fonts,
            images=images[:200],
            metadata=meta,
            distinct_page_sizes=distinct_sizes,
            mixed_page_sizes=mixed,
            page_details=page_details,
        )
    finally:
        doc.close()


# ---------------------------------------------------------------------------
#  Fix: normalize mixed page sizes to one uniform size (Acrobat-style)
# ---------------------------------------------------------------------------

# Named print sizes in PDF points (1 pt = 1/72"). Values are portrait W×H.
_NAMED_SIZES_PT: dict[str, tuple[float, float]] = {
    "a3": (841.89, 1190.55),
    "a4": (595.28, 841.89),
    "a5": (419.53, 595.28),
    "letter": (612.0, 792.0),
    "legal": (612.0, 1008.0),
    "square": (612.0, 612.0),  # 216×216 mm
}


def _mode_page_size(doc: "fitz.Document") -> tuple[float, float]:
    """Return the most common (width, height) across pages, in points.

    Sizes are bucketed to the nearest point so tiny floating-point differences
    between otherwise-identical pages collapse together. Ties break toward the
    larger area so content is never forced to shrink onto a smaller majority.
    """

    counts: dict[tuple[int, int], int] = {}
    exact: dict[tuple[int, int], tuple[float, float]] = {}
    for pno in range(doc.page_count):
        r = doc.load_page(pno).rect
        key = (round(r.width), round(r.height))
        counts[key] = counts.get(key, 0) + 1
        exact.setdefault(key, (r.width, r.height))
    # Most frequent, then largest area.
    best = max(counts, key=lambda k: (counts[k], k[0] * k[1]))
    return exact[best]


def resolve_target_size(doc: "fitz.Document", target: str) -> tuple[float, float]:
    """Resolve a target-size spec to (width_pt, height_pt).

    ``target`` may be:
      * ``"auto"`` / ``""`` — the most common existing page size (the majority).
      * ``"max"`` — the largest page (by area), so nothing is downscaled.
      * a named size: ``a3``/``a4``/``a5``/``letter``/``legal``/``square``. The
        orientation is chosen to match the document's dominant orientation.
      * ``"custom:<W_mm>x<H_mm>"`` — an explicit size in millimetres, e.g.
        ``"custom:210x297"`` for A4.
    """

    spec = (target or "auto").strip().lower()
    if spec in ("auto", "", "mode", "common"):
        return _mode_page_size(doc)

    if spec == "max":
        best = None
        best_area = -1.0
        for pno in range(doc.page_count):
            r = doc.load_page(pno).rect
            area = r.width * r.height
            if area > best_area:
                best_area = area
                best = (r.width, r.height)
        return best or _mode_page_size(doc)

    # Custom size: "custom:<W_mm>x<H_mm>"
    if spec.startswith("custom:"):
        try:
            dims = spec[7:]  # strip "custom:"
            parts = dims.split("x")
            w_mm = float(parts[0])
            h_mm = float(parts[1])
            w_pt = w_mm * 72.0 / 25.4
            h_pt = h_mm * 72.0 / 25.4
            if w_pt > 0 and h_pt > 0:
                return (w_pt, h_pt)
        except (ValueError, IndexError):
            pass
        return _mode_page_size(doc)

    if spec in _NAMED_SIZES_PT:
        w, h = _NAMED_SIZES_PT[spec]
        # Square doesn't need orientation matching.
        if spec == "square":
            return (w, h)
        # Match the document's dominant orientation (landscape vs portrait).
        landscape_votes = sum(
            1 for pno in range(doc.page_count)
            if doc.load_page(pno).rect.width > doc.load_page(pno).rect.height
        )
        if landscape_votes > doc.page_count / 2:
            w, h = h, w
        return (w, h)

    # Unknown spec → fall back to the majority size.
    return _mode_page_size(doc)


def normalize_page_sizes(
    file_bytes: bytes,
    *,
    target: str = "auto",
    tolerance_pt: float = 1.0,
    fill_mode: str = "fit",
    skip_pages: Optional[list[int]] = None,
) -> NormalizeResult:
    """Resize every page to one uniform size, scaling content to fit.

    This mirrors Acrobat's preflight "fix" for mixed page geometry: each page
    that differs from the chosen target size is rebuilt at the target size with
    its original content scaled proportionally and centred (no distortion, no
    cropping). Pages that already match the target (within ``tolerance_pt``) are
    copied through untouched, so vector/text quality is preserved everywhere
    possible.

    ``fill_mode`` controls how content is placed on the target page:
      * ``"fit"`` (default) — scale proportionally to fit inside the target,
        preserving aspect ratio. Near-match pages have extra white bands auto-
        reduced.
      * ``"stretch"`` — stretch/squash content to fill the entire target rect
        (distorts aspect ratio if source and target differ).

    ``skip_pages`` is an optional list of 1-indexed page numbers that should be
    copied through untouched regardless of their size.

    Content is placed with :meth:`Page.show_pdf_page`, which renders the source
    page as a vector form XObject — text stays selectable and lines stay crisp.
    """

    skip_set: set[int] = set(skip_pages) if skip_pages else set()

    src = fitz.open(stream=file_bytes, filetype="pdf")
    out = fitz.open()
    try:
        tw, th = resolve_target_size(src, target)
        target_rect = fitz.Rect(0, 0, tw, th)
        changed = 0

        for pno in range(src.page_count):
            page = src.load_page(pno)
            r = page.rect

            # Skip pages the user explicitly excluded.
            if (pno + 1) in skip_set:
                out.insert_pdf(src, from_page=pno, to_page=pno)
                continue

            if abs(r.width - tw) <= tolerance_pt and abs(r.height - th) <= tolerance_pt:
                # Already the right size — copy the page verbatim.
                out.insert_pdf(src, from_page=pno, to_page=pno)
                continue

            if fill_mode == "stretch":
                # Stretch: fill the entire target rect (distorts if aspect differs).
                dest = target_rect
            else:
                # Fit: scale proportionally, centre on the target page.
                scale = min(tw / r.width, th / r.height) if r.width and r.height else 1.0
                draw_w = r.width * scale
                draw_h = r.height * scale
                x0 = (tw - draw_w) / 2.0
                y0 = (th - draw_h) / 2.0
                dest = fitz.Rect(x0, y0, x0 + draw_w, y0 + draw_h)

            new_page = out.new_page(width=tw, height=th)
            new_page.show_pdf_page(dest, src, pno)
            changed += 1

        data = out.tobytes(garbage=4, deflate=True, clean=True)
        label = _page_size_label(target_rect)
        if changed:
            note = (
                f"Normalized {changed} of {src.page_count} page(s) to {label}. "
                "Resized pages had their content scaled to fit and centred."
            )
        else:
            note = f"Every page was already {label} — nothing to change."
        return NormalizeResult(
            data=data,
            target_width=tw,
            target_height=th,
            target_label=label,
            pages_total=src.page_count,
            pages_changed=changed,
            note=note,
        )
    finally:
        src.close()
        out.close()
