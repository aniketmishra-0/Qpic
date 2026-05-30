"""Regression tests for structural page-furniture removal (PW solution cards).

These cover the case the text-repetition heuristic missed: a branding footer
("Android App | iOS App | PW Website") that sits *mid-page* on a short solution
card — with the branding as hyperlinks + a logo image — plus a decorative
vertical accent rule drawn as a vector. Both must be kept out of the crop:

  * the footer must not *size* the crop (detection drops it), and
  * any furniture that still falls inside a rendered region (e.g. the middle of
    a cross-page stitch) must be painted out of the pixels.
"""

from __future__ import annotations

import io

import fitz
import numpy as np
from PIL import Image

from app.services.crop_service import crop_and_stitch_hires
from app.services.detector.furniture import (
    collect_document_furniture,
    collect_page_furniture,
)
from app.services.detector.text_detector import TextDetector

W, H = 595, 842
PLAY = "https://play.google.com/store/apps/details?id=xyz.penpencil"
APPLE = "https://apps.apple.com/in/app/physicswallah/id123"
PW = "https://www.pw.live"


def _logo_png() -> bytes:
    b = io.BytesIO()
    Image.new("RGB", (24, 24), (250, 180, 40)).save(b, "PNG")
    return b.getvalue()


def _build_pw_pdf() -> bytes:
    """Two-page solution: S5 content on page 1 (short card) continues to page 2.
    Each page carries a vertical accent bar + branding footer at ~y=75%.
    """

    doc = fitz.open()
    for pno in range(2):
        p = doc.new_page(width=W, height=H)
        # vertical accent bar (vector) right of the text column
        p.draw_rect(fitz.Rect(540, 70, 548, 620), color=None, fill=(0.7, 0.7, 0.72))
        if pno == 0:
            p.insert_text((60, 90), "Q5 Text Solution:", fontsize=12)
            p.insert_text((60, 115), "Ans: (b)", fontsize=11)
            p.insert_text((60, 150), "The social cost of carbon refers to monetary", fontsize=11)
            p.insert_text((60, 175), "value of damage from each ton of CO2 emitted.", fontsize=11)
        else:
            p.insert_text((60, 90), "weather events. Policymakers use SCC widely.", fontsize=11)
            p.insert_text((60, 115), "Thus, Option B is the correct answer.", fontsize=11)
        # branding footer at ~ y=622-646 (≈74-77% of the page)
        p.insert_image(fitz.Rect(150, 622, 174, 646), stream=_logo_png())
        p.insert_text((185, 640), "Android App  |  iOS App  |  PW Website", fontsize=12)
        p.insert_link({"kind": fitz.LINK_URI, "from": fitz.Rect(185, 628, 300, 646), "uri": PLAY})
        p.insert_link({"kind": fitz.LINK_URI, "from": fitz.Rect(310, 628, 380, 646), "uri": APPLE})
        p.insert_link({"kind": fitz.LINK_URI, "from": fitz.Rect(390, 628, 500, 646), "uri": PW})
    data = doc.tobytes()
    doc.close()
    return data


def test_collect_page_furniture_finds_links_logo_and_bar() -> None:
    pdf = _build_pw_pdf()
    with fitz.open(stream=pdf, filetype="pdf") as doc:
        rects = collect_page_furniture(doc.load_page(0))
    # 3 branding links + 1 branding text line + 1 logo image + 1 vector bar.
    assert len(rects) >= 5
    # A tall thin vector bar must be among them.
    assert any((r.y1 - r.y0) > 0.4 * H and (r.x1 - r.x0) < 0.05 * W for r in rects)


def test_footer_does_not_size_the_crop() -> None:
    """The S5 content ends well above the footer band (~74%), proving the footer
    no longer extends the question's content region."""

    pdf = _build_pw_pdf()
    questions = TextDetector().detect(pdf, padding_px=10)
    s5 = next((q for q in questions if q.q_num == "5"), None)
    assert s5 is not None
    page1_seg = next((s for s in s5.segments if s.page == 1), None)
    assert page1_seg is not None
    assert page1_seg.y_end_pct < 70.0, page1_seg.y_end_pct


def _tall_bar_columns(img: Image.Image) -> list[int]:
    arr = np.asarray(img.convert("RGB")).astype(int)
    col_cov = (arr.min(axis=2) < 235).mean(axis=0)
    return [x for x in range(img.width) if col_cov[x] > 0.5]


def test_rendered_crop_has_no_accent_bar_or_footer() -> None:
    pdf = _build_pw_pdf()
    questions = TextDetector().detect(pdf, padding_px=10)
    s5 = next((q for q in questions if q.q_num == "5"), None)
    assert s5 is not None

    with fitz.open(stream=pdf, filetype="pdf") as doc:
        furniture = collect_document_furniture(doc)
        img = crop_and_stitch_hires(
            doc, s5, padding_px=10, detection_dpi=200, crop_dpi=200,
            furniture_by_page=furniture,
        )

    # No tall vertical rule survives anywhere in the crop.
    assert _tall_bar_columns(img) == []
    # The crop still contains real content (not blanked out).
    arr = np.asarray(img.convert("RGB")).astype(int)
    assert (arr.min(axis=2) < 128).mean() > 0.005


# --- Multi-column, multi-page question (the hardest case) -------------------

_DIV = W / 2


def _hdr_ftr(p, pageno: int) -> None:
    """PW running header + branding footer + url/page-number strip."""

    p.insert_text((40, 25), "2/5/26, 9:40 PM", fontsize=8)
    p.insert_text((360, 25), "UPSC_DPP 4", fontsize=8)
    p.insert_text((520, 45), "UPSC", fontsize=9)
    p.draw_line((_DIV, 60), (_DIV, 700), width=0.6)
    p.insert_text((120, 612), "Android App  |  iOS App  |  PW Website", fontsize=10)
    p.insert_link({"kind": fitz.LINK_URI, "from": fitz.Rect(120, 600, 190, 616), "uri": PLAY})
    p.insert_link({"kind": fitz.LINK_URI, "from": fitz.Rect(400, 600, 470, 616), "uri": PW})
    p.insert_text((40, 700), "https://qbg-admin.penpencil.co/finalize-question-paper/print-preview", fontsize=7)
    p.insert_text((540, 700), f"{pageno}/5", fontsize=8)


def _build_multicol_pdf() -> bytes:
    """Q5 starts at the bottom of page-1 right column and continues into the
    left then right column of page 2 — three segments to stitch."""

    doc = fitz.open()
    rx = _DIV + 20

    p1 = doc.new_page(width=W, height=H)
    _hdr_ftr(p1, 4)
    p1.insert_text((rx, 450), "Q5 Text Solution:", fontsize=10)
    p1.insert_text((rx, 470), "Ans: (b)", fontsize=9)
    p1.insert_text((rx, 488), "Exp:", fontsize=9)

    p2 = doc.new_page(width=W, height=H)
    _hdr_ftr(p2, 5)
    for i, t in enumerate([
        "The social cost of carbon (SCC) refers to the",
        "monetary value associated with the damage",
        "caused by each additional ton of carbon",
        "dioxide emitted into the atmosphere. It",
    ]):
        p2.insert_text((40, 80 + i * 18), t, fontsize=9)
    for i, t in enumerate([
        "health costs, rising sea levels, and extreme",
        "weather events. Policymakers use SCC widely.",
        "Thus, Option B is the correct answer.",
    ]):
        p2.insert_text((rx, 80 + i * 18), t, fontsize=9)

    data = doc.tobytes()
    doc.close()
    return data


def test_multicol_question_has_three_tight_segments() -> None:
    pdf = _build_multicol_pdf()
    questions = TextDetector().detect(pdf, padding_px=15)
    q5 = next((q for q in questions if q.q_num == "5"), None)
    assert q5 is not None

    # Exactly three segments: p1 right, p2 left, p2 right (no spurious 4th from
    # the "UPSC" running title or the page-number strip).
    assert len(q5.segments) == 3
    pages = [s.page for s in q5.segments]
    assert pages == [1, 2, 2]

    # Each segment ends at real content, not dragged down to the footer/url band
    # (~73% branding, ~83% url). All must finish well above that.
    for s in q5.segments:
        assert s.y_end_pct < 70.0, s


def test_multicol_rendered_crop_is_clean() -> None:
    pdf = _build_multicol_pdf()
    questions = TextDetector().detect(pdf, padding_px=15)
    q5 = next((q for q in questions if q.q_num == "5"), None)
    assert q5 is not None

    with fitz.open(stream=pdf, filetype="pdf") as doc:
        furniture = collect_document_furniture(doc)
        img = crop_and_stitch_hires(
            doc, q5, padding_px=15, detection_dpi=200, crop_dpi=200,
            furniture_by_page=furniture,
        )

    assert _tall_bar_columns(img) == []
    arr = np.asarray(img.convert("RGB")).astype(int)
    assert (arr.min(axis=2) < 128).mean() > 0.005  # still has real content
