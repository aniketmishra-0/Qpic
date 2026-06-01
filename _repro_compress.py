"""Reproduce the dark-background regression from compress_pdf.

Builds a one-page PDF that embeds an image WITH a soft mask (SMask) kept in the
PDF, runs the current compressor, then renders before/after to see whether the
previously-transparent region turns black.
"""

import io

import fitz
from PIL import Image

from app.services.pdf_tools.compress_service import compress_pdf, _apply_preset, LEVELS


def build_pdf_with_smask() -> bytes:
    w = h = 600
    # Foreground navy image with a transparent square hole.
    base = Image.new("RGB", (w, h), (16, 24, 64))
    alpha = Image.new("L", (w, h), 255)
    for y in range(200, 400):
        for x in range(200, 400):
            alpha.putpixel((x, y), 0)
    base.putalpha(alpha)
    buf = io.BytesIO()
    base.save(buf, format="PNG")
    fg_png = buf.getvalue()

    doc = fitz.open()
    page = doc.new_page(width=w, height=h)
    # white page; draw a red rectangle UNDER the image so transparent hole shows red
    page.draw_rect(page.rect, color=None, fill=(1, 0, 0))
    page.insert_image(page.rect, stream=fg_png, keep_proportion=False, overlay=True)
    out = doc.tobytes()
    doc.close()
    return out


def count_smasks(pdf_bytes: bytes) -> int:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n = 0
    for xref in range(1, doc.xref_length()):
        try:
            if doc.xref_get_key(xref, "SMask")[0] != "null":
                n += 1
        except Exception:
            pass
    doc.close()
    return n


def sample_render(pdf_bytes: bytes, label: str):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    pix = page.get_pixmap(dpi=72)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    cx, cy = pix.width // 2, pix.height // 2
    print(f"{label}: hole={img.getpixel((cx, cy))} navy={img.getpixel((5, 5))} "
          f"size={len(pdf_bytes)} smasks={count_smasks(pdf_bytes)}")


if __name__ == "__main__":
    src = build_pdf_with_smask()
    sample_render(src, "ORIGINAL  ")
    res = compress_pdf(src, level="balanced")
    sample_render(res.data, "COMPRESSED")
