"""Test whether rewrite_images darkens CMYK / Adobe-transform JPEGs and
whether re-encoding a lossy SMask shifts opacity (darkening)."""

import io

import fitz
from PIL import Image

from app.services.pdf_tools.compress_service import compress_pdf


def build_cmyk_pdf() -> bytes:
    w = h = 400
    rgb = Image.new("RGB", (w, h), (16, 24, 90))  # navy
    cmyk = rgb.convert("CMYK")
    buf = io.BytesIO()
    cmyk.save(buf, format="JPEG", quality=95)
    jpg = buf.getvalue()

    doc = fitz.open()
    page = doc.new_page(width=w, height=h)
    page.insert_image(page.rect, stream=jpg)
    out = doc.tobytes()
    doc.close()
    return out


def render_center(pdf_bytes, label):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pix = doc[0].get_pixmap(dpi=72)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    print(f"{label}: center={img.getpixel((pix.width//2, pix.height//2))} size={len(pdf_bytes)}")


if __name__ == "__main__":
    src = build_cmyk_pdf()
    render_center(src, "CMYK ORIGINAL  ")
    res = compress_pdf(src, level="balanced")
    render_center(res.data, "CMYK COMPRESSED")
