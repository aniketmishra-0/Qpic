"""Reproduce grayscale darkening via extreme level and target ladder."""

import fitz
from PIL import Image

from app.services.pdf_tools.compress_service import _apply_preset, _save_optimized, LEVELS, _Preset

SRC = "/Users/Chrome/Current Affairs (Week 139th) Class Notes.pdf"


def report(data, label):
    doc = fitz.open(stream=data, filetype="pdf")
    cs = None
    for img in doc[0].get_images(full=True):
        cs = doc.extract_image(img[0]).get("colorspace")
        break
    pix = doc[0].get_pixmap(dpi=50)
    im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    px = im.getpixel((pix.width // 2, int(pix.height * 0.15)))
    doc.close()
    print(f"{label:40s} channels={cs} centerpx={px} size={len(data)}")


def run(preset, label):
    doc = fitz.open(SRC)
    _apply_preset(doc, preset)
    data = _save_optimized(doc)
    doc.close()
    report(data, label)


run(LEVELS["extreme"], "extreme (to_gray=True)")
run(_Preset(dpi_target=80, quality=38, to_gray=True), "ladder rung4 to_gray")
