"""Run each stage of compress on the real file to find where color is lost."""

import fitz
from PIL import Image

from app.services.pdf_tools.compress_service import _apply_preset, _save_optimized, LEVELS

SRC = "/Users/Chrome/Current Affairs (Week 139th) Class Notes.pdf"


def first_cs_and_px(doc, label):
    page = doc[0]
    cs = None
    for img in page.get_images(full=True):
        d = doc.extract_image(img[0])
        cs = d.get("colorspace")
        break
    pix = page.get_pixmap(dpi=50)
    im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    px = im.getpixel((pix.width // 2, int(pix.height * 0.15)))
    print(f"{label:30s} channels={cs} centerpx={px}")


preset = LEVELS["balanced"]
print("preset:", preset)

doc = fitz.open(SRC)
first_cs_and_px(doc, "0. opened")

_apply_preset(doc, preset)
first_cs_and_px(doc, "1. after _apply_preset")

data = _save_optimized(doc)
doc.close()

doc2 = fitz.open(stream=data, filetype="pdf")
first_cs_and_px(doc2, "2. after _save_optimized")
doc2.close()
print("final size:", len(data))
