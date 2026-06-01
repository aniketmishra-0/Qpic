"""Verify color is preserved across all levels + target on the real file."""

import fitz
from PIL import Image

from app.services.pdf_tools.compress_service import compress_pdf

SRC = "/Users/Chrome/Current Affairs (Week 139th) Class Notes.pdf"

with open(SRC, "rb") as f:
    data = f.read()


def report(out, label):
    doc = fitz.open(stream=out, filetype="pdf")
    cs = None
    for img in doc[0].get_images(full=True):
        cs = doc.extract_image(img[0]).get("colorspace")
        break
    pix = doc[0].get_pixmap(dpi=50)
    im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    px = im.getpixel((pix.width // 2, int(pix.height * 0.15)))
    doc.close()
    mb = len(out) / (1024 * 1024)
    print(f"{label:24s} channels={cs} centerpx={px} size={mb:.2f}MB")


print(f"ORIGINAL size={len(data)/(1024*1024):.2f}MB  (navy slide centerpx ~ (10,11,28))\n")
for lvl in ("light", "balanced", "strong", "extreme"):
    report(compress_pdf(data, level=lvl).data, f"level={lvl}")

r = compress_pdf(data, target_mb=2.0)
report(r.data, "target_mb=2.0")
print("target_met:", r.target_met, "note:", r.note)
