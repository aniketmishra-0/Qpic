"""Find which rewrite_images param combo flips color images to grayscale."""

import fitz
from PIL import Image

SRC = "/Users/Chrome/Current Affairs (Week 139th) Class Notes.pdf"


def first_image_cs(doc):
    page = doc[0]
    for img in page.get_images(full=True):
        d = doc.extract_image(img[0])
        return d.get("colorspace"), d.get("ext")
    return None


def trial(label, **kw):
    doc = fitz.open(SRC)
    try:
        doc.rewrite_images(dpi_threshold=151, dpi_target=150, quality=72, **kw)
        cs = first_image_cs(doc)
        # render
        pix = doc[0].get_pixmap(dpi=50)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        px = img.getpixel((pix.width//2, int(pix.height*0.15)))
        print(f"{label:55s} -> cs(channels)={cs} centerpx={px}")
    except Exception as e:
        print(f"{label:55s} -> ERROR {e}")
    finally:
        doc.close()


# baseline current code
trial("CURRENT(lossy=T,lossless=T,color=T,gray=T,set_gray=F)",
      lossy=True, lossless=True, bitonal=False, color=True, gray=True, set_to_gray=False)

trial("lossy=T only, no lossless",
      lossy=True, lossless=False, bitonal=False, color=True, gray=True, set_to_gray=False)

trial("color=F",
      lossy=True, lossless=True, bitonal=False, color=False, gray=True, set_to_gray=False)

trial("gray=F",
      lossy=True, lossless=True, bitonal=False, color=True, gray=False, set_to_gray=False)

trial("defaults (only dpi+quality)")
