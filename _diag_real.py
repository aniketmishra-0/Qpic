"""Compare the real source vs compressed PDFs page 1 to find the darkening."""

import fitz
from PIL import Image

SRC = "/Users/Chrome/Current Affairs (Week 139th) Class Notes.pdf"
COMP = "/Users/Chrome/compressed (3).pdf"


def info(path, label):
    doc = fitz.open(path)
    page = doc[0]
    imgs = page.get_images(full=True)
    print(f"\n=== {label} ===  pages={len(doc)} page0_images={len(imgs)}")
    for img in imgs[:6]:
        xref = img[0]
        smask = img[1]
        cs = img[5]
        bpc = img[7] if len(img) > 7 else "?"
        # get pixmap meta
        try:
            d = doc.extract_image(xref)
            print(f"  xref={xref} smask={smask} cs={cs} ext={d.get('ext')} "
                  f"colorspace={d.get('colorspace')} w={d.get('width')} h={d.get('height')}")
        except Exception as e:
            print(f"  xref={xref} smask={smask} cs={cs} err={e}")
    pix = page.get_pixmap(dpi=60)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    img.save(f"/tmp/{label}.png")
    # sample a grid
    print("  samples:", [img.getpixel((pix.width*fx//1, 5)) for fx in []] )
    for (fx, fy, name) in [(0.5, 0.15, "title-area"), (0.1, 0.1, "tl"), (0.5,0.5,"center")]:
        print(f"    {name}={img.getpixel((int(pix.width*fx), int(pix.height*fy)))}")
    doc.close()


info(SRC, "ORIGINAL")
info(COMP, "COMPRESSED")
print("\nsaved /tmp/ORIGINAL.png and /tmp/COMPRESSED.png")
