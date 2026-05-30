"""Repro: does crop-time padding bleed into the next question's first line?"""
import fitz
from app.services.detector.text_detector import TextDetector
from app.services.pdf_service import pdf_to_images
from app.services.crop_service import crop_and_stitch

def build_pdf(gap_pt: float) -> bytes:
    doc = fitz.open()
    p = doc.new_page(width=595, height=842)
    def line(txt, yy, x=40, size=11):
        p.insert_text((x, yy), txt, fontsize=size)
    y = 60
    line("19.  Which of the following is a correct statement?", y); y+=18
    line("(A) Brownian motion destabilizes sols.", y, x=70); y+=18
    line("(B) Any amount of dispersed phase can be added.", y, x=70); y+=18
    line("(C) Mixing two oppositely charged sols neutralizes.", y, x=70); y+=18
    line("(D) Presence of equal and similar charges provides stability.", y, x=70)
    y += gap_pt  # gap to next question
    q20_y = y
    line("20.  2.5 mL of 2/5 M weak monoacidic base is titrated. The", y); y+=18
    line("(A) opt a", y, x=70); y+=18
    line("(B) opt b", y, x=70); y+=40
    line("21.  Bounding question.", y)
    return doc.tobytes(), q20_y

for gap in (6, 12, 24):
    data, q20_y = build_pdf(gap)
    qs = TextDetector().detect(data, padding_px=20)
    q19 = next((q for q in qs if q.q_num == "19"), None)
    imgs = pdf_to_images(data, 200)
    if q19:
        crop = crop_and_stitch(page_images=imgs, question=q19, padding_px=20)
        # Where does Q19 crop bottom fall vs Q20 marker (in % of page)?
        seg = q19.segments[-1]
        q20_pct = q20_y / 842 * 100
        print(f"gap={gap}pt: Q19 seg y_end={seg.y_end_pct:.1f}% | Q20 marker at {q20_pct:.1f}% | crop_px_h={crop.size[1]}")
        # crop bottom in page-% = y_end + padding(20px@200dpi -> in pt -> %)
        pad_pct = (20 * 72/200) / 842 * 100
        print(f"         crop bottom ~= {seg.y_end_pct + pad_pct:.1f}%  (bleeds into Q20 if > {q20_pct:.1f}%)")
