import fitz

from app.services.detector.text_detector import TextDetector


def _make_pdf_bytes(pages: list[list[tuple[float, str]]]) -> bytes:
    doc = fitz.open()
    for lines in pages:
        page = doc.new_page()
        for y, text in lines:
            page.insert_text((72, y), text, fontsize=12)
    data = doc.tobytes()
    doc.close()
    return data


def test_detects_numbered_questions() -> None:
    pdf_bytes = _make_pdf_bytes(
        [
            [
                (72, "1. What is 2+2?"),
                (200, "2. Next question"),
            ]
        ]
    )
    detector = TextDetector()
    questions = detector.detect(pdf_bytes, padding_px=20)
    assert [q.q_num for q in questions] == ["1", "2"]


def test_detects_q_prefix() -> None:
    pdf_bytes = _make_pdf_bytes([[ (72, "Q1. Hello world"), (200, "Q2) Another") ]])
    detector = TextDetector()
    questions = detector.detect(pdf_bytes, padding_px=0)
    assert [q.q_num for q in questions] == ["1", "2"]


def test_cross_page_question_has_two_segments() -> None:
    pdf_bytes = _make_pdf_bytes(
        [
            [(760, "1. This question continues"), (800, "(A) option")],
            [(72, "continued option"), (200, "2. Next question")],
        ]
    )
    detector = TextDetector()
    questions = detector.detect(pdf_bytes, padding_px=0)

    q1 = next(q for q in questions if q.q_num == "1")
    assert len(q1.segments) == 2
    assert [s.page for s in q1.segments] == [1, 2]


def test_segment_y_values_are_valid() -> None:
    pdf_bytes = _make_pdf_bytes(
        [
            [(72, "1. One"), (200, "2. Two")],
            [(72, "3. Three")],
        ]
    )
    detector = TextDetector()
    questions = detector.detect(pdf_bytes, padding_px=0)

    for q in questions:
        for seg in q.segments:
            assert 0.0 <= seg.y_start_pct <= 100.0
            assert 0.0 <= seg.y_end_pct <= 100.0
            assert seg.y_end_pct > seg.y_start_pct
