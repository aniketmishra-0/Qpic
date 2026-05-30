from app.models.schemas import DetectedQuestion, QuestionSegment
from app.services.vision_service import merge_detected_questions


def test_merge_detected_questions_deduplicates() -> None:
    raw = [
        DetectedQuestion(
            q_num="1",
            segments=[QuestionSegment(page=1, y_start_pct=10.0, y_end_pct=20.0)],
        ),
        DetectedQuestion(
            q_num="1",
            segments=[QuestionSegment(page=1, y_start_pct=11.0, y_end_pct=21.0)],
        ),
    ]
    merged = merge_detected_questions(raw)
    assert len(merged) == 1
    assert merged[0].q_num == "1"
    assert len(merged[0].segments) == 1


def test_merge_detected_questions_stitches_cross_page() -> None:
    raw = [
        DetectedQuestion(
            q_num="2",
            segments=[QuestionSegment(page=1, y_start_pct=80.0, y_end_pct=100.0)],
        ),
        DetectedQuestion(
            q_num="2",
            segments=[QuestionSegment(page=2, y_start_pct=0.0, y_end_pct=30.0)],
        ),
    ]
    merged = merge_detected_questions(raw)
    assert len(merged) == 1
    assert len(merged[0].segments) == 2
    assert [s.page for s in merged[0].segments] == [1, 2]


def test_merge_sorts_by_q_num() -> None:
    raw = [
        DetectedQuestion(q_num="10", segments=[QuestionSegment(page=1, y_start_pct=0.0, y_end_pct=10.0)]),
        DetectedQuestion(q_num="2", segments=[QuestionSegment(page=1, y_start_pct=20.0, y_end_pct=30.0)]),
    ]
    merged = merge_detected_questions(raw)
    assert [q.q_num for q in merged] == ["2", "10"]
