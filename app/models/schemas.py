"""Pydantic request/response schemas."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class QuestionSegment(BaseModel):
    """A single question fragment on a page.

    ``x_start_pct`` / ``x_end_pct`` describe the horizontal extent of the
    column the fragment lives in. They default to the full page width so
    single-column layouts behave exactly as before.
    """

    page: int
    y_start_pct: float
    y_end_pct: float
    x_start_pct: float = 0.0
    x_end_pct: float = 100.0


class DetectedQuestion(BaseModel):
    """A detected question (or solution) with one or more page segments.

    ``option_labels`` records which MCQ option letters (A-D) were seen inside the
    question's content during text/OCR detection, e.g. ``"ABCD"`` for a complete
    question or ``"AC"`` when only the left column of a 2-up option grid was
    captured. It is informational only (default empty) and lets the review step
    flag a crop that probably lost its right-hand options. Detectors that don't
    track options (the AI tier, manual items) leave it empty.
    """

    q_num: str
    segments: list[QuestionSegment]
    is_solution: bool = False
    option_labels: str = ""
    source: Literal["auto", "manual"] = "auto"

class CropResponse(BaseModel):
    """Response after creating a crop job.

    ``download_url`` is the combined archive (questions + solutions) and is kept
    for backward compatibility. ``questions_download_url`` /
    ``solutions_download_url`` point to the per-type archives and are present
    only when that side produced at least one crop.
    """

    job_id: str
    total_questions: int
    stitched_questions: int
    method_used: Literal["text", "ocr", "ai"]
    download_url: str
    questions_download_url: Optional[str] = None
    solutions_download_url: Optional[str] = None
    questions_count: int = 0
    solutions_count: int = 0


# --- Smart analyze / manual-review / finalize flow ---------------------------


class PageInfo(BaseModel):
    """Geometry of a single PDF page, used by the manual-crop canvas."""

    page: int  # 1-indexed
    width_pt: float
    height_pt: float
    preview_url: str


class AnalyzedItem(BaseModel):
    """A detected (or user-added) item returned for on-screen review.

    ``source`` distinguishes the pipeline's own detections ("auto") from items
    the user draws in the review popup ("manual"). ``flagged`` marks an item the
    review heuristics are unsure about (a likely duplicate or a suspiciously
    tiny crop) so the UI can highlight it.
    """

    q_num: str
    is_solution: bool = False
    segments: list[QuestionSegment]
    source: Literal["auto", "manual"] = "auto"
    flagged: bool = False
    flag_reason: Optional[str] = None


class ReviewNote(BaseModel):
    """A single human-readable thing to check in the review popup."""

    kind: Literal["duplicate", "gap", "tiny", "incomplete", "low_confidence"]
    message: str
    q_num: Optional[str] = None
    page: Optional[int] = None
    is_solution: bool = False


class AnalyzeResponse(BaseModel):
    """Result of the smart analyze pass, before final ZIP generation."""

    job_id: str
    total_pages: int
    method_used: Literal["text", "ocr", "ai"]
    pages: list[PageInfo]
    items: list[AnalyzedItem]
    notes: list[ReviewNote]
    needs_review: bool


class SnapRequest(BaseModel):
    """A roughly drawn box to tighten to the content inside it."""

    job_id: str
    page: int
    x_start_pct: float
    x_end_pct: float
    y_start_pct: float
    y_end_pct: float


class SnapResponse(BaseModel):
    """The content-tightened region (page percentages)."""

    x_start_pct: float
    x_end_pct: float
    y_start_pct: float
    y_end_pct: float


class FinalizeItem(BaseModel):
    """One item to crop in the finalize step (auto-kept or manually drawn).

    ``source`` mirrors the review item's origin: ``"manual"`` for boxes the user
    drew/re-selected by hand, ``"auto"`` for kept pipeline detections. Finalize
    uses it to apply manual-only post-processing (content left-alignment of
    stitched column-split parts) without touching auto crops.
    """

    q_num: str
    is_solution: bool = False
    segments: list[QuestionSegment]
    source: Literal["auto", "manual"] = "auto"


class FinalizeRequest(BaseModel):
    """Payload that turns a reviewed item list into the downloadable ZIP."""

    job_id: str
    items: list[FinalizeItem]
    dpi: int = 200
    padding: int = 20
    question_prefix: str = "Q"
    solution_prefix: str = "S"
    start_number: int = 1
    image_format: Literal["png", "jpg", "jpeg"] = "png"
    jpg_quality: int = 90


class HealthResponse(BaseModel):
    status: str
    tesseract_available: bool
    ai_available: bool
    version: str
    ai_provider: Optional[str] = None
    ai_model: Optional[str] = None


# --- Batch rename tool -------------------------------------------------------


class RenamePlanItem(BaseModel):
    """A single before/after pair in a rename preview."""

    original: str
    renamed: str


class RenamePreviewResponse(BaseModel):
    """Preview of how a batch of files will be renamed."""

    count: int
    items: list[RenamePlanItem]


class PdfImageItem(BaseModel):
    """One PDF page rendered to a PNG, returned as an inline data URL.

    ``data_url`` is a ready-to-use ``data:image/png;base64,…`` string so the
    browser can show a thumbnail and turn it back into a File for the rename
    batch — no extra round-trip to fetch the bytes.
    """

    name: str
    data_url: str
    width: int
    height: int
    size: int


class PdfToImagesResponse(BaseModel):
    """All pages of an uploaded PDF, rasterised to PNG images."""

    count: int
    images: list[PdfImageItem]


class RenameSessionResponse(BaseModel):
    """A freshly created upload session for a large rename batch."""

    session_id: str


class RenameUploadResponse(BaseModel):
    """Acknowledges a chunk of files appended to a rename session."""

    session_id: str
    received: int  # files accepted in this request
    total: int  # files staged in the session so far


class RenameFinalizeResponse(BaseModel):
    """The packed ZIP is ready; ``download_url`` streams it to the client."""

    session_id: str
    count: int
    download_url: str
