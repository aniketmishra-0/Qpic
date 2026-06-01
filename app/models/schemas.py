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
    # True when an answer sheet (answers.csv + answers.json mapping each
    # question image to its correct option) was bundled into the download.
    answer_sheet_included: bool = False
    # Number of questions the answer sheet carries a correct option for.
    answers_count: int = 0


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
    # Number of answers parsed from the paper's answer key (0 when no key was
    # found). Lets the review UI tell the user up front whether the finalized
    # download will include an answer sheet.
    answer_key_count: int = 0


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
    # When False, skip the answer-sheet (answers.csv/json) even if a key was
    # found at analyze time. Defaults True so the sheet ships by default.
    answer_sheet: bool = True


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


# --- PDF power tools: Compress / Edit / Preflight ----------------------------


class CompressResponse(BaseModel):
    """Result of a PDF compression job."""

    job_id: str
    original_size: int
    compressed_size: int
    ratio: float  # fraction of original size removed (0.0-1.0)
    level: str
    target_met: Optional[bool] = None
    note: str = ""
    download_url: str


class EditableSpanModel(BaseModel):
    """One editable text run on a page, with its geometry and style."""

    id: str
    page: int
    text: str
    bbox: list[float]  # [x0, y0, x1, y1] in PDF points
    font: str
    size: float
    color: int
    bold: bool = False
    italic: bool = False


class EditPageModel(BaseModel):
    """Geometry + preview for one page in the editor."""

    page: int
    width: float
    height: float
    preview_url: str


class EditExtractResponse(BaseModel):
    """All editable text spans for a PDF opened in the editor."""

    job_id: str
    has_text: bool
    pages: list[EditPageModel]
    spans: list[EditableSpanModel]


class EditOpModel(BaseModel):
    """A single span edit submitted from the editor."""

    page: int
    bbox: list[float]
    new_text: str
    font: Optional[str] = None
    size: Optional[float] = None
    color: Optional[int] = None


class OperationModel(BaseModel):
    """A single Acrobat-style edit operation submitted from the editor.

    ``type`` is one of: ``edit_text``, ``add_text``, ``add_image``,
    ``add_link``, ``erase``.
    """

    type: str
    page: int
    bbox: list[float]
    text: str = ""
    font: Optional[str] = None
    size: Optional[float] = None
    color: Optional[int] = None
    bold: bool = False
    italic: bool = False
    align: int = 0
    image_b64: Optional[str] = None
    url: Optional[str] = None
    fill: Optional[int] = None


class EditApplyRequest(BaseModel):
    """Payload that applies a set of in-place text edits to a job's PDF.

    Either ``edits`` (legacy text-only) or ``operations`` (full Acrobat-style
    set) may be supplied; ``operations`` wins when both are present.
    """

    job_id: str
    edits: list[EditOpModel] = []
    operations: list[OperationModel] = []


class EditApplyResponse(BaseModel):
    """The edited PDF is ready for download."""

    job_id: str
    edits_applied: int
    download_url: str


class OcrResponse(BaseModel):
    """Result of adding a searchable OCR text layer to a PDF."""

    job_id: str
    pages_ocred: int
    languages: str
    note: str
    download_url: str


class PreflightCheckModel(BaseModel):
    id: str
    title: str
    status: str  # ok | warn | fail | info
    detail: str


class PreflightFontModel(BaseModel):
    name: str
    type: str
    embedded: bool
    subset: bool


class PreflightImageModel(BaseModel):
    page: int
    width: int
    height: int
    dpi: float
    colorspace: str
    bpc: int


class PreflightPageDetail(BaseModel):
    """Per-page geometry detail for the Preflight Check table."""

    page: int
    w_mm: float
    h_mm: float
    w_pt: float
    h_pt: float
    w_px: int
    h_px: int
    format: str  # "A4" | "A3" | "Letter" | "Legal" | "A5" | "Custom"
    orientation: str  # "Portrait" | "Landscape"


class PreflightResponse(BaseModel):
    """Full read-only preflight report for a PDF."""

    verdict: str  # pass | warn | fail
    page_count: int
    page_sizes: list[str]
    file_size: int
    is_encrypted: bool
    has_text_layer: bool
    checks: list[PreflightCheckModel]
    fonts: list[PreflightFontModel]
    images: list[PreflightImageModel]
    metadata: dict[str, str]
    # All distinct page-size labels + a flag so the UI can offer the one-click
    # "Fix page sizes" action when the document mixes geometries.
    distinct_page_sizes: list[str] = []
    mixed_page_sizes: bool = False
    # Per-page detailed geometry for the Preflight Check modal table.
    page_details: list[PreflightPageDetail] = []


class PreflightFixResponse(BaseModel):
    """Result of normalizing a PDF's pages to one uniform size."""

    job_id: str
    target_label: str
    target_width: float   # PDF points
    target_height: float  # PDF points
    pages_total: int
    pages_changed: int
    note: str
    download_url: str
