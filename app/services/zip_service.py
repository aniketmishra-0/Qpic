"""ZIP packaging utilities."""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Disk filenames for the per-job archives. These are job-scoped so they never
# collide between jobs; the user-facing download name (e.g. ``Q.zip`` /
# ``QScombined.zip``) is applied by the API/frontend at download time.
COMBINED_ZIP = "qpic_crops_{job_id}.zip"
QUESTIONS_ZIP = "qpic_questions_{job_id}.zip"
SOLUTIONS_ZIP = "qpic_solutions_{job_id}.zip"


def _write_zip(zip_path: Path, image_paths: list[Path]) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for img_path in image_paths:
            zf.write(img_path, arcname=img_path.name)


def create_zip(image_paths: list[Path], job_id: str, output_dir: Path) -> Path:
    """Create a single combined ZIP containing all cropped images for a job."""

    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / COMBINED_ZIP.format(job_id=job_id)
    _write_zip(zip_path, image_paths)
    logger.info("created_zip=%s files=%s", zip_path.name, len(image_paths))
    return zip_path


def create_zip_set(
    question_paths: list[Path],
    solution_paths: list[Path],
    job_id: str,
    output_dir: Path,
    extra_paths: list[Path] | None = None,
) -> dict[str, Path]:
    """Create up to three ZIPs for a job and return the ones produced.

    Always writes the combined archive (questions + solutions). The
    questions-only and solutions-only archives are written only when that side
    actually has crops, so a questions-only paper doesn't ship an empty
    solutions ZIP. The returned mapping uses the keys ``"questions"``,
    ``"solutions"`` and ``"combined"``.

    ``extra_paths`` are non-image sidecar files (e.g. the answer-sheet
    ``answers.csv`` / ``answers.json``) bundled into the **questions** and
    **combined** archives — never the solutions-only one, since the answer sheet
    is keyed to the question images. They're omitted when the questions side is
    empty so a solutions-only run doesn't ship a dangling sheet.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    extras = list(extra_paths or [])

    if question_paths:
        q_zip = output_dir / QUESTIONS_ZIP.format(job_id=job_id)
        _write_zip(q_zip, question_paths + extras)
        result["questions"] = q_zip

    if solution_paths:
        s_zip = output_dir / SOLUTIONS_ZIP.format(job_id=job_id)
        _write_zip(s_zip, solution_paths)
        result["solutions"] = s_zip

    combined = output_dir / COMBINED_ZIP.format(job_id=job_id)
    combined_extras = extras if question_paths else []
    _write_zip(combined, question_paths + solution_paths + combined_extras)
    result["combined"] = combined

    logger.info(
        "created_zip_set job_id=%s questions=%s solutions=%s combined=%s extras=%s",
        job_id,
        len(question_paths),
        len(solution_paths),
        len(question_paths) + len(solution_paths),
        len(extras),
    )
    return result
