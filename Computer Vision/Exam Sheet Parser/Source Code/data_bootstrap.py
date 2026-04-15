"""Dataset bootstrap for Exam Sheet Parser.

Prepares a local evaluation set for the OCR + layout parsing pipeline.
It attempts the registered public dataset first and falls back to a
synthetic exam-sheet corpus when that source is unavailable.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("exam_sheet.data_bootstrap")

PROJECT_KEY = "exam_sheet_parser"
DATA_ROOT = REPO_ROOT / "data" / PROJECT_KEY

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_exam_sheet_dataset(*, force: bool = False) -> Path:
    """Download or generate the exam sheet evaluation dataset."""
    ready_marker = DATA_ROOT / "processed" / ".ready"
    if ready_marker.exists() and not force:
        log.info(
            "[%s] Dataset already prepared at %s -- skipping",
            PROJECT_KEY, DATA_ROOT,
        )
        return DATA_ROOT

    if force and DATA_ROOT.exists():
        for child in DATA_ROOT.iterdir():
            if child.is_dir():
                for nested in sorted(child.rglob("*"), reverse=True):
                    if nested.is_file():
                        nested.unlink()
                    elif nested.is_dir():
                        nested.rmdir()
                child.rmdir()
            else:
                child.unlink()

    data_path = DATA_ROOT
    raw_dir = data_path / "raw"
    processed_dir = data_path / "processed"
    images_dir = processed_dir / "images"
    raw_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    image_count = _try_prepare_public_dataset(data_path, images_dir, force=force)
    dataset_source = "public"
    if image_count == 0:
        image_count = _create_synthetic_dataset(images_dir)
        dataset_source = "synthetic"

    _write_info(data_path, dataset_source=dataset_source, image_count=image_count)

    ready_marker.write_text(
        time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8",
    )
    log.info("[%s] Dataset prepared at %s", PROJECT_KEY, DATA_ROOT)
    return DATA_ROOT


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _try_prepare_public_dataset(
    data_path: Path,
    images_dir: Path,
    *,
    force: bool,
) -> int:
    try:
        from scripts.download_data import ensure_dataset as _ensure
    except Exception as exc:
        log.warning("[%s] Public dataset helper unavailable: %s", PROJECT_KEY, exc)
        return 0

    try:
        public_path = _ensure(PROJECT_KEY, force=force)
    except Exception as exc:
        log.warning("[%s] Public dataset download failed: %s", PROJECT_KEY, exc)
        return 0

    count = 0
    for img in public_path.rglob("*"):
        if img.suffix.lower() not in IMAGE_EXTS or "processed" in img.parts:
            continue
        dst = images_dir / img.name
        if dst.exists():
            continue
        dst.write_bytes(img.read_bytes())
        count += 1

    log.info("[%s] Prepared %d public images", PROJECT_KEY, count)
    return count


def _create_synthetic_dataset(images_dir: Path) -> int:
    pages = _synthetic_pages()
    labels = []
    for index, page in enumerate(pages, start=1):
        image = np.full((1600, 1200, 3), 255, dtype=np.uint8)
        _draw_page(image, page)
        image_name = f"exam_sheet_{index:02d}.png"
        image_path = images_dir / image_name
        cv2.imwrite(str(image_path), image)
        labels.append(
            {
                "image": image_name,
                "heading": page[0]["text"],
                "question_count": sum(
                    1 for line in page if line.get("role") == "question"
                ),
                "total_marks": sum(line.get("marks", 0) for line in page),
            }
        )

    labels_path = images_dir.parent / "ocr_labels.json"
    labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    log.info("[%s] Created %d synthetic exam sheets", PROJECT_KEY, len(pages))
    return len(pages)


def _draw_page(image: np.ndarray, page: list[dict[str, object]]) -> None:
    y = 110
    for line in page:
        font_scale = float(line.get("font_scale", 1.0))
        thickness = int(line.get("thickness", 2))
        cv2.putText(
            image,
            str(line["text"]),
            (90, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (25, 25, 25),
            thickness,
            cv2.LINE_AA,
        )
        y += int(line.get("line_gap", 62))


def _synthetic_pages() -> list[list[dict[str, object]]]:
    return [
        [
            _line("MIDTERM EXAMINATION 2026", role="heading", font_scale=1.35, thickness=3, line_gap=90),
            _line("Mathematics - Grade 10", role="heading", font_scale=1.0, thickness=2, line_gap=70),
            _line("Instructions: Answer all questions clearly.", role="section", font_scale=0.85, line_gap=55),
            _line("Section A", role="section", font_scale=0.95, thickness=2, line_gap=65),
            _line("Q1. Define algebra. [5 marks]", role="question", marks=5),
            _line("Q2. State one factor of twelve. [4 marks]", role="question", marks=4),
            _line("Q3. Which graph is linear? [2 marks]", role="question", marks=2),
            _line("A) A straight line", role="mcq_option", font_scale=0.85, line_gap=48),
            _line("B) A circle", role="mcq_option", font_scale=0.85, line_gap=48),
            _line("C) A curve", role="mcq_option", font_scale=0.85, line_gap=48),
            _line("D) A spiral", role="mcq_option", font_scale=0.85, line_gap=60),
            _line("Section B", role="section", font_scale=0.95, thickness=2, line_gap=65),
            _line("Q4. Explain mean and median. [6 marks]", role="question", marks=6),
            _line("Q5. Describe a square number. [8 marks]", role="question", marks=8),
        ],
        [
            _line("SCIENCE FINAL PAPER", role="heading", font_scale=1.3, thickness=3, line_gap=88),
            _line("General Science - Secondary Level", role="heading", font_scale=0.95, thickness=2, line_gap=68),
            _line("Note: Draw neat diagrams where required.", role="section", font_scale=0.82, line_gap=54),
            _line("Section A", role="section", font_scale=0.95, thickness=2, line_gap=65),
            _line("1) Define photosynthesis. [3 marks]", role="question", marks=3),
            _line("2) Name the planet known as the red planet. [1 mark]", role="question", marks=1),
            _line("3) Select the state of matter for oxygen. [2 marks]", role="question", marks=2),
            _line("A) Solid", role="mcq_option", font_scale=0.84, line_gap=47),
            _line("B) Liquid", role="mcq_option", font_scale=0.84, line_gap=47),
            _line("C) Gas", role="mcq_option", font_scale=0.84, line_gap=47),
            _line("D) Plasma", role="mcq_option", font_scale=0.84, line_gap=58),
            _line("Section B", role="section", font_scale=0.95, thickness=2, line_gap=65),
            _line("4) Explain why metals conduct electricity. [5 marks]", role="question", marks=5),
            _line("5) Describe one use of microbes in food. [4 marks]", role="question", marks=4),
        ],
        [
            _line("ENGLISH LANGUAGE TEST", role="heading", font_scale=1.28, thickness=3, line_gap=86),
            _line("Reading and Grammar", role="heading", font_scale=0.96, thickness=2, line_gap=66),
            _line("Instructions: Read each passage before answering.", role="section", font_scale=0.82, line_gap=54),
            _line("Part I", role="section", font_scale=0.94, thickness=2, line_gap=64),
            _line("Q1. Identify the adjective in the sentence. [2 marks]", role="question", marks=2),
            _line("A) quickly", role="mcq_option", font_scale=0.84, line_gap=47),
            _line("B) beautiful", role="mcq_option", font_scale=0.84, line_gap=47),
            _line("C) walked", role="mcq_option", font_scale=0.84, line_gap=47),
            _line("D) under", role="mcq_option", font_scale=0.84, line_gap=58),
            _line("Q2. Rewrite the sentence in passive voice. [4 marks]", role="question", marks=4),
            _line("Q3. Write a short paragraph on teamwork. [6 marks]", role="question", marks=6),
            _line("Part II", role="section", font_scale=0.94, thickness=2, line_gap=64),
            _line("Q4. Summarise the passage in three lines. [8 marks]", role="question", marks=8),
        ],
    ]


def _line(text: str, **kwargs: object) -> dict[str, object]:
    line = {"text": text, "line_gap": 60, "font_scale": 0.9, "thickness": 2}
    line.update(kwargs)
    return line


def _write_info(data_path: Path, *, dataset_source: str, image_count: int) -> None:
    info = {
        "dataset_key": PROJECT_KEY,
        "source_type": dataset_source,
        "description": "Exam-sheet layout evaluation set with headings, questions, MCQ blocks, and marks.",
        "image_count": image_count,
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    info_path = data_path / "dataset_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare the exam sheet dataset")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the dataset even if the ready marker already exists",
    )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    data_root = ensure_exam_sheet_dataset(force=parser.parse_args(argv).force)
    print(data_root)


if __name__ == "__main__":
    main()
