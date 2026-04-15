"""High-level pipeline for Exam Sheet Parser.

Orchestrates OCR, layout classification, and question extraction into a
single :class:`ExamSheetResult`.

Usage::

    from parser import ExamSheetPipeline

    pipeline = ExamSheetPipeline(cfg)
    result = pipeline.process(image)
    result, blocks, elements = pipeline.process_with_details(image)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ExamSheetConfig
from layout_parser import (
    LayoutElement,
    LayoutParser,
    QuestionBlock,
    ROLE_HEADING,
    ROLE_SECTION,
)
from ocr_engine import OCRBlock, OCREngine


@dataclass
class ExamSheetResult:
    """Complete exam sheet extraction result."""

    headings: list[str] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)
    questions: list[QuestionBlock] = field(default_factory=list)
    raw_text: str = ""
    num_blocks: int = 0
    num_questions: int = 0
    total_marks: int | None = None
    mean_confidence: float = 0.0


class ExamSheetPipeline:
    """Full exam sheet pipeline: OCR -> layout -> questions."""

    def __init__(self, cfg: ExamSheetConfig) -> None:
        self.cfg = cfg
        self._engine = OCREngine(cfg)
        self._layout = LayoutParser(cfg)

    def process(self, image: np.ndarray) -> ExamSheetResult:
        """Process a single exam sheet image."""
        blocks = self._engine.run(image)
        elements = self._layout.parse(blocks)
        return self._build_result(blocks, elements)

    def process_with_details(
        self, image: np.ndarray,
    ) -> tuple[ExamSheetResult, list[OCRBlock], list[LayoutElement]]:
        """Process and also return raw OCR blocks + layout elements."""
        blocks = self._engine.run(image)
        elements = self._layout.parse(blocks)
        result = self._build_result(blocks, elements)
        return result, blocks, elements

    def _build_result(
        self,
        blocks: list[OCRBlock],
        elements: list[LayoutElement],
    ) -> ExamSheetResult:
        if not blocks:
            return ExamSheetResult()

        # Extract headings and sections
        headings = [e.text for e in elements if e.role == ROLE_HEADING]
        sections = [e.text for e in elements if e.role == ROLE_SECTION]

        # Extract structured questions
        questions = self._layout.extract_questions(elements)

        # Total marks
        marks_values = [q.marks for q in questions if q.marks is not None]
        total_marks = sum(marks_values) if marks_values else None

        # Confidence
        confs = [b.confidence for b in blocks]
        mean_conf = float(np.mean(confs)) if confs else 0.0

        raw_text = self._engine.full_text(blocks)

        return ExamSheetResult(
            headings=headings,
            sections=sections,
            questions=questions,
            raw_text=raw_text,
            num_blocks=len(blocks),
            num_questions=len(questions),
            total_marks=total_marks,
            mean_confidence=mean_conf,
        )
