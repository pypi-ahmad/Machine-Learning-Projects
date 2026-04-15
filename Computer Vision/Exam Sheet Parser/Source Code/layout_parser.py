"""Layout-aware parsing for Exam Sheet Parser.

Classifies OCR blocks into structural roles (heading, question,
MCQ option, marks annotation, body text, section header) using
spatial heuristics and regex patterns.

This module is **purely rule-based** — no ML model is required.

Usage::

    from layout_parser import LayoutParser
    from config import ExamSheetConfig

    lp = LayoutParser(ExamSheetConfig())
    elements = lp.parse(ocr_blocks)
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ExamSheetConfig
from ocr_engine import OCRBlock


# ── Block roles ───────────────────────────────────────────────────
ROLE_HEADING = "heading"
ROLE_SECTION = "section"
ROLE_QUESTION = "question"
ROLE_MCQ_OPTION = "mcq_option"
ROLE_MARKS = "marks"
ROLE_BODY = "body"


@dataclass
class LayoutElement:
    """An OCR block annotated with its structural role."""

    role: str                       # one of the ROLE_* constants
    text: str
    confidence: float
    bbox: list[list[int]]
    centre: tuple[int, int]
    question_number: int | None = None
    option_letter: str | None = None
    marks_value: int | None = None
    block_index: int = 0


@dataclass
class QuestionBlock:
    """A parsed question with optional sub-elements."""

    number: int
    text: str
    marks: int | None = None
    options: list[str] = field(default_factory=list)
    option_letters: list[str] = field(default_factory=list)
    confidence: float = 0.0
    bbox: list[list[int]] = field(default_factory=list)
    body_lines: list[str] = field(default_factory=list)


class LayoutParser:
    """Classify OCR blocks into exam-sheet structural roles."""

    def __init__(self, cfg: ExamSheetConfig) -> None:
        self.cfg = cfg
        self._q_re = re.compile(cfg.question_number_pattern, re.IGNORECASE)
        self._mcq_re = re.compile(cfg.mcq_option_pattern)
        self._marks_re = re.compile(cfg.marks_pattern, re.IGNORECASE)
        self._section_kw = {k.lower() for k in cfg.section_keywords}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, blocks: list[OCRBlock]) -> list[LayoutElement]:
        """Classify every block and return annotated layout elements."""
        if not blocks:
            return []

        median_h = self._median_height(blocks)
        elements: list[LayoutElement] = []

        for idx, blk in enumerate(blocks):
            role, q_num, opt_letter, marks_val = self._classify(
                blk, median_h,
            )
            elements.append(LayoutElement(
                role=role,
                text=blk.text,
                confidence=blk.confidence,
                bbox=blk.bbox,
                centre=blk.centre,
                question_number=q_num,
                option_letter=opt_letter,
                marks_value=marks_val,
                block_index=idx,
            ))

        return elements

    def extract_questions(
        self, elements: list[LayoutElement],
    ) -> list[QuestionBlock]:
        """Group layout elements into structured question blocks."""
        questions: list[QuestionBlock] = []
        current: QuestionBlock | None = None

        for elem in elements:
            if elem.role == ROLE_QUESTION and elem.question_number is not None:
                # Start a new question
                if current is not None:
                    questions.append(current)
                current = QuestionBlock(
                    number=elem.question_number,
                    text=elem.text,
                    confidence=elem.confidence,
                    bbox=elem.bbox,
                )
                # Check for inline marks
                if elem.marks_value is not None:
                    current.marks = elem.marks_value

            elif current is not None:
                # Attach sub-elements to current question
                if elem.role == ROLE_MCQ_OPTION and elem.option_letter:
                    current.options.append(elem.text)
                    current.option_letters.append(elem.option_letter)
                elif elem.role == ROLE_MARKS and elem.marks_value is not None:
                    current.marks = elem.marks_value
                elif elem.role == ROLE_BODY:
                    if self._is_question_stub(current.text):
                        current.text = self._merge_question_text(current.text, elem.text)
                    else:
                        current.body_lines.append(elem.text)
                    if current.marks is None and elem.marks_value is not None:
                        current.marks = elem.marks_value

                # Update confidence to minimum
                current.confidence = min(current.confidence, elem.confidence)

        if current is not None:
            questions.append(current)

        return questions

    # ------------------------------------------------------------------
    # Classification logic
    # ------------------------------------------------------------------

    def _classify(
        self, blk: OCRBlock, median_h: float,
    ) -> tuple[str, int | None, str | None, int | None]:
        """Return (role, question_number, option_letter, marks_value)."""
        text = blk.text.strip()
        text_lower = text.lower()

        # 1. Marks annotation (e.g. "[5 marks]", "(3 pts)")
        marks_match = self._marks_re.search(text)
        marks_val = int(marks_match.group(1)) if marks_match else None

        if marks_val is not None and self._is_standalone_marks_text(text):
            return ROLE_MARKS, None, None, marks_val

        # 2. Question number (e.g. "Q1.", "3)", "Q. 12:")
        q_match = self._q_re.match(text)
        if q_match is None:
            q_match = re.match(r"^\s*Q\.?\s*(\d{1,3})[_\-\.]?\s*$", text, re.IGNORECASE)
        if q_match:
            q_num = int(q_match.group(1))
            return ROLE_QUESTION, q_num, None, marks_val

        # 3. MCQ option (e.g. "A)", "(B).", "c.")
        mcq_match = self._mcq_re.match(text)
        if mcq_match:
            letter = mcq_match.group(1).upper()
            return ROLE_MCQ_OPTION, None, letter, marks_val

        # 4. Section heading by keyword
        if self._is_section(text_lower):
            return ROLE_SECTION, None, None, marks_val

        # 5. Visual heading — significantly taller than median
        if median_h > 0 and blk.height > median_h * self.cfg.heading_font_ratio:
            return ROLE_HEADING, None, None, marks_val

        # 6. Explicit heading heuristic — short all-caps text
        if (
            len(text) <= 60
            and text == text.upper()
            and blk.height >= self.cfg.heading_min_height
            and sum(1 for c in text if c.isalpha()) >= 3
        ):
            return ROLE_HEADING, None, None, marks_val

        # 7. Standalone marks (block is just marks annotation)
        if marks_val is not None and len(text) < 25:
            return ROLE_MARKS, None, None, marks_val

        # 8. Default: body text
        return ROLE_BODY, None, None, marks_val

    def _is_section(self, text_lower: str) -> bool:
        """Check if text starts with a section keyword."""
        for kw in self._section_kw:
            if text_lower.startswith(kw):
                return True
        return False

    def _is_standalone_marks_text(self, text: str) -> bool:
        stripped = self._marks_re.sub("", text)
        stripped = re.sub(r"[\d\s\[\]\(\)\.:;,+\-]+", "", stripped)
        return len(stripped) <= 1

    @staticmethod
    def _is_question_stub(text: str) -> bool:
        return bool(re.match(r"^\s*Q\.?\s*\d{1,3}[_\-\.]?\s*$", text, re.IGNORECASE))

    @staticmethod
    def _merge_question_text(stub: str, body_text: str) -> str:
        clean_stub = re.sub(r"[_\-\.]?\s*$", "", stub.strip())
        return f"{clean_stub}. {body_text}".strip()

    @staticmethod
    def _median_height(blocks: list[OCRBlock]) -> float:
        heights = [b.height for b in blocks if b.height > 0]
        if not heights:
            return 0.0
        return float(np.median(heights))
