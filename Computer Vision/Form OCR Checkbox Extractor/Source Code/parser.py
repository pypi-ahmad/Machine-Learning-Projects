"""Form parser — orchestrates OCR + checkbox detection.

Associates each detected checkbox/radio with its nearest text
label, groups results into structured form fields, and returns
a single :class:`FormParseResult`.

Usage::

    from parser import FormParser

    parser = FormParser(cfg)
    result = parser.parse(image)
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkbox_detector import CheckboxDetector, ControlState, ControlType, FormControl
from config import FormCheckboxConfig
from ocr_engine import OCRBlock, OCREngine


@dataclass
class CheckboxField:
    """A checkbox/radio with its associated text label."""

    label: str
    state: str                  # "checked" | "unchecked" | "unknown"
    control_type: str           # "checkbox" | "radio"
    confidence: float           # OCR confidence of the label
    fill_ratio: float
    bbox: tuple[int, int, int, int]
    label_block_idx: int = -1   # index into OCR blocks


@dataclass
class TextField:
    """A plain text field extracted from OCR."""

    name: str
    value: str
    confidence: float
    source_block_idx: int = -1


@dataclass
class FormParseResult:
    """Complete form extraction result."""

    text_fields: dict[str, TextField] = field(default_factory=dict)
    checkbox_fields: list[CheckboxField] = field(default_factory=list)
    raw_text: str = ""
    num_ocr_blocks: int = 0
    num_checkboxes: int = 0
    num_checked: int = 0


class FormParser:
    """Orchestrate checkbox detection + OCR + label association."""

    def __init__(self, cfg: FormCheckboxConfig) -> None:
        self.cfg = cfg
        self._detector = CheckboxDetector(cfg)
        self._engine = OCREngine(cfg)

    def parse(self, image: np.ndarray) -> FormParseResult:
        """Run full extraction pipeline on a single form image."""
        blocks = self._engine.run(image)
        controls = self._detector.detect(image)

        checkbox_fields = self._associate_labels(controls, blocks)
        text_fields = self._extract_text_fields(blocks, controls)

        raw_text = self._engine.full_text(blocks)

        return FormParseResult(
            text_fields=text_fields,
            checkbox_fields=checkbox_fields,
            raw_text=raw_text,
            num_ocr_blocks=len(blocks),
            num_checkboxes=len(checkbox_fields),
            num_checked=sum(
                1 for cb in checkbox_fields if cb.state == "checked"
            ),
        )

    @property
    def ocr_blocks(self) -> list[OCRBlock]:
        """Return OCR blocks from the most recent run (for visualisation)."""
        return getattr(self, "_last_blocks", [])

    @property
    def controls(self) -> list[FormControl]:
        """Return detected controls from the most recent run."""
        return getattr(self, "_last_controls", [])

    def parse_with_details(
        self, image: np.ndarray,
    ) -> tuple[FormParseResult, list[OCRBlock], list[FormControl]]:
        """Parse and also return raw OCR blocks + controls for overlay."""
        blocks = self._engine.run(image)
        controls = self._detector.detect(image)

        self._last_blocks = blocks
        self._last_controls = controls

        checkbox_fields = self._associate_labels(controls, blocks)
        text_fields = self._extract_text_fields(blocks, controls)
        raw_text = self._engine.full_text(blocks)

        result = FormParseResult(
            text_fields=text_fields,
            checkbox_fields=checkbox_fields,
            raw_text=raw_text,
            num_ocr_blocks=len(blocks),
            num_checkboxes=len(checkbox_fields),
            num_checked=sum(
                1 for cb in checkbox_fields if cb.state == "checked"
            ),
        )
        return result, blocks, controls

    # ------------------------------------------------------------------
    # Label association
    # ------------------------------------------------------------------

    def _associate_labels(
        self,
        controls: list[FormControl],
        blocks: list[OCRBlock],
    ) -> list[CheckboxField]:
        """Match each checkbox/radio to its nearest text label."""
        fields: list[CheckboxField] = []
        used_blocks: set[int] = set()

        for ctrl in controls:
            best_idx = -1
            best_dist = float("inf")
            best_block: OCRBlock | None = None

            for i, blk in enumerate(blocks):
                if i in used_blocks:
                    continue
                dist = self._label_distance(ctrl, blk)
                if dist < best_dist and dist <= self.cfg.label_max_distance:
                    best_dist = dist
                    best_idx = i
                    best_block = blk

            label = ""
            conf = 0.0
            if best_block is not None:
                label = best_block.text
                conf = best_block.confidence
                used_blocks.add(best_idx)

            fields.append(CheckboxField(
                label=label,
                state=ctrl.state.value,
                control_type=ctrl.control_type.value,
                confidence=conf,
                fill_ratio=ctrl.fill_ratio,
                bbox=ctrl.bbox,
                label_block_idx=best_idx,
            ))

        return fields

    def _label_distance(self, ctrl: FormControl, blk: OCRBlock) -> float:
        """Distance metric weighted by preferred label direction."""
        cx, cy = ctrl.centre
        bx, by = blk.centre

        dx = bx - cx
        dy = by - cy

        # Penalise labels that are on the wrong side
        direction = self.cfg.label_prefer_direction
        if direction == "right" and dx < 0:
            dx *= 2.0   # double penalty for left-side labels
        elif direction == "left" and dx > 0:
            dx *= 2.0

        # Penalise labels that are far vertically
        return math.sqrt(dx * dx + (dy * 3.0) ** 2)

    # ------------------------------------------------------------------
    # Text field extraction (label: value patterns)
    # ------------------------------------------------------------------

    _LABEL_PATTERNS = {
        "name": ["name", "full name", "applicant", "first name", "last name"],
        "date": ["date", "dated", "date of birth", "dob"],
        "address": ["address", "street", "city", "zip", "postal"],
        "phone": ["phone", "tel", "telephone", "mobile", "contact"],
        "email": ["email", "e-mail", "mail"],
        "signature": ["signature", "sign", "signed"],
        "id_number": ["id", "id no", "identification", "ssn", "social security"],
    }

    def _extract_text_fields(
        self,
        blocks: list[OCRBlock],
        controls: list[FormControl],
    ) -> dict[str, TextField]:
        """Extract labelled text fields via keyword matching."""
        # Blocks used by checkboxes should not be re-used for text fields
        checkbox_zones = {
            (c.bbox[0], c.bbox[1], c.bbox[2], c.bbox[3]) for c in controls
        }

        fields: dict[str, TextField] = {}
        for i, blk in enumerate(blocks):
            text_lower = blk.text.lower().strip()

            for field_name, keywords in self._LABEL_PATTERNS.items():
                if field_name in fields:
                    continue
                for kw in keywords:
                    if kw in text_lower:
                        # Try to find the value on the same line or next block
                        value = self._extract_value(blk, i, blocks, kw)
                        if value:
                            fields[field_name] = TextField(
                                name=field_name,
                                value=value,
                                confidence=blk.confidence,
                                source_block_idx=i,
                            )
                        break

        return fields

    def _extract_value(
        self,
        label_block: OCRBlock,
        idx: int,
        blocks: list[OCRBlock],
        keyword: str,
    ) -> str:
        """Extract field value from the label block or adjacent block."""
        text = label_block.text.strip()

        # Check for "Label: Value" or "Label Value" on same line
        for sep in [":", "-", "="]:
            if sep in text:
                parts = text.split(sep, 1)
                val = parts[1].strip()
                if val and len(val) > 1:
                    return val

        # Value might be after the keyword on the same block
        kw_lower = keyword.lower()
        t_lower = text.lower()
        kw_end = t_lower.find(kw_lower)
        if kw_end >= 0:
            after = text[kw_end + len(keyword) :].strip()
            if after and len(after) > 1:
                return after

        # Check next block on roughly the same y-coordinate or just below
        if idx + 1 < len(blocks):
            next_blk = blocks[idx + 1]
            dy = abs(next_blk.centre[1] - label_block.centre[1])
            if dy < 40 and next_blk.centre[0] > label_block.centre[0]:
                return next_blk.text.strip()

        return ""
