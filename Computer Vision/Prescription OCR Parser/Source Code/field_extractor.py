"""Prescription field extractor.

Pattern-based extraction of medicine names, dosages, frequencies,
routes, durations, and instructions from OCR text blocks.

This module is **informational only** and must NOT be used for
clinical decision-making.  Always verify with a licensed
healthcare professional.

Usage::

    from field_extractor import FieldExtractor
    from config import PrescriptionConfig

    extractor = FieldExtractor(PrescriptionConfig())
    fields = extractor.extract(blocks)
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PrescriptionConfig
from ocr_engine import OCRBlock


@dataclass
class ExtractedField:
    """Single extracted prescription field."""

    name: str               # field name (e.g. "medicine_name")
    value: str
    confidence: float       # OCR confidence of source block
    source_text: str        # original OCR text
    source_block_idx: int   # index into OCR blocks list


@dataclass
class MedicineEntry:
    """A single medicine item with its details."""

    medicine_name: str = ""
    dosage: str = ""
    frequency: str = ""
    duration: str = ""
    route: str = ""
    instructions: str = ""
    confidence: float = 0.0
    source_blocks: list[int] = field(default_factory=list)


class FieldExtractor:
    """Extract prescription fields from OCR blocks using pattern matching.

    **DISCLAIMER:** This tool is for informational and educational
    purposes only.  It does not provide medical advice and must not
    be used for diagnosis, treatment, or clinical workflows.
    """

    def __init__(self, cfg: PrescriptionConfig) -> None:
        self.cfg = cfg
        self._dosage_re = self._compile_dosage_patterns(cfg.dosage_patterns)
        self._freq_set = {k.lower() for k in cfg.frequency_keywords}
        self._route_set = {k.lower() for k in cfg.route_keywords}
        self._dur_set = {k.lower() for k in cfg.duration_keywords}
        self._instr_set = {k.lower() for k in cfg.instruction_keywords}
        self._non_medicine_terms = {
            "for", "day", "days", "week", "weeks", "month", "months",
            "informational sample only", "verify with", "a licensed clinician",
            "licensed clinician", "sample only",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self, blocks: list[OCRBlock],
    ) -> tuple[list[MedicineEntry], dict[str, ExtractedField]]:
        """Extract medicine entries and header fields from *blocks*.

        Returns
        -------
        medicines : list[MedicineEntry]
            Detected medicine items with dosage/frequency/etc.
        header_fields : dict[str, ExtractedField]
            Top-level fields like prescriber, patient_name, date.
        """
        header_fields = self._extract_header_fields(blocks)
        medicines = self._extract_medicines(blocks)
        return medicines, header_fields

    # ------------------------------------------------------------------
    # Header fields (prescriber, patient, date)
    # ------------------------------------------------------------------

    _HEADER_PATTERNS: dict[str, list[str]] = {
        "prescriber": [
            "dr", "doctor", "physician", "prescriber", "md", "mbbs",
        ],
        "patient_name": [
            "patient", "name", "mr", "mrs", "ms",
        ],
        "date": [
            "date", "dated",
        ],
    }

    def _extract_header_fields(
        self, blocks: list[OCRBlock],
    ) -> dict[str, ExtractedField]:
        fields: dict[str, ExtractedField] = {}

        for i, blk in enumerate(blocks):
            text_lower = blk.text.lower().strip()

            for field_name, keywords in self._HEADER_PATTERNS.items():
                if field_name in fields:
                    continue
                for kw in keywords:
                    if kw in text_lower:
                        value = self._extract_label_value(blk.text, kw)
                        if not value and i + 1 < len(blocks):
                            next_blk = blocks[i + 1]
                            dy = abs(next_blk.centre[1] - blk.centre[1])
                            if dy < 50:
                                value = next_blk.text.strip()
                        if value:
                            fields[field_name] = ExtractedField(
                                name=field_name,
                                value=value,
                                confidence=blk.confidence,
                                source_text=blk.text,
                                source_block_idx=i,
                            )
                        break

            # Date via regex if not found by keyword
            if "date" not in fields:
                date_match = re.search(
                    r"\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}", blk.text,
                )
                if date_match:
                    fields["date"] = ExtractedField(
                        name="date",
                        value=date_match.group(),
                        confidence=blk.confidence,
                        source_text=blk.text,
                        source_block_idx=i,
                    )

        return fields

    # ------------------------------------------------------------------
    # Medicine extraction
    # ------------------------------------------------------------------

    def _extract_medicines(
        self, blocks: list[OCRBlock],
    ) -> list[MedicineEntry]:
        """Identify medicine lines and group dosage/frequency/instructions."""
        candidates: list[tuple[int, OCRBlock, str]] = []
        max_y = max((blk.y_max for blk in blocks), default=0)

        for i, blk in enumerate(blocks):
            role = self._classify_line(blk.text)
            if role == "medicine":
                if blk.y_min < 220:
                    role = "header"
                elif blk.y_max > max_y - 140:
                    role = "instruction"
            candidates.append((i, blk, role))

        medicines: list[MedicineEntry] = []
        current: MedicineEntry | None = None

        for idx, blk, role in candidates:
            if role == "medicine":
                if current and current.medicine_name:
                    medicines.append(current)
                current = MedicineEntry(
                    medicine_name=blk.text.strip(),
                    confidence=blk.confidence,
                    source_blocks=[idx],
                )
            elif current is not None:
                self._attach_detail(current, blk, role, idx)
            # else: skip header/unknown lines before first medicine

        if current and current.medicine_name:
            medicines.append(current)

        return medicines

    def _classify_line(self, text: str) -> str:
        """Classify an OCR line as medicine, dosage, frequency, etc."""
        t = text.lower().strip()

        if not t:
            return "unknown"

        if t in self._non_medicine_terms:
            return "instruction"

        if "licensed" in t or "informational" in t or "verify" in t or "finish" in t:
            return "instruction"

        # Check dosage pattern first (e.g. "500mg", "2 tablets")
        if self._dosage_re.search(t):
            return "dosage"

        # Duration (e.g. "for 5 days", "x 7 days")
        if self._has_keyword(t, self._dur_set) and re.search(r"\d", t):
            return "duration"
        if t in self._dur_set:
            return "duration"

        # Frequency
        if self._has_keyword(t, self._freq_set):
            return "frequency"

        # Route
        if self._has_keyword(t, self._route_set):
            return "route"

        # Instruction
        if self._has_keyword(t, self._instr_set):
            return "instruction"

        # Header-like lines
        for keywords in self._HEADER_PATTERNS.values():
            if self._has_keyword(t, set(keywords)):
                return "header"

        # If it looks like a drug name (mostly alpha, possibly with
        # numbers for strength like "Amoxicillin 500")
        alpha_ratio = sum(1 for c in t if c.isalpha()) / max(len(t), 1)
        word_count = len([part for part in t.split() if part])
        if alpha_ratio >= 0.5 and len(t) >= 4 and word_count <= 4:
            return "medicine"

        return "unknown"

    def _attach_detail(
        self,
        entry: MedicineEntry,
        blk: OCRBlock,
        role: str,
        idx: int,
    ) -> None:
        """Attach a detail line to an existing medicine entry."""
        text = blk.text.strip()
        entry.source_blocks.append(idx)

        if role == "dosage" and not entry.dosage:
            entry.dosage = text
        elif role == "frequency" and not entry.frequency:
            entry.frequency = text
        elif role == "duration" and not entry.duration:
            entry.duration = text
        elif role == "route" and not entry.route:
            entry.route = text
        elif role == "instruction":
            if entry.instructions:
                entry.instructions += "; " + text
            else:
                entry.instructions = text

        # Update confidence to minimum across all source blocks
        entry.confidence = min(entry.confidence, blk.confidence)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compile_dosage_patterns(patterns: list[str]) -> re.Pattern:
        combined = "|".join(f"(?:{p})" for p in patterns)
        return re.compile(combined, re.IGNORECASE)

    @staticmethod
    def _has_keyword(text: str, keywords: set[str]) -> bool:
        for kw in keywords:
            if kw in text:
                return True
        return False

    @staticmethod
    def _extract_label_value(text: str, keyword: str) -> str:
        """Extract value after a label keyword using separators."""
        for sep in [":", "-", "="]:
            if sep in text:
                parts = text.split(sep, 1)
                val = parts[1].strip()
                if val and len(val) > 1:
                    return val

        kw_lower = keyword.lower()
        t_lower = text.lower()
        pos = t_lower.find(kw_lower)
        if pos >= 0:
            after = text[pos + len(keyword) :].strip()
            # Strip common punctuation
            after = after.lstrip(".:- ")
            if after and len(after) > 1:
                return after

        return ""
