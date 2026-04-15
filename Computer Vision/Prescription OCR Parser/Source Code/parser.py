"""High-level parser for Prescription OCR Parser.

Orchestrates OCR -> field extraction -> structured result.

**DISCLAIMER:** This tool is for informational and educational
purposes only.  It does not provide medical advice.

Usage::

    from parser import PrescriptionParser

    parser = PrescriptionParser(cfg)
    result = parser.parse(image)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PrescriptionConfig
from field_extractor import ExtractedField, FieldExtractor, MedicineEntry
from ocr_engine import OCRBlock, OCREngine


@dataclass
class PrescriptionResult:
    """Complete prescription extraction result."""

    medicines: list[MedicineEntry] = field(default_factory=list)
    header_fields: dict[str, ExtractedField] = field(default_factory=dict)
    raw_text: str = ""
    num_blocks: int = 0
    num_medicines: int = 0
    mean_confidence: float = 0.0


class PrescriptionParser:
    """Orchestrate OCR + field extraction for prescriptions."""

    def __init__(self, cfg: PrescriptionConfig) -> None:
        self.cfg = cfg
        self._engine = OCREngine(cfg)
        self._extractor = FieldExtractor(cfg)

    def parse(self, image: np.ndarray) -> PrescriptionResult:
        """Full pipeline: OCR -> extract -> structure."""
        blocks = self._engine.run(image)
        return self._build_result(blocks)

    def parse_with_blocks(
        self, image: np.ndarray,
    ) -> tuple[PrescriptionResult, list[OCRBlock]]:
        """Parse and also return raw OCR blocks for overlay."""
        blocks = self._engine.run(image)
        result = self._build_result(blocks)
        return result, blocks

    def _build_result(self, blocks: list[OCRBlock]) -> PrescriptionResult:
        if not blocks:
            return PrescriptionResult()

        medicines, header_fields = self._extractor.extract(blocks)
        raw_text = self._engine.full_text(blocks)

        # Mean confidence across all blocks
        confs = [b.confidence for b in blocks]
        mean_conf = float(np.mean(confs)) if confs else 0.0

        return PrescriptionResult(
            medicines=medicines,
            header_fields=header_fields,
            raw_text=raw_text,
            num_blocks=len(blocks),
            num_medicines=len(medicines),
            mean_confidence=mean_conf,
        )
