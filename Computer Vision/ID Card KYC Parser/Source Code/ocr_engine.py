"""EasyOCR engine wrapper for ID Card KYC Parser.

Thin abstraction over EasyOCR that returns structured
``OCRBlock`` dataclasses, hiding the raw tuple format.

Usage::

    from ocr_engine import OCREngine
    from config import IDCardConfig

    engine = OCREngine(IDCardConfig())
    blocks = engine.run(image)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("id_card_kyc.ocr_engine")


@dataclass
class OCRBlock:
    """Single OCR text detection."""

    text: str
    confidence: float
    bbox: list[list[int]]       # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    centre: tuple[int, int]


class OCREngine:
    """EasyOCR wrapper with lazy initialisation."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._ocr = None

    def _init_ocr(self) -> None:
        import easyocr

        lang = self.cfg.ocr_lang if hasattr(self.cfg, "ocr_lang") else "en"
        self._ocr = easyocr.Reader(
            [lang],
            gpu=getattr(self.cfg, "use_gpu", True),
        )
        log.info("EasyOCR initialised (lang=%s)", lang)

    def run(self, image: np.ndarray) -> list[OCRBlock]:
        """Run OCR on a BGR image and return structured blocks."""
        if self._ocr is None:
            self._init_ocr()

        result = self._ocr.readtext(image)
        blocks: list[OCRBlock] = []

        if not result:
            return blocks

        for bbox_raw, text, conf in result:
                bbox = [[int(p[0]), int(p[1])] for p in bbox_raw]
                cx = int(np.mean([p[0] for p in bbox]))
                cy = int(np.mean([p[1] for p in bbox]))
                blocks.append(
                    OCRBlock(
                        text=text.strip(),
                        confidence=float(conf),
                        bbox=bbox,
                        centre=(cx, cy),
                    )
                )

        blocks.sort(key=lambda b: (b.centre[1], b.centre[0]))
        return blocks

    def full_text(self, blocks: list[OCRBlock]) -> str:
        """Concatenate all blocks into a single text string."""
        return "\n".join(b.text for b in blocks)
