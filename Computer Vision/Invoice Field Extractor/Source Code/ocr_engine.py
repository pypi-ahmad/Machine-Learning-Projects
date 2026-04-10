"""PaddleOCR engine wrapper for Invoice Field Extractor.

Provides a thin abstraction over PaddleOCR that returns structured
``OCRBlock`` dataclasses, hiding the raw PaddleOCR tuple format.

Usage::

    from ocr_engine import OCREngine
    from config import InvoiceConfig

    engine = OCREngine(InvoiceConfig())
    blocks = engine.run(image)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("invoice_extractor.ocr_engine")


@dataclass
class OCRBlock:
    """Single OCR text detection."""

    text: str
    confidence: float
    bbox: list[list[int]]   # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] — quad
    centre: tuple[int, int]


class OCREngine:
    """PaddleOCR wrapper with lazy initialisation."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._ocr = None

    def _init_ocr(self) -> None:
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(
            use_angle_cls=True,
            lang=self.cfg.ocr_lang,
            use_gpu=self.cfg.use_gpu,
            det_db_thresh=self.cfg.det_db_thresh,
            rec_batch_num=self.cfg.rec_batch_num,
            show_log=False,
        )
        log.info("PaddleOCR initialised (lang=%s, gpu=%s)",
                 self.cfg.ocr_lang, self.cfg.use_gpu)

    def run(self, image: np.ndarray) -> list[OCRBlock]:
        """Run OCR on a BGR image and return structured blocks.

        Parameters
        ----------
        image : np.ndarray
            BGR image (OpenCV convention).

        Returns
        -------
        list[OCRBlock]
            Detected text blocks sorted top-to-bottom, left-to-right.
        """
        if self._ocr is None:
            self._init_ocr()

        result = self._ocr.ocr(image, cls=True)
        blocks: list[OCRBlock] = []

        if result is None:
            return blocks

        for page in result:
            if page is None:
                continue
            for line in page:
                bbox_raw, (text, conf) = line
                bbox = [[int(p[0]), int(p[1])] for p in bbox_raw]
                cx = int(np.mean([p[0] for p in bbox]))
                cy = int(np.mean([p[1] for p in bbox]))
                blocks.append(OCRBlock(
                    text=text.strip(),
                    confidence=float(conf),
                    bbox=bbox,
                    centre=(cx, cy),
                ))

        # Sort reading order: top-to-bottom, left-to-right
        blocks.sort(key=lambda b: (b.centre[1], b.centre[0]))
        return blocks

    def full_text(self, blocks: list[OCRBlock]) -> str:
        """Concatenate all blocks into a single text string."""
        return "\n".join(b.text for b in blocks)
