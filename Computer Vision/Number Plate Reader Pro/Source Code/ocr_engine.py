"""PaddleOCR engine wrapper for Number Plate Reader Pro.

Thin abstraction over PaddleOCR optimised for license plate text
recognition with lazy initialisation.

Usage::

    from ocr_engine import OCREngine
    from config import PlateConfig

    engine = OCREngine(PlateConfig())
    text, confidence = engine.read_plate(crop)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger("plate_reader.ocr_engine")


@dataclass
class OCRResult:
    """OCR result for a single plate crop."""

    raw_text: str
    confidence: float
    char_count: int


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
        log.info(
            "PaddleOCR initialised (lang=%s, gpu=%s)",
            self.cfg.ocr_lang,
            self.cfg.use_gpu,
        )

    def read_plate(self, crop: np.ndarray) -> OCRResult:
        """Run OCR on a plate crop and return joined text with confidence.

        Parameters
        ----------
        crop : np.ndarray
            Greyscale or BGR plate crop (preferably rectified).
        """
        if self._ocr is None:
            self._init_ocr()

        # Ensure 3 channels for PaddleOCR
        if len(crop.shape) == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

        result = self._ocr.ocr(crop, cls=True)

        if result is None or not result[0]:
            return OCRResult(raw_text="", confidence=0.0, char_count=0)

        texts: list[str] = []
        confs: list[float] = []

        for page in result:
            if page is None:
                continue
            for item in page:
                texts.append(item[1][0])
                confs.append(float(item[1][1]))

        joined = " ".join(texts)
        mean_conf = float(np.mean(confs)) if confs else 0.0

        return OCRResult(
            raw_text=joined,
            confidence=mean_conf,
            char_count=len(joined.replace(" ", "")),
        )
