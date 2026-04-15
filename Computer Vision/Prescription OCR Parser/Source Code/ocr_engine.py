"""OCR engine wrapper for Prescription OCR Parser.

Uses PaddleOCR first when available and falls back to EasyOCR on
unsupported local runtimes. Returns structured ``OCRBlock``
dataclasses, hiding backend-specific tuple formats.

Usage::

    from ocr_engine import OCREngine
    from config import PrescriptionConfig

    engine = OCREngine(PrescriptionConfig())
    blocks = engine.run(image)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("prescription_ocr.ocr_engine")


@dataclass
class OCRBlock:
    """Single OCR text detection."""

    text: str
    confidence: float
    bbox: list[list[int]]       # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    centre: tuple[int, int]

    @property
    def x_min(self) -> int:
        return min(p[0] for p in self.bbox)

    @property
    def y_min(self) -> int:
        return min(p[1] for p in self.bbox)

    @property
    def x_max(self) -> int:
        return max(p[0] for p in self.bbox)

    @property
    def y_max(self) -> int:
        return max(p[1] for p in self.bbox)

    @property
    def height(self) -> int:
        return self.y_max - self.y_min


class OCREngine:
    """PaddleOCR-first wrapper with lazy initialisation."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._ocr = None
        self._backend = ""

    def _init_ocr(self) -> None:
        preferred = getattr(self.cfg, "ocr_backend", "auto").lower()

        if preferred in {"auto", "paddleocr", "paddle"}:
            try:
                from paddleocr import PaddleOCR

                lang = self._normalise_paddle_lang(
                    getattr(self.cfg, "ocr_lang", "en"),
                )
                self._ocr = PaddleOCR(
                    lang=lang,
                    text_det_thresh=getattr(self.cfg, "det_db_thresh", 0.3),
                    text_recognition_batch_size=getattr(self.cfg, "rec_batch_num", 6),
                )
                self._backend = "paddleocr"
                log.info("PaddleOCR initialised (lang=%s)", lang)
                return
            except Exception as exc:
                if preferred not in {"auto"}:
                    raise
                log.warning(
                    "PaddleOCR unavailable on this runtime (%s); falling back to EasyOCR",
                    exc,
                )

        self._init_easyocr()

    def _init_easyocr(self) -> None:
        import easyocr

        lang = self.cfg.ocr_lang if hasattr(self.cfg, "ocr_lang") else "en"
        self._ocr = easyocr.Reader(
            [lang],
            gpu=getattr(self.cfg, "use_gpu", True),
        )
        self._backend = "easyocr"
        log.info("EasyOCR initialised (lang=%s)", lang)

    def run(self, image: np.ndarray) -> list[OCRBlock]:
        """Run OCR on a BGR image and return structured blocks."""
        if self._ocr is None:
            self._init_ocr()

        if self._backend == "paddleocr":
            try:
                result = self._ocr.ocr(image)
                return self._from_paddle_result(result)
            except Exception as exc:
                log.warning(
                    "PaddleOCR inference failed on this runtime (%s); switching to EasyOCR",
                    exc,
                )
                self._init_easyocr()

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

    def _from_paddle_result(self, result) -> list[OCRBlock]:
        blocks: list[OCRBlock] = []
        if not result:
            return blocks

        lines = result[0] if isinstance(result, list) else result
        if not lines:
            return blocks

        for item in lines:
            if not item or len(item) < 2:
                continue
            bbox_raw, rec = item
            text = rec[0] if rec else ""
            conf = rec[1] if rec and len(rec) > 1 else 0.0
            bbox = [[int(p[0]), int(p[1])] for p in bbox_raw]
            cx = int(np.mean([p[0] for p in bbox]))
            cy = int(np.mean([p[1] for p in bbox]))
            blocks.append(
                OCRBlock(
                    text=str(text).strip(),
                    confidence=float(conf),
                    bbox=bbox,
                    centre=(cx, cy),
                )
            )

        blocks.sort(key=lambda b: (b.centre[1], b.centre[0]))
        return blocks

    @staticmethod
    def _normalise_paddle_lang(lang: str) -> str:
        if lang.lower().startswith("en"):
            return "en"
        return lang

    def full_text(self, blocks: list[OCRBlock]) -> str:
        """Concatenate all blocks into a single text string."""
        return "\n".join(b.text for b in blocks)
