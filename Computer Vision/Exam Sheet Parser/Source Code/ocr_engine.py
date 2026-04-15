"""OCR engine wrapper for Exam Sheet Parser.

Uses PaddleOCR first when available and falls back to EasyOCR on
unsupported local runtimes. Returns structured
``OCRBlock`` dataclasses with bounding-box polygons, text,
and confidence scores.

Usage::

    from ocr_engine import OCREngine
    from config import ExamSheetConfig

    engine = OCREngine(ExamSheetConfig())
    blocks = engine.run(image)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("exam_sheet.ocr_engine")


@dataclass
class OCRBlock:
    """Single OCR text detection from a document image."""

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
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min


class OCREngine:
    """PaddleOCR-first wrapper with lazy initialisation for document images."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._ocr = None
        self._backend = ""
        self._preferred_backend = getattr(cfg, "ocr_backend", "auto").lower()

    def _init_ocr(self) -> None:
        if self._preferred_backend in {"auto", "paddleocr", "paddle"}:
            try:
                from paddleocr import PaddleOCR

                lang = self._normalise_paddle_lang(
                    getattr(self.cfg, "ocr_lang", "en"),
                )
                self._ocr = PaddleOCR(
                    lang=lang,
                    use_angle_cls=getattr(self.cfg, "use_angle_cls", True),
                )
                self._backend = "paddleocr"
                log.info("PaddleOCR initialised (lang=%s)", lang)
                return
            except Exception as exc:
                if self._preferred_backend not in {"auto"}:
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
            gpu=getattr(self.cfg, "use_gpu", False),
        )
        self._backend = "easyocr"
        log.info("EasyOCR initialised (lang=%s)", lang)

    def _normalise_paddle_lang(self, lang: str) -> str:
        if lang.startswith("en"):
            return "en"
        return lang

    def _run_paddleocr(self, image: np.ndarray) -> list[OCRBlock]:
        result = self._ocr.ocr(
            image,
            cls=getattr(self.cfg, "use_angle_cls", True),
        )
        blocks: list[OCRBlock] = []
        if not result:
            return blocks

        for page_result in result:
            if not page_result:
                continue
            for line in page_result:
                if len(line) < 2:
                    continue
                bbox_raw, rec = line[0], line[1]
                if not rec or len(rec) < 2:
                    continue
                text, conf = rec[0], rec[1]
                if float(conf) < self.cfg.min_block_confidence:
                    continue
                bbox = [[int(point[0]), int(point[1])] for point in bbox_raw]
                cx = int(np.mean([point[0] for point in bbox]))
                cy = int(np.mean([point[1] for point in bbox]))
                blocks.append(
                    OCRBlock(
                        text=str(text).strip(),
                        confidence=float(conf),
                        bbox=bbox,
                        centre=(cx, cy),
                    )
                )
        return blocks

    def _run_easyocr(self, image: np.ndarray) -> list[OCRBlock]:
        result = self._ocr.readtext(image)
        blocks: list[OCRBlock] = []
        if not result:
            return blocks

        for bbox_raw, text, conf in result:
            if float(conf) < self.cfg.min_block_confidence:
                continue
            bbox = [[int(point[0]), int(point[1])] for point in bbox_raw]
            cx = int(np.mean([point[0] for point in bbox]))
            cy = int(np.mean([point[1] for point in bbox]))
            blocks.append(
                OCRBlock(
                    text=text.strip(),
                    confidence=float(conf),
                    bbox=bbox,
                    centre=(cx, cy),
                )
            )
        return blocks

    def run(self, image: np.ndarray) -> list[OCRBlock]:
        """Run OCR on a BGR image and return structured blocks.

        Parameters
        ----------
        image : np.ndarray
            BGR document image.

        Returns
        -------
        list[OCRBlock]
            Detected text blocks sorted top-to-bottom, left-to-right.
        """
        if self._ocr is None:
            self._init_ocr()

        try:
            if self._backend == "paddleocr":
                blocks = self._run_paddleocr(image)
            else:
                blocks = self._run_easyocr(image)
        except Exception as exc:
            if self._backend != "paddleocr":
                raise
            log.warning(
                "PaddleOCR inference failed (%s); retrying with EasyOCR fallback",
                exc,
            )
            self._init_easyocr()
            blocks = self._run_easyocr(image)

        blocks.sort(key=lambda b: (b.centre[1], b.centre[0]))
        return blocks

    def full_text(self, blocks: list[OCRBlock]) -> str:
        """Concatenate all blocks into a single text string."""
        return "\n".join(b.text for b in blocks)
