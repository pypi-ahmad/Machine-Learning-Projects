"""OCR engine wrapper for Number Plate Reader Pro.

Uses PaddleOCR first for license plate recognition and falls back to
EasyOCR on unsupported local runtimes.

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

    def read_plate(self, crop: np.ndarray) -> OCRResult:
        """Run OCR on a plate crop and return joined text with confidence.

        Parameters
        ----------
        crop : np.ndarray
            Greyscale or BGR plate crop (preferably rectified).
        """
        if self._ocr is None:
            self._init_ocr()

        # Ensure 3 channels for OCR backends.
        if len(crop.shape) == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

        if self._backend == "paddleocr":
            try:
                result = self._ocr.ocr(crop)
                return self._from_paddle_result(result)
            except Exception as exc:
                log.warning(
                    "PaddleOCR inference failed on this runtime (%s); switching to EasyOCR",
                    exc,
                )
                self._init_easyocr()

        result = self._ocr.readtext(crop)
        return self._from_easyocr_result(result)

    @staticmethod
    def _from_easyocr_result(result) -> OCRResult:
        if not result:
            return OCRResult(raw_text="", confidence=0.0, char_count=0)

        texts: list[str] = []
        confs: list[float] = []
        for _bbox, text, conf in result:
            texts.append(text)
            confs.append(float(conf))

        joined = " ".join(texts).strip()
        mean_conf = float(np.mean(confs)) if confs else 0.0
        return OCRResult(
            raw_text=joined,
            confidence=mean_conf,
            char_count=len(joined.replace(" ", "")),
        )

    @staticmethod
    def _from_paddle_result(result) -> OCRResult:
        if not result:
            return OCRResult(raw_text="", confidence=0.0, char_count=0)

        texts: list[str] = []
        confs: list[float] = []
        lines = result[0] if isinstance(result, list) else result
        if not lines:
            return OCRResult(raw_text="", confidence=0.0, char_count=0)

        for item in lines:
            if not item:
                continue
            if isinstance(item, dict):
                text = str(item.get("rec_text", "")).strip()
                conf = float(item.get("rec_score", 0.0))
            elif len(item) >= 2:
                rec = item[1]
                text = str(rec[0]).strip() if rec else ""
                conf = float(rec[1]) if rec and len(rec) > 1 else 0.0
            else:
                continue
            if text:
                texts.append(text)
                confs.append(conf)

        joined = " ".join(texts).strip()
        mean_conf = float(np.mean(confs)) if confs else 0.0
        return OCRResult(
            raw_text=joined,
            confidence=mean_conf,
            char_count=len(joined.replace(" ", "")),
        )

    @staticmethod
    def _normalise_paddle_lang(lang: str) -> str:
        if lang.lower().startswith("en"):
            return "en"
        return lang
