"""
Modern Text Detection + Recognition — PaddleOCR v2
====================================================
Replaces legacy EAST text detector with PaddleOCR (two-stage: detection + recognition).

Original: main.py (EAST frozen model — missing)
Modern:   PaddleOCR handles text detection (DBNet) + text recognition (CRNN) as a
          proper two-stage pipeline. Falls back to EasyOCR if unavailable.

Registry: ocr_detect + ocr_recognize → PaddleOCR sentinels

Usage:
    python -m core.runner --import-all text_detection_v2 --source 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


@register("text_detection_v2")
class TextDetectionV2(CVProject):
    display_name = "Text Detection + Recognition (PaddleOCR)"
    category = "detection"

    CONF_THRESHOLD = 0.3
    _ocr = None
    _backend = None  # "paddle" or "easyocr"

    def load(self):
        # Priority 1: PaddleOCR (best two-stage pipeline)
        try:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=True, show_log=False)
            self._backend = "paddle"
            print("  [text_detection] Using PaddleOCR (DBNet detection + CRNN recognition)")
            return
        except ImportError:
            pass

        # Priority 2: EasyOCR fallback
        try:
            import easyocr
            self._ocr = easyocr.Reader(["en"], gpu=True)
            self._backend = "easyocr"
            print("  [text_detection] PaddleOCR not installed — using EasyOCR fallback")
            return
        except ImportError:
            pass

        print("  [text_detection] No OCR engine available")
        print("  [text_detection] Install: pip install paddleocr  (or)  pip install easyocr")

    def predict(self, frame: np.ndarray):
        if self._ocr is None:
            return {"texts": []}

        if self._backend == "paddle":
            result = self._ocr.ocr(frame, cls=True)
            texts = []
            if result and result[0]:
                for line in result[0]:
                    pts = np.array(line[0], dtype=np.int32)
                    x1, y1 = pts.min(axis=0)
                    x2, y2 = pts.max(axis=0)
                    text, conf = line[1]
                    texts.append({
                        "box": (int(x1), int(y1), int(x2), int(y2)),
                        "polygon": pts.tolist(),
                        "text": text,
                        "conf": float(conf),
                    })
            return {"texts": texts}

        # easyocr
        ocr_results = self._ocr.readtext(frame)
        texts = []
        for bbox, text, conf in ocr_results:
            pts = np.array(bbox, dtype=np.int32)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            texts.append({
                "box": (int(x1), int(y1), int(x2), int(y2)),
                "polygon": pts.tolist(),
                "text": text,
                "conf": float(conf),
            })
        return {"texts": texts}

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        if not isinstance(output, dict):
            return frame
        vis = frame.copy()
        for t in output.get("texts", []):
            x1, y1, x2, y2 = t["box"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{t['text']} ({t['conf']:.2f})"
            cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        backend_label = "PaddleOCR" if self._backend == "paddle" else "EasyOCR"
        cv2.putText(vis, f"{backend_label} | Texts: {len(output['texts'])}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        return vis
