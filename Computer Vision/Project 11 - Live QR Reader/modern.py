"""
Modern QR Reader — CVProject wrapper v2
=========================================
Wraps existing QR code reading in the unified CVProject framework.

Original: QR_Reader.py (pyzbar/OpenCV QR detection)
Modern:   OpenCV QRCodeDetector, unified interface

Usage:
    python -m core.runner --import-all qr_reader_v2 --source 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


@register("qr_reader_v2")
class QRReaderV2(CVProject):
    display_name = "QR Reader (v2)"
    category = "opencv_utility"

    def load(self):
        self.detector = cv2.QRCodeDetector()

    def predict(self, frame: np.ndarray):
        data, bbox, _ = self.detector.detectAndDecode(frame)
        return {"data": data, "bbox": bbox}

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        annotated = frame.copy()
        bbox = output["bbox"]
        data = output["data"]
        if bbox is not None and data:
            pts = bbox[0].astype(int)
            for i in range(len(pts)):
                cv2.line(annotated, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]),
                         (0, 255, 0), 3)
            cv2.putText(annotated, f"QR: {data}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(annotated, "Scanning for QR code...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        return annotated
