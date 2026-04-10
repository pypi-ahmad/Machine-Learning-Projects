"""
Modern Watermarking — CVProject wrapper v2
============================================
Wraps watermark overlay logic in the unified CVProject framework.

Original: watermarking on images using OpenCV.ipynb
Modern:   Same core logic (addWeighted), unified interface

Usage:
    python -m core.runner --import-all watermarking_v2 --source 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


@register("watermarking_v2")
class WatermarkingV2(CVProject):
    display_name = "Watermarking (v2)"
    category = "opencv_utility"

    WATERMARK_TEXT = "CV Projects v2"
    ALPHA = 0.7

    def load(self):
        pass

    def predict(self, frame: np.ndarray):
        overlay = frame.copy()
        h, w = overlay.shape[:2]
        # Create text watermark
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(self.WATERMARK_TEXT, font, 1.5, 3)[0]
        tx = w - text_size[0] - 20
        ty = h - 20
        cv2.putText(overlay, self.WATERMARK_TEXT, (tx, ty), font, 1.5,
                    (255, 255, 255), 3, cv2.LINE_AA)
        watermarked = cv2.addWeighted(overlay, self.ALPHA, frame, 1 - self.ALPHA, 0)
        return {"watermarked": watermarked}

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        return output["watermarked"]
