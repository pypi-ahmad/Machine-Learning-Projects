"""
Modern Contrast Enhancement (Color) — CVProject wrapper v2
============================================================
Wraps CLAHE contrast enhancement in the unified CVProject framework.

Original: OpenCV (Contrast enhancing of color images).ipynb
Modern:   Same core logic (CLAHE on LAB), unified interface

Usage:
    python -m core.runner --import-all contrast_color_v2 --source 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


@register("contrast_color_v2")
class ContrastColorV2(CVProject):
    display_name = "Contrast Enhancement Color (v2)"
    category = "opencv_utility"

    CLIP_LIMIT = 3.0
    TILE_SIZE = (8, 8)

    def load(self):
        self.clahe = cv2.createCLAHE(clipLimit=self.CLIP_LIMIT,
                                     tileGridSize=self.TILE_SIZE)

    def predict(self, frame: np.ndarray):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        return {"enhanced": enhanced}

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        h, w = frame.shape[:2]
        half_w = w // 2
        result = output["enhanced"].copy()
        result[:, :half_w] = frame[:, :half_w]
        cv2.line(result, (half_w, 0), (half_w, h), (0, 255, 0), 2)
        cv2.putText(result, "Original", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(result, "CLAHE Enhanced", (half_w + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return result
