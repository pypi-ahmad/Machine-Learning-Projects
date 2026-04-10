"""
Modern Contrast Enhancement (Grayscale) — CVProject wrapper v2
================================================================
Wraps grayscale CLAHE/histogram equalization in the unified CVProject framework.

Original: Enhancing contrast of gray scale image (OpenCV).ipynb
Modern:   Same core logic, unified interface

Usage:
    python -m core.runner --import-all contrast_gray_v2 --source 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


@register("contrast_gray_v2")
class ContrastGrayV2(CVProject):
    display_name = "Contrast Enhancement Gray (v2)"
    category = "opencv_utility"

    def load(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def predict(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eq_hist = cv2.equalizeHist(gray)
        eq_clahe = self.clahe.apply(gray)
        return {"gray": gray, "hist_eq": eq_hist, "clahe": eq_clahe}

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        g = cv2.cvtColor(output["gray"], cv2.COLOR_GRAY2BGR)
        h = cv2.cvtColor(output["hist_eq"], cv2.COLOR_GRAY2BGR)
        c = cv2.cvtColor(output["clahe"], cv2.COLOR_GRAY2BGR)
        ht, w = g.shape[:2]
        sz = (w // 3, ht)
        row = np.hstack([cv2.resize(g, sz), cv2.resize(h, sz), cv2.resize(c, sz)])
        cv2.putText(row, "Original", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(row, "Hist EQ", (sz[0] + 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(row, "CLAHE", (sz[0] * 2 + 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return row
