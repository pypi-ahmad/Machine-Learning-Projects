"""
Modern Thresholding Techniques — CVProject wrapper v2
=======================================================
Wraps various OpenCV thresholding techniques in the unified CVProject framework.

Original: Thresholding techniques (OpenCV).ipynb
Modern:   Same core logic, unified interface with comparison view

Usage:
    python -m core.runner --import-all thresholding_v2 --source 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


@register("thresholding_v2")
class ThresholdingV2(CVProject):
    display_name = "Thresholding Techniques (v2)"
    category = "opencv_utility"

    def load(self):
        pass

    def predict(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        _, binary_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        return {"binary": binary, "binary_inv": binary_inv,
                "otsu": otsu, "adaptive": adaptive}

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        h, w = frame.shape[:2]
        sz = (w // 2, h // 2)
        top = np.hstack([
            cv2.resize(cv2.cvtColor(output["binary"], cv2.COLOR_GRAY2BGR), sz),
            cv2.resize(cv2.cvtColor(output["binary_inv"], cv2.COLOR_GRAY2BGR), sz)
        ])
        bottom = np.hstack([
            cv2.resize(cv2.cvtColor(output["otsu"], cv2.COLOR_GRAY2BGR), sz),
            cv2.resize(cv2.cvtColor(output["adaptive"], cv2.COLOR_GRAY2BGR), sz)
        ])
        grid = np.vstack([top, bottom])
        cv2.putText(grid, "Binary", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(grid, "Binary Inv", (sz[0] + 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(grid, "Otsu", (5, sz[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(grid, "Adaptive", (sz[0] + 5, sz[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return grid
