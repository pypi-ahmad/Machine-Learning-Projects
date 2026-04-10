"""
Modern Image Blurring — CVProject wrapper v2
==============================================
Wraps various blurring techniques in the unified CVProject framework.

Original: image blurring (OpenCV).ipynb
Modern:   Same core logic, unified interface with comparison view

Usage:
    python -m core.runner --import-all image_blurring_v2 --source 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


@register("image_blurring_v2")
class ImageBlurringV2(CVProject):
    display_name = "Image Blurring (v2)"
    category = "opencv_utility"

    KSIZE = 15

    def load(self):
        pass

    def predict(self, frame: np.ndarray):
        k = self.KSIZE
        gaussian = cv2.GaussianBlur(frame, (k, k), 0)
        median = cv2.medianBlur(frame, k)
        bilateral = cv2.bilateralFilter(frame, 9, 75, 75)
        box = cv2.blur(frame, (k, k))
        return {"gaussian": gaussian, "median": median,
                "bilateral": bilateral, "box": box}

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        h, w = frame.shape[:2]
        sz = (w // 2, h // 2)
        top = np.hstack([cv2.resize(output["gaussian"], sz),
                         cv2.resize(output["median"], sz)])
        bottom = np.hstack([cv2.resize(output["bilateral"], sz),
                            cv2.resize(output["box"], sz)])
        grid = np.vstack([top, bottom])
        cv2.putText(grid, "Gaussian", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(grid, "Median", (sz[0] + 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(grid, "Bilateral", (5, sz[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(grid, "Box", (sz[0] + 5, sz[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return grid
