"""
Modern Noise Removal — CVProject wrapper v2
=============================================
Wraps OpenCV denoising in the unified CVProject framework.

Original: Noise Removing (OpenCV).ipynb (fastNlMeansDenoisingColored)
Modern:   Same core logic, unified interface

Note: P39 is labeled "Pencil drawing effect" in the folder name but
the actual notebook is "Noise Removing (OpenCV).ipynb".

Usage:
    python -m core.runner --import-all noise_removal_v2 --source 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


@register("noise_removal_v2")
class NoiseRemovalV2(CVProject):
    display_name = "Noise Removal (v2)"
    category = "opencv_utility"

    def load(self):
        pass

    def predict(self, frame: np.ndarray):
        denoised_fast = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        median = cv2.medianBlur(frame, 5)
        return {"denoised": denoised_fast, "gaussian": blurred, "median": median}

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        h, w = frame.shape[:2]
        sz = (w // 2, h // 2)
        top = np.hstack([cv2.resize(frame, sz), cv2.resize(output["denoised"], sz)])
        bottom = np.hstack([cv2.resize(output["gaussian"], sz), cv2.resize(output["median"], sz)])
        grid = np.vstack([top, bottom])
        cv2.putText(grid, "Original", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(grid, "NLMeans", (sz[0] + 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(grid, "Gaussian", (5, sz[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(grid, "Median", (sz[0] + 5, sz[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return grid
