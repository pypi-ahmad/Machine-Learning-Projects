"""
Modern Image Sharpening — CVProject wrapper v2
================================================
Wraps kernel-based sharpening in the unified CVProject framework.

Original: sharpening of images using opencv.ipynb
Modern:   Same core logic (Laplacian / Unsharp mask), unified interface

Usage:
    python -m core.runner --import-all image_sharpening_v2 --source 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


@register("image_sharpening_v2")
class ImageSharpeningV2(CVProject):
    display_name = "Image Sharpening (v2)"
    category = "opencv_utility"

    def load(self):
        self.kernel_sharpen = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        self.kernel_edge = np.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]
        ])

    def predict(self, frame: np.ndarray):
        sharp1 = cv2.filter2D(frame, -1, self.kernel_sharpen)
        sharp2 = cv2.filter2D(frame, -1, self.kernel_edge)
        # Unsharp mask
        blurred = cv2.GaussianBlur(frame, (9, 9), 10)
        unsharp = cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)
        return {"laplacian": sharp1, "edge_enhance": sharp2, "unsharp": unsharp}

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        h, w = frame.shape[:2]
        sz = (w // 2, h // 2)
        top = np.hstack([cv2.resize(frame, sz), cv2.resize(output["laplacian"], sz)])
        bottom = np.hstack([cv2.resize(output["edge_enhance"], sz), cv2.resize(output["unsharp"], sz)])
        grid = np.vstack([top, bottom])
        cv2.putText(grid, "Original", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(grid, "Laplacian Sharp", (sz[0] + 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(grid, "Edge Enhance", (5, sz[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(grid, "Unsharp Mask", (sz[0] + 5, sz[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return grid
