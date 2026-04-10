"""
Modern Image Resizing — CVProject wrapper v2
==============================================
Wraps various interpolation methods in the unified CVProject framework.

Original: image resizing (OpenCV).ipynb
Modern:   Same core logic, unified interface with comparison view

Usage:
    python -m core.runner --import-all image_resizing_v2 --source image.jpg
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


@register("image_resizing_v2")
class ImageResizingV2(CVProject):
    display_name = "Image Resizing (v2)"
    category = "opencv_utility"

    def load(self):
        pass

    def predict(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        small = (w // 4, h // 4)
        target = (w, h)
        # Downscale then upscale to show interpolation differences
        methods = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
        }
        results = {}
        for name, method in methods.items():
            down = cv2.resize(frame, small, interpolation=cv2.INTER_AREA)
            up = cv2.resize(down, target, interpolation=method)
            results[name] = up
        return results

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        h, w = frame.shape[:2]
        sz = (w // 2, h // 2)
        top = np.hstack([cv2.resize(output["nearest"], sz),
                         cv2.resize(output["linear"], sz)])
        bottom = np.hstack([cv2.resize(output["cubic"], sz),
                            cv2.resize(output["lanczos"], sz)])
        grid = np.vstack([top, bottom])
        cv2.putText(grid, "Nearest", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(grid, "Linear", (sz[0] + 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(grid, "Cubic", (5, sz[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(grid, "Lanczos4", (sz[0] + 5, sz[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return grid
