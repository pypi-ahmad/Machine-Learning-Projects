"""
Modern Pencil Drawing Effect — CVProject wrapper v2
=====================================================
Wraps pencil sketch effect in the unified CVProject framework.

Original: Pencil drawing effect (openCV).ipynb (cv2.pencilSketch)
Modern:   Same core logic, unified interface

Usage:
    python -m core.runner --import-all pencil_sketch_v2 --source 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


@register("pencil_sketch_v2")
class PencilSketchV2(CVProject):
    display_name = "Pencil Sketch (v2)"
    category = "opencv_utility"

    def load(self):
        pass

    def predict(self, frame: np.ndarray):
        gray_sketch, color_sketch = cv2.pencilSketch(
            frame, sigma_s=60, sigma_r=0.07, shade_factor=0.05
        )
        return {"gray_sketch": gray_sketch, "color_sketch": color_sketch}

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        h, w = frame.shape[:2]
        sz = (w // 3, h)
        gray_bgr = cv2.cvtColor(output["gray_sketch"], cv2.COLOR_GRAY2BGR)
        row = np.hstack([
            cv2.resize(frame, sz),
            cv2.resize(gray_bgr, sz),
            cv2.resize(output["color_sketch"], sz)
        ])
        cv2.putText(row, "Original", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(row, "Pencil B&W", (sz[0] + 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(row, "Pencil Color", (sz[0] * 2 + 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return row
