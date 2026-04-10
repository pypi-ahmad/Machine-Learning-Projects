"""
Modern Crop & Resize — CVProject wrapper v2
=============================================
Wraps image crop/resize operations in the unified CVProject framework.

Original: Crop_Resize_Images.py
Modern:   Same core logic, unified interface

Usage:
    python -m core.runner --import-all crop_resize_v2 --source image.jpg
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


@register("crop_resize_v2")
class CropResizeV2(CVProject):
    display_name = "Crop & Resize (v2)"
    category = "opencv_utility"

    SCALE_FACTOR = 0.5

    def load(self):
        pass

    def predict(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        # Center crop to 50%
        ch, cw = h // 4, w // 4
        cropped = frame[ch:h - ch, cw:w - cw]
        # Resize
        resized = cv2.resize(cropped, None, fx=self.SCALE_FACTOR,
                             fy=self.SCALE_FACTOR, interpolation=cv2.INTER_AREA)
        return {"cropped": cropped, "resized": resized,
                "orig_size": (w, h), "crop_size": cropped.shape[:2][::-1]}

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        ch, cw = h // 4, w // 4
        cv2.rectangle(annotated, (cw, ch), (w - cw, h - ch), (0, 255, 0), 2)
        cv2.putText(annotated, f"Original: {output['orig_size']}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(annotated, f"Crop: {output['crop_size']}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return annotated
