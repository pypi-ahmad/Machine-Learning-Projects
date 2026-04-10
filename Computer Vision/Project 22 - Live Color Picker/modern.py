"""
Modern Color Picker — CVProject wrapper v2
============================================
Wraps live HSV color picking in the unified CVProject framework.

Original: ColorPicker.py (HSV trackbar + webcam)
Modern:   Same core logic, unified interface

Usage:
    python -m core.runner --import-all color_picker_v2 --source 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


@register("color_picker_v2")
class ColorPickerV2(CVProject):
    display_name = "Color Picker (v2)"
    category = "opencv_utility"

    def load(self):
        self.last_color = (0, 0, 0)

    def predict(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        roi = frame[cy - 5:cy + 5, cx - 5:cx + 5]
        avg_bgr = roi.mean(axis=(0, 1)).astype(int)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        avg_hsv = hsv[cy - 5:cy + 5, cx - 5:cx + 5].mean(axis=(0, 1)).astype(int)
        return {"bgr": tuple(avg_bgr), "hsv": tuple(avg_hsv), "center": (cx, cy)}

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        annotated = frame.copy()
        cx, cy = output["center"]
        bgr = output["bgr"]
        hsv = output["hsv"]
        cv2.circle(annotated, (cx, cy), 8, (0, 255, 0), 2)
        cv2.crosshair = cv2.drawMarker(annotated, (cx, cy), (0, 255, 0),
                                        cv2.MARKER_CROSS, 20, 1)
        # Color info panel
        panel = np.zeros((80, 300, 3), dtype=np.uint8)
        panel[:] = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
        cv2.putText(panel, f"BGR: {bgr}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(panel, f"HSV: {hsv}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Place panel on frame
        ph, pw = panel.shape[:2]
        annotated[10:10 + ph, 10:10 + pw] = panel
        return annotated
