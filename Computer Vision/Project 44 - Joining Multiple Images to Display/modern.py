"""
Modern Image Joining — CVProject wrapper v2
=============================================
Wraps image joining/concatenation logic in the unified CVProject framework.

Original: Joining_Multiple_Images_To_Display.py
Modern:   Same core logic, unified interface

Usage:
    python -m core.runner --import-all image_joining_v2 --source 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


@register("image_joining_v2")
class ImageJoiningV2(CVProject):
    display_name = "Image Joining (v2)"
    category = "opencv_utility"

    def load(self):
        pass

    def predict(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        edged = cv2.Canny(gray, 50, 150)
        edged_bgr = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)
        blurred = cv2.GaussianBlur(frame, (15, 15), 0)
        return {"original": frame, "gray": gray_bgr,
                "edges": edged_bgr, "blurred": blurred}

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        h, w = frame.shape[:2]
        # Create 2x2 grid
        top = np.hstack([
            cv2.resize(output["original"], (w // 2, h // 2)),
            cv2.resize(output["gray"], (w // 2, h // 2))
        ])
        bottom = np.hstack([
            cv2.resize(output["edges"], (w // 2, h // 2)),
            cv2.resize(output["blurred"], (w // 2, h // 2))
        ])
        grid = np.vstack([top, bottom])
        cv2.putText(grid, "Original", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(grid, "Gray", (w // 2 + 5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(grid, "Edges", (5, h // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(grid, "Blurred", (w // 2 + 5, h // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return grid
