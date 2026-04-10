"""
Modern Cartoonize — CVProject wrapper v2
==========================================
Pure OpenCV artistic filter — bilateral smoothing + adaptive-threshold edges.
No DL replacement needed.

Merged: Absorbs "Project 15 - Live Image Cartoonifier" and
"Project 43 - Funny Cartoonizing Images". All three used the same
bilateral + adaptive-threshold algorithm with different parameters.
Now unified with configurable PRESET: "light", "standard", "smooth".

Original: main.py / cartoon.py / cartooning an image (OpenCV).ipynb
Modern:   Unified configurable cartoon filter

Usage:
    python -m core.runner --import-all cartoonize_image --source 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register

# Presets: (num_bilateral_passes, d, sigma_color, sigma_space, block_size, C)
_PRESETS = {
    "light":    (1, 8, 250, 250, 9, 9),   # original Cartoonize The Image
    "standard": (7, 9, 200, 200, 7, 7),   # original Project 15
    "smooth":   (7, 9, 300, 300, 9, 9),   # original Project 43
}


@register("cartoonize_image")
@register("cartoonifier_v2")
@register("cartoon_effect_v2")
class CartoonizeImageModern(CVProject):
    display_name = "Cartoonize Image"
    category = "opencv_utility"

    PRESET = "standard"  # "light" | "standard" | "smooth"
    SIDE_BY_SIDE = False  # Show original vs cartoon (from P43)

    def load(self):
        pass

    def predict(self, frame: np.ndarray):
        n, d, sc, ss, bs, c = _PRESETS.get(self.PRESET, _PRESETS["standard"])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, bs if bs % 2 == 1 else bs + 1)
        edges = cv2.adaptiveThreshold(
            gray_blur, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, bs, c,
        )
        color = frame.copy()
        for _ in range(n):
            color = cv2.bilateralFilter(color, d, sc, ss)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return {"cartoon": cartoon, "edges": edges}

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        cartoon = output["cartoon"]
        if self.SIDE_BY_SIDE:
            h, w = frame.shape[:2]
            half = w // 2
            result = np.hstack([
                cv2.resize(frame, (half, h)),
                cv2.resize(cartoon, (w - half, h)),
            ])
            cv2.putText(result, "Original", (5, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(result, f"Cartoon ({self.PRESET})", (half + 5, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            return result
        cv2.putText(cartoon, f"Cartoon ({self.PRESET})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return cartoon
