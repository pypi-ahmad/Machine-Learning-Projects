"""
Modern Grayscale Converter — CVProject wrapper v2
===================================================
Wraps grayscale conversion in the unified CVProject framework.

Original: GrayscaleConverter.py
Modern:   Same core logic, unified interface

Usage:
    python -m core.runner --import-all grayscale_converter_v2 --source 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


@register("grayscale_converter_v2")
class GrayscaleConverterV2(CVProject):
    display_name = "Grayscale Converter (v2)"
    category = "opencv_utility"

    def load(self):
        pass

    def predict(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return {"gray": gray}

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        gray_bgr = cv2.cvtColor(output["gray"], cv2.COLOR_GRAY2BGR)
        cv2.putText(gray_bgr, "Grayscale V2", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return gray_bgr
