"""
Modern Shape Detection — CVProject wrapper v2
===============================================
Wraps contour-based shape classification in the unified CVProject framework.

Original: RealTime_Shape_Detection_Contours.py (contour + polygon approx)
Modern:   Same core logic, unified interface

Usage:
    python -m core.runner --import-all shape_detection_v2 --source 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


def _classify_shape(approx):
    n = len(approx)
    if n == 3:
        return "Triangle"
    elif n == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h) if h > 0 else 0
        return "Square" if 0.85 <= ar <= 1.15 else "Rectangle"
    elif n == 5:
        return "Pentagon"
    elif n == 6:
        return "Hexagon"
    elif n > 6:
        return "Circle"
    return "Unknown"


@register("shape_detection_v2")
class ShapeDetectionV2(CVProject):
    display_name = "Shape Detection (v2)"
    category = "opencv_utility"

    def load(self):
        pass

    def predict(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 4)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shapes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 500:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            shape_name = _classify_shape(approx)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0
            shapes.append({"contour": c, "name": shape_name, "center": (cx, cy)})
        return shapes

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        annotated = frame.copy()
        for s in output:
            cv2.drawContours(annotated, [s["contour"]], -1, (0, 255, 0), 2)
            cv2.putText(annotated, s["name"], (s["center"][0] - 20, s["center"][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(annotated, f"Shapes: {len(output)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return annotated
