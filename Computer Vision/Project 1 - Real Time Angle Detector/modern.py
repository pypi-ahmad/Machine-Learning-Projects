"""
Modern Angle Detector — CVProject wrapper v2
==============================================
Wraps existing OpenCV angle detection logic in the unified CVProject framework.
No DL replacement needed — pure geometry/math.

Original: AngleDetector.py (OpenCV contour + angle math)
Modern:   Same core logic, unified interface

Usage:
    python -m core.runner --import-all angle_detector_v2 --source 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import math

from core.base import CVProject
from core.registry import register


@register("angle_detector_v2")
class AngleDetectorV2(CVProject):
    display_name = "Angle Detector (v2)"
    category = "opencv_utility"

    def load(self):
        self.points = []

    def predict(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80,
                                minLineLength=50, maxLineGap=10)
        return {"edges": edges, "lines": lines}

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        annotated = frame.copy()
        lines = output.get("lines")
        if lines is not None and len(lines) >= 2:
            for line in lines[:10]:
                x1, y1, x2, y2 = line[0]
                cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Measure angle between first two lines
            l1 = lines[0][0]
            l2 = lines[1][0]
            a1 = math.atan2(l1[3] - l1[1], l1[2] - l1[0])
            a2 = math.atan2(l2[3] - l2[1], l2[2] - l2[0])
            angle = abs(math.degrees(a1 - a2))
            if angle > 180:
                angle = 360 - angle
            cv2.putText(annotated, f"Angle: {angle:.1f} deg", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            cv2.putText(annotated, "Detecting lines...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        return annotated
