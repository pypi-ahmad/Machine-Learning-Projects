"""
Modern Virtual Pen — CVProject wrapper v2
===========================================
Wraps virtual pen drawing logic in the unified CVProject framework.

Upgrade note: Replace HSV color tracking with MediaPipe Hand Landmarker
(index-finger tip = landmark 8) for marker-free air drawing. See
Project 6 (hand_tracking_v2) for the MediaPipe hand pipeline.

Original: VPen.ipynb (color tracking + drawing)
Modern:   Same core logic, unified interface

Usage:
    python -m core.runner --import-all virtual_pen_v2 --source 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from collections import deque

from core.base import CVProject
from core.registry import register


@register("virtual_pen_v2")
class VirtualPenV2(CVProject):
    display_name = "Virtual Pen (v2)"
    category = "opencv_utility"

    # Track green objects by default
    LOWER_HSV = np.array([35, 100, 100])
    UPPER_HSV = np.array([85, 255, 255])

    def load(self):
        self.points = deque(maxlen=2048)

    def predict(self, frame: np.ndarray):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.LOWER_HSV, self.UPPER_HSV)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None
        if contours:
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            if radius > 10:
                center = (int(x), int(y))
                self.points.append(center)
        return {"center": center, "points": list(self.points)}

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        annotated = frame.copy()
        points = output["points"]
        for i in range(1, len(points)):
            if points[i - 1] is None or points[i] is None:
                continue
            cv2.line(annotated, points[i - 1], points[i], (0, 0, 255), 3)
        if output["center"]:
            cv2.circle(annotated, output["center"], 8, (0, 255, 0), -1)
        cv2.putText(annotated, "Virtual Pen V2 (green marker)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        return annotated
