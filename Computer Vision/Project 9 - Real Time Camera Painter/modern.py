"""
Modern Real-Time Painter — CVProject wrapper v2
=================================================
Wraps existing color-tracking paint logic in the unified CVProject framework.

Upgrade note: Replace HSV color tracking with MediaPipe Hand Landmarker
(index-finger tip = landmark 8) for marker-free air drawing. See Project 6
(hand_tracking_v2) for the MediaPipe hand pipeline.

Original: Paint.py (HSV color tracking + drawing on frame)
Modern:   Same core logic, unified interface

Usage:
    python -m core.runner --import-all painter_v2 --source 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from collections import deque

from core.base import CVProject
from core.registry import register


@register("painter_v2")
class PainterV2(CVProject):
    display_name = "Real-Time Painter (v2)"
    category = "opencv_utility"

    # Default HSV range for blue marker
    LOWER_COLOR = np.array([100, 60, 60])
    UPPER_COLOR = np.array([140, 255, 255])

    def load(self):
        self.canvas_points = deque(maxlen=1024)

    def predict(self, frame: np.ndarray):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.LOWER_COLOR, self.UPPER_COLOR)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None
        if contours:
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            if radius > 10:
                center = (int(x), int(y))
                self.canvas_points.append(center)
        return {"center": center, "points": list(self.canvas_points)}

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        annotated = cv2.flip(frame, 1)
        h, w = annotated.shape[:2]
        points = output["points"]
        for i in range(1, len(points)):
            if points[i - 1] is None or points[i] is None:
                continue
            p1 = (w - points[i - 1][0], points[i - 1][1])
            p2 = (w - points[i][0], points[i][1])
            cv2.line(annotated, p1, p2, (0, 0, 255), 3)
        if output["center"]:
            cx = w - output["center"][0]
            cv2.circle(annotated, (cx, output["center"][1]), 8, (0, 255, 0), -1)
        cv2.putText(annotated, "Paint Mode (blue marker)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        return annotated
