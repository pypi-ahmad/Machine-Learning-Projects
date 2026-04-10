"""
Modern Object Measurement — CVProject wrapper v2
==================================================
Wraps existing contour-based measurement logic in the unified CVProject framework.

Upgrade note: Replace hard-coded PIXELS_PER_METRIC with a calibration
workflow — e.g. ArUco marker of known size or user-placed reference object.
See Project 7 (object_size_v2) for a similar pipeline.

Original: ObjectMeasurement.py (contour + reference scale)
Modern:   Same core logic, unified interface

Usage:
    python -m core.runner --import-all object_measurement_v2 --source 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


@register("object_measurement_v2")
class ObjectMeasurementV2(CVProject):
    display_name = "Object Measurement (v2)"
    category = "opencv_utility"

    PIXELS_PER_METRIC = 30.0  # Default calibration

    def load(self):
        pass

    def predict(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edged = cv2.Canny(blurred, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        measurements = []
        for c in contours:
            if cv2.contourArea(c) < 800:
                continue
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            (cx, cy), (w, h), angle = rect
            w_real = w / self.PIXELS_PER_METRIC
            h_real = h / self.PIXELS_PER_METRIC
            measurements.append({
                "box": box, "center": (int(cx), int(cy)),
                "w": w_real, "h": h_real
            })
        return measurements

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        annotated = frame.copy()
        for m in output:
            cv2.drawContours(annotated, [m["box"]], 0, (0, 255, 0), 2)
            cv2.circle(annotated, m["center"], 4, (0, 0, 255), -1)
            label = f"{m['w']:.1f} x {m['h']:.1f}"
            cv2.putText(annotated, label,
                        (m["center"][0] - 30, m["center"][1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        cv2.putText(annotated, f"Objects: {len(output)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return annotated
