"""
Modern Object Size Detector — CVProject wrapper v2
====================================================
Wraps existing OpenCV contour-based object size measurement.

Upgrade note: Replace the hard-coded PIXELS_PER_CM with ArUco marker
calibration — detect a known-size ArUco tag in the scene and derive
pixels-per-metric dynamically. See cv2.aruco for detector API.

Original: object_size.py (contour + reference object)
Modern:   Same core logic, unified interface

Usage:
    python -m core.runner --import-all object_size_v2 --source 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


@register("object_size_v2")
class ObjectSizeV2(CVProject):
    display_name = "Object Size Detector (v2)"
    category = "opencv_utility"

    PIXELS_PER_CM = 37.8  # Default calibration; override as needed

    def load(self):
        pass

    def predict(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edged = cv2.Canny(blurred, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objects = []
        for c in contours:
            if cv2.contourArea(c) < 500:
                continue
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            (x, y), (w, h), angle = rect
            w_cm = w / self.PIXELS_PER_CM
            h_cm = h / self.PIXELS_PER_CM
            objects.append({"box": box, "center": (int(x), int(y)),
                            "w_cm": w_cm, "h_cm": h_cm})
        return objects

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        annotated = frame.copy()
        for obj in output:
            cv2.drawContours(annotated, [obj["box"]], 0, (0, 255, 0), 2)
            label = f"{obj['w_cm']:.1f}x{obj['h_cm']:.1f}cm"
            cv2.putText(annotated, label, (obj["center"][0] - 30, obj["center"][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(annotated, f"Objects: {len(output)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return annotated
