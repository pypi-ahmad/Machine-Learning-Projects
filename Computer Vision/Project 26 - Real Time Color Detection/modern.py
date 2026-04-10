"""
Modern Color Detection — CVProject wrapper v2
===============================================
Wraps live HSV color detection/tracking in the unified CVProject framework.

Original: LiveHSVAdjustor.py (HSV thresholding + trackbars)
Modern:   Same core logic, unified interface

Usage:
    python -m core.runner --import-all color_detection_v2 --source 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


@register("color_detection_v2")
class ColorDetectionV2(CVProject):
    display_name = "Color Detection (v2)"
    category = "opencv_utility"

    # Default: detect red
    LOWER_HSV = np.array([0, 120, 70])
    UPPER_HSV = np.array([10, 255, 255])
    LOWER_HSV2 = np.array([170, 120, 70])
    UPPER_HSV2 = np.array([180, 255, 255])

    def load(self):
        pass

    def predict(self, frame: np.ndarray):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.LOWER_HSV, self.UPPER_HSV)
        mask2 = cv2.inRange(hsv, self.LOWER_HSV2, self.UPPER_HSV2)
        mask = mask1 | mask2
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 500:
                x, y, w, h = cv2.boundingRect(c)
                detections.append({"bbox": (x, y, w, h), "area": area})
        return {"mask": mask, "detections": detections}

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        annotated = frame.copy()
        for d in output["detections"]:
            x, y, w, h = d["bbox"]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(annotated, "Red", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(annotated, f"Detected: {len(output['detections'])} regions", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return annotated
