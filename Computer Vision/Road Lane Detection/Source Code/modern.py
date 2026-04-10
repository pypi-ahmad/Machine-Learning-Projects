"""
Modern Road Lane Detection — CVProject wrapper v2
===================================================
Classical CV lane detection via Canny + Hough transform.
No DL replacement needed for basic lane overlay.

Upgrade note: For production, consider Ultra-Fast-Lane-Detection, CLRNet,
or a YOLOv8-seg lane-segmentation model for curved / multi-lane roads.

Original: detection_on_vid.py / detection_on_image.py
Modern:   Same core logic, unified interface

Usage:
    python -m core.runner --import-all road_lane_detection --source dashcam.mp4
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


@register("road_lane_detection")
class RoadLaneDetectionModern(CVProject):
    display_name = "Road Lane Detection"
    category = "opencv_utility"

    def load(self):
        pass

    def predict(self, frame: np.ndarray):
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.dilate(gray, kernel=np.ones((3, 3), np.uint8))
        canny = cv2.Canny(gray, 130, 220)

        roi_vertices = np.array(
            [[(0, height), (2 * width // 3, 2 * height // 3), (width, height)]],
            np.int32,
        )
        mask = np.zeros_like(canny)
        cv2.fillPoly(mask, roi_vertices, 255)
        roi_img = cv2.bitwise_and(canny, mask)

        lines = cv2.HoughLinesP(roi_img, 1, np.pi / 180, 10,
                                minLineLength=15, maxLineGap=2)
        return {"lines": lines, "canny": roi_img}

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        annotated = frame.copy()
        lines = output.get("lines")
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        n = len(lines) if lines is not None else 0
        cv2.putText(annotated, f"Lanes: {n} segments", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return annotated
