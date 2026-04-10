"""
Modern Car Detection — YOLO v2
================================
Replaces legacy Haar cascade vehicle detection with YOLO26.

Original: Vehicles_detection.py (Haar cascade on video)
Modern:   YOLO26 detection (car=2, bus=5, truck=7 in COCO)

Usage:
    python -m core.runner --import-all car_detection_v2 --source path/to/video.mp4
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register
from utils.yolo import load_yolo


# COCO vehicle class IDs
VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck", 3: "motorcycle"}


@register("car_detection_v2")
class CarDetectionV2(CVProject):
    display_name = "Car Detection (YOLO)"
    category = "detection"

    CONF_THRESHOLD = 0.4

    def load(self):
        from models.registry import resolve
        weights, version, is_default = resolve("car_detection", "detect")
        print(f"  [car_detection] version={version}  weights={weights}  pretrained_fallback={is_default}")
        self.model = load_yolo(weights)

    def predict(self, frame: np.ndarray):
        results = self.model.track(
            frame, verbose=False, conf=self.CONF_THRESHOLD,
            classes=list(VEHICLE_CLASSES.keys()), persist=True,
        )
        return results

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        annotated = output[0].plot()
        n = len(output[0].boxes)
        cv2.putText(annotated, f"Vehicles: {n}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return annotated
