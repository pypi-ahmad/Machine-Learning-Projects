"""
Modern Object Detection — YOLO v2
===================================
Replaces legacy Caffe MobileNet-SSD with YOLO26m.

Original: real_time_object_detection.py (Caffe MobileNet-SSD, 20 classes)
Modern:   YOLO26m detection (80 COCO classes, much higher accuracy)

Usage:
    python -m core.runner --import-all object_detection_v2 --source 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from core.base import CVProject
from core.registry import register
from utils.yolo import load_yolo


@register("object_detection_v2")
class ObjectDetectionV2(CVProject):
    display_name = "Real-Time Object Detection (YOLO26m)"
    category = "detection"

    CONF_THRESHOLD = 0.25

    def load(self):
        from models.registry import resolve
        weights, version, is_default = resolve("object_detection", "detect")
        print(f"  [object_detection] version={version}  weights={weights}  pretrained_fallback={is_default}")
        self.model = load_yolo(weights)

    def predict(self, frame: np.ndarray):
        results = self.model(frame, verbose=False, conf=self.CONF_THRESHOLD)
        return results

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        return output[0].plot()
