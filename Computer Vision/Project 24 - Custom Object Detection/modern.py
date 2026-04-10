"""
Modern Custom Object Detection — YOLO v2
==========================================
Replaces legacy Haar cascade collection with YOLO26m + custom weights.

Original: objectDetectoin.py (Haar cascades directory, 18 XML classifiers)
Modern:   YOLO26m detection with custom-trained weights

Usage:
    python -m core.runner --import-all custom_object_detection_v2 --source 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from core.base import CVProject
from core.registry import register
from utils.yolo import load_yolo


@register("custom_object_detection_v2")
class CustomObjectDetectionV2(CVProject):
    display_name = "Custom Object Detection (YOLO26m)"
    category = "detection"

    CONF_THRESHOLD = 0.3

    def load(self):
        from models.registry import resolve
        weights, version, is_default = resolve("custom_object_detection", "detect")
        if is_default:
            print(f"  [custom_object_detection] Using pretrained COCO — register custom weights for domain-specific detection")
        print(f"  [custom_object_detection] version={version}  weights={weights}  pretrained_fallback={is_default}")
        self.model = load_yolo(weights)

    def predict(self, frame: np.ndarray):
        results = self.model(frame, verbose=False, conf=self.CONF_THRESHOLD)
        return results

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        return output[0].plot()
