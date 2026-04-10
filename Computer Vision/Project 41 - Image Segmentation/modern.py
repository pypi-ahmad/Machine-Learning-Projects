"""
Modern Image Segmentation — YOLO-Seg v2
=========================================
Replaces legacy OpenCV watershed segmentation with YOLO26m-seg.

Original: Image segmentation (openCV).ipynb (watershed-based)
Modern:   YOLO26m-seg (instance segmentation with per-object masks)

Usage:
    python -m core.runner --import-all image_segmentation_v2 --source image.jpg
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register
from utils.yolo import load_yolo


@register("image_segmentation_v2")
class ImageSegmentationV2(CVProject):
    display_name = "Image Segmentation (YOLO26m-seg)"
    category = "segmentation"

    CONF_THRESHOLD = 0.4

    def load(self):
        from models.registry import resolve
        weights, version, is_default = resolve("image_segmentation", "seg")
        print(f"  [image_segmentation] version={version}  weights={weights}  pretrained_fallback={is_default}")
        self.model = load_yolo(weights)

    def predict(self, frame: np.ndarray):
        results = self.model(frame, verbose=False, conf=self.CONF_THRESHOLD)
        return results

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        annotated = output[0].plot()
        n_objects = len(output[0].boxes) if output[0].boxes is not None else 0
        cv2.putText(
            annotated, f"Segments: {n_objects}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )
        return annotated
