"""
Modern Ball Tracking — YOLO v2
================================
Replaces legacy HSV color-range tracking with YOLO26 object tracking.

Original: ballTracking.py (HSV range + contour center tracking)
Modern:   YOLO26 detection + built-in object tracking

Usage:
    python -m core.runner --import-all ball_tracking_v2 --source 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register
from utils.yolo import load_yolo


# COCO class 32 = sports ball
SPORTS_BALL_CLASS = 32


@register("ball_tracking_v2")
class BallTrackingV2(CVProject):
    display_name = "Ball Tracking (YOLO)"
    category = "detection"

    CONF_THRESHOLD = 0.3

    def load(self):
        from models.registry import resolve
        weights, version, is_default = resolve("ball_tracking", "detect")
        print(f"  [ball_tracking] version={version}  weights={weights}  pretrained_fallback={is_default}")
        self.model = load_yolo(weights)

    def predict(self, frame: np.ndarray):
        results = self.model.track(
            frame, verbose=False, conf=self.CONF_THRESHOLD,
            classes=[SPORTS_BALL_CLASS], persist=True,
        )
        return results

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        annotated = output[0].plot()
        # Show count of detected balls
        n_balls = len(output[0].boxes)
        cv2.putText(
            annotated, f"Balls detected: {n_balls}",
            (10, annotated.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )
        return annotated
