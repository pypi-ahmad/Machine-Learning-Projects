"""Modern v2 pipeline — Pedestrian Detection.

Replaces: OpenCV HOG + SVM
Uses:     YOLO26 person detection + ByteTrack for video

Pipeline: detect persons → track with persist=True for video streams
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register
from utils.yolo import load_yolo
from models.registry import resolve


@register("pedestrian_detection")
class PedestrianDetectionModern(CVProject):
    project_type = "detection"
    description = "Pedestrian detection + tracking (replaces HOG+SVM)"
    legacy_tech = "OpenCV HOG + SVM"
    modern_tech = "YOLO26 detect + ByteTrack"

    CONF_THRESHOLD = 0.4

    def load(self):
        weights, ver, fallback = resolve("pedestrian_detection", "detect")
        print(f"Using model for pedestrian_detection: version={ver} weights={weights} pretrained_fallback={fallback}")
        self.model = load_yolo(weights)

    def predict(self, input_data):
        return self.model.track(input_data, classes=[0], verbose=False,
                                conf=self.CONF_THRESHOLD, persist=True)

    def visualize(self, input_data, output):
        annotated = output[0].plot()
        n = len(output[0].boxes)
        cv2.putText(annotated, f"Pedestrians: {n}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return annotated
