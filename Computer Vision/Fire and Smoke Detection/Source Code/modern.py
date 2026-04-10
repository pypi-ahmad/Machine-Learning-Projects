"""Modern v2 pipeline — Fire and Smoke Detection.

Uses:     Custom YOLO model (already modern — Phase 1B migrated to ultralytics)

The original implementation is preserved in main.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2

from core.base import CVProject
from core.registry import register
from utils.yolo import load_yolo
from utils.paths import PathResolver
from models.registry import resolve

paths = PathResolver()
_custom_model = paths.models("fire_and_smoke_detection") / "best.pt"


@register("fire_smoke_detection")
class FireSmokeDetectionModern(CVProject):
    project_type = "detection"
    description = "Fire & smoke detection with temporal smoothing and alert thresholds"
    legacy_tech = "YOLOv5 (subprocess)"
    modern_tech = "Ultralytics YOLO (custom weights) + temporal hysteresis"

    ALERT_FRAMES = 3       # Consecutive frames with fire to trigger alert
    COOLDOWN_FRAMES = 10   # Frames after alert clears before resetting

    def load(self):
        self._fire_count = 0
        self._cooldown = 0
        self._alert_active = False
        weights, ver, fallback = resolve("fire_smoke_detection", "detect")
        if not fallback:
            self.model = load_yolo(weights)
            print(f"Using model for fire_smoke_detection: version={ver} weights={weights} pretrained_fallback=False")
            return
        if _custom_model.exists():
            try:
                from ultralytics import YOLO
                self.model = YOLO(str(_custom_model))
                print(f"Using model for fire_smoke_detection: version=legacy weights={_custom_model} pretrained_fallback=False")
                return
            except Exception:
                pass  # YOLOv5 weights incompatible with ultralytics v8+
        self.model = load_yolo(weights)
        print(f"Using model for fire_smoke_detection: version={ver} weights={weights} pretrained_fallback={fallback}")

    def predict(self, input_data):
        results = self.model(input_data, verbose=False)
        # Temporal smoothing: count consecutive frames with fire detections
        has_fire = len(results[0].boxes) > 0
        if has_fire:
            self._fire_count = min(self._fire_count + 1, self.ALERT_FRAMES + 5)
            self._cooldown = self.COOLDOWN_FRAMES
        else:
            if self._cooldown > 0:
                self._cooldown -= 1
            else:
                self._fire_count = max(0, self._fire_count - 1)

        self._alert_active = self._fire_count >= self.ALERT_FRAMES
        return results

    def visualize(self, input_data, output):
        annotated = output[0].plot()
        if self._alert_active:
            cv2.putText(annotated, "FIRE ALERT", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        return annotated
