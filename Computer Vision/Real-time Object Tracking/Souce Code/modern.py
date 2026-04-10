"""Modern v2 pipeline — Real-time Object Tracking.

Uses:     YOLO26s/m detect + Ultralytics built-in tracker (ByteTrack/BoT-SORT)

Pipeline: YOLO detection → built-in tracker with persist=True
Note:     Use yolo26s for max FPS on constrained GPU (4 GB),
          yolo26m when accuracy is the priority.
          Tracker config and trajectory smoothing can be tuned via
          tracker='bytetrack.yaml' or tracker='botsort.yaml'.

The original Flask implementation is preserved in webapp.py.
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
_custom_model = paths.models("realtime_object_tracking") / "best.pt"


@register("realtime_object_tracking")
class RealtimeObjectTrackingModern(CVProject):
    project_type = "tracking"
    description = "Real-time multi-object tracking (YOLO detect + ByteTrack/BoT-SORT)"
    legacy_tech = "YOLOv5 (subprocess) + Flask"
    modern_tech = "YOLO26s/m detect + Ultralytics tracker"

    CONF_THRESHOLD = 0.3

    def load(self):
        weights, ver, fallback = resolve("realtime_object_tracking", "tracking")
        if not fallback:
            self.model = load_yolo(weights)
            print(f"Using model for realtime_object_tracking: version={ver} weights={weights} pretrained_fallback=False")
            return
        if _custom_model.exists():
            try:
                from ultralytics import YOLO
                self.model = YOLO(str(_custom_model))
                print(f"Using model for realtime_object_tracking: version=legacy weights={_custom_model} pretrained_fallback=False")
                return
            except Exception:
                pass
        self.model = load_yolo(weights)
        print(f"Using model for realtime_object_tracking: version={ver} weights={weights} pretrained_fallback={fallback}")

    def predict(self, input_data):
        try:
            return self.model.track(input_data, persist=True, verbose=False,
                                    conf=self.CONF_THRESHOLD)
        except Exception:
            return self.model(input_data, verbose=False, conf=self.CONF_THRESHOLD)

    def visualize(self, input_data, output):
        annotated = output[0].plot()
        n = len(output[0].boxes) if output[0].boxes is not None else 0
        # Show tracked object IDs if available
        ids = output[0].boxes.id
        n_tracked = len(ids) if ids is not None else 0
        label = f"Tracked: {n_tracked}" if n_tracked else f"Detected: {n}"
        cv2.putText(annotated, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return annotated
