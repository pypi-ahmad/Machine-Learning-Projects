"""Modern v2 pipeline — Real-Time Face Detection.

Replaces: Caffe SSD face detector
Uses:     Custom YOLO face detector (primary, trained on WIDER FACE / FDDB)
          MediaPipe Face Detection (fallback if custom weights not available)
          Haar cascade (always available via OpenCV)

Pipeline: YOLO face detector → face bounding boxes
          OR MediaPipe face_detection → face bounding boxes
          OR Haar cascade → face rectangles

Note: YOLO26 pretrained on COCO detects *persons* (class 0 = full body),
NOT faces. COCO has no "face" class. The registry resolves face_detect to
custom YOLO weights trained on face data (WIDER FACE / FDDB).

Merged: Absorbs "Project 46 - Face Detection Second Approach" (Haar demo).

Usage:
    python -m core.runner --import-all face_detection_v2 --source 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


@register("face_detection_v2")
@register("face_detection_haar_v2")
class FaceDetectionV2(CVProject):
    display_name = "Face Detection (YOLO / MediaPipe / Haar)"
    category = "detection"

    CONF_THRESHOLD = 0.4
    _mp_face = None
    _haar = None
    _yolo = None
    _backend = "none"

    def load(self):
        # Priority 1: Custom YOLO face detector (trained on WIDER FACE / FDDB)
        try:
            from models.registry import resolve
            weights, version, is_default = resolve("face_detection", "face_detect")
            weights_path = Path(weights)
            if not weights_path.is_absolute():
                weights_path = Path(__file__).resolve().parent.parent / weights
            if weights_path.exists():
                from utils.yolo import load_yolo
                self._yolo = load_yolo(str(weights_path))
                self._backend = "yolo_custom"
                print(f"  [face_detection] YOLO face detector loaded: {weights}")
                return
        except Exception:
            pass

        # Priority 2: MediaPipe Face Detection
        try:
            import mediapipe as mp
            self._mp_face = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=self.CONF_THRESHOLD,
            )
            self._backend = "mediapipe"
            print("  [face_detection] MediaPipe Face Detection loaded")
            return
        except ImportError:
            pass

        # Priority 3: Haar cascade (always available via OpenCV)
        self._haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._backend = "haar"
        print("  [face_detection] Haar cascade face detection")

    def predict(self, frame: np.ndarray):
        if self._backend == "mediapipe":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self._mp_face.process(rgb)
            faces = []
            h, w = frame.shape[:2]
            if result.detections:
                for det in result.detections:
                    bb = det.location_data.relative_bounding_box
                    x1 = max(0, int(bb.xmin * w))
                    y1 = max(0, int(bb.ymin * h))
                    x2 = min(w, int((bb.xmin + bb.width) * w))
                    y2 = min(h, int((bb.ymin + bb.height) * h))
                    faces.append({
                        "box": (x1, y1, x2, y2),
                        "conf": det.score[0] if det.score else 0.0,
                    })
            return {"faces": faces, "backend": "mediapipe"}

        if self._backend == "yolo_custom":
            results = self._yolo(frame, verbose=False, conf=self.CONF_THRESHOLD)
            faces = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                faces.append({
                    "box": (x1, y1, x2, y2),
                    "conf": float(box.conf[0]),
                })
            return {"faces": faces, "backend": "yolo_custom"}

        # Haar cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self._haar.detectMultiScale(gray, 1.3, 5)
        faces = []
        for (x, y, w, h) in rects:
            faces.append({"box": (x, y, x + w, y + h), "conf": 1.0})
        return {"faces": faces, "backend": "haar"}

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        vis = frame.copy()
        backend = output.get("backend", "?")
        for f in output.get("faces", []):
            x1, y1, x2, y2 = f["box"]
            label = f"face {f['conf']:.2f}"
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
        n = len(output.get("faces", []))
        cv2.putText(vis, f"Faces: {n} [{backend}]", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        return vis
