"""Modern v2 pipeline — Emotion Recognition from Facial Expression.
"""Modern v2 pipeline — Emotion Recognition from Facial Expression.

Replaces: Haar cascade face detector + Keras emotion classifier
Uses:     YOLO face detector for face ROI + DeepFace emotion analysis

Pipeline: YOLO face detector → crop face ROI → DeepFace.analyze(crop, emotion)
Fallback: DeepFace single-pass (detection + analysis) if no YOLO face weights

Merged: Absorbs "Face Emotion Recognition" (identical pipeline).
"""
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


@register("emotion_recognition")
@register("face_emotion_recognition")
class EmotionRecognitionModern(CVProject):
    project_type = "detection"
    description = "Face detection + DeepFace emotion analysis"
    legacy_tech = "Haar cascade + Keras CNN"
    modern_tech = "YOLO face detector + DeepFace emotion analysis"

    CONF_THRESHOLD = 0.4
    _deepface = None
    _face_detector = None
    _det_backend = None  # "yolo_face" or None

    def load(self):
        # Face ROI: custom YOLO face detector
        try:
            from models.registry import resolve
            from utils.yolo import load_yolo
            weights, ver, is_default = resolve("emotion_recognition", "face_detect")
            w_path = Path(weights) if Path(weights).is_absolute() else Path(__file__).resolve().parents[2] / weights
            if w_path.exists():
                self._face_detector = load_yolo(str(w_path))
                self._det_backend = "yolo_face"
        except Exception:
            pass

        # Attribute analysis: DeepFace
        try:
            from deepface import DeepFace
            self._deepface = DeepFace
        except ImportError:
            self._deepface = None

        if self._det_backend == "yolo_face" and self._deepface:
            print("  [emotion] YOLO face detector + DeepFace emotion analysis")
        elif self._deepface:
            print("  [emotion] DeepFace single-pass (detection + emotion analysis)")
        else:
            print("  [emotion] No backend available -- install deepface: pip install deepface")

    def predict(self, input_data):
        frame = input_data if isinstance(input_data, np.ndarray) else cv2.imread(str(input_data))

        # Pipeline A: YOLO face detect → crop → DeepFace emotion analysis
        if self._det_backend == "yolo_face" and self._deepface:
            results = self._face_detector(frame, verbose=False, conf=self.CONF_THRESHOLD)
            faces = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                try:
                    analyses = self._deepface.analyze(
                        crop, actions=["emotion"],
                        enforce_detection=False, silent=True,
                    )
                    if not isinstance(analyses, list):
                        analyses = [analyses]
                    a = analyses[0]
                    faces.append({
                        "box": (x1, y1, x2, y2),
                        "emotion": a.get("dominant_emotion"),
                        "conf": float(box.conf[0]),
                    })
                except Exception:
                    faces.append({
                        "box": (x1, y1, x2, y2),
                        "emotion": None,
                        "conf": float(box.conf[0]),
                    })
            return {"faces": faces}

        # Pipeline B: DeepFace single-pass (detection + emotion)
        if self._deepface is not None:
            try:
                analyses = self._deepface.analyze(
                    frame, actions=["emotion"],
                    enforce_detection=True, silent=True,
                )
                if not isinstance(analyses, list):
                    analyses = [analyses]
                faces = []
                for a in analyses:
                    r = a.get("region", {})
                    x, y, w, h = r.get("x", 0), r.get("y", 0), r.get("w", 0), r.get("h", 0)
                    faces.append({
                        "box": (x, y, x + w, y + h),
                        "emotion": a.get("dominant_emotion"),
                        "conf": a.get("face_confidence", 0.0),
                    })
                return {"faces": faces}
            except Exception:
                return {"faces": []}

        return {"faces": []}

    def visualize(self, input_data, output):
        if isinstance(output, dict):
            frame = input_data if isinstance(input_data, np.ndarray) else cv2.imread(str(input_data))
            vis = frame.copy()
            for f in output.get("faces", []):
                x1, y1, x2, y2 = f["box"]
                label = f["emotion"] or f"face {f['conf']:.2f}"
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 200, 0), 2)
                cv2.putText(vis, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
            return vis
        return output[0].plot()
