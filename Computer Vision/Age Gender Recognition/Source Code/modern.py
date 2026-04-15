"""Modern v2 pipeline — Age & Gender Recognition.

Replaces: Caffe face detector + Caffe age/gender classifiers
Uses:     YOLO face detector for face ROI + DeepFace age/gender analysis

Pipeline: YOLO face detector → crop face ROI → DeepFace.analyze(crop)
Fallback: DeepFace single-pass (detection + analysis) if no YOLO face weights
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


@register("age_gender_recognition")
class AgeGenderModern(CVProject):
    project_type = "detection"
    description = "Face detection + DeepFace age/gender analysis"
    legacy_tech = "Caffe DNN face detector + Caffe age/gender nets"
    modern_tech = "YOLO face detector + DeepFace age/gender analysis"

    CONF_THRESHOLD = 0.4
    _deepface = None
    _face_detector = None
    _det_backend = None  # "yolo_face" or None

    def load(self):
        # Face ROI: custom YOLO face detector
        try:
            from models.registry import resolve
            from utils.yolo import load_yolo
            weights, ver, is_default = resolve("age_gender_recognition", "face_detect")
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
            print("  [age_gender] YOLO face detector + DeepFace age/gender analysis")
        elif self._deepface:
            print("  [age_gender] DeepFace single-pass (detection + analysis)")
        else:
            print("  [age_gender] No backend available -- install deepface: pip install deepface")

    def predict(self, input_data):
        frame = input_data if isinstance(input_data, np.ndarray) else cv2.imread(str(input_data))

        # Pipeline A: YOLO face detect → crop → DeepFace analysis
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
                        crop, actions=["age", "gender"],
                        enforce_detection=False, silent=True,
                    )
                    if not isinstance(analyses, list):
                        analyses = [analyses]
                    a = analyses[0]
                    faces.append({
                        "box": (x1, y1, x2, y2),
                        "age": a.get("age"),
                        "gender": a.get("dominant_gender"),
                        "conf": float(box.conf[0]),
                    })
                except Exception:
                    faces.append({
                        "box": (x1, y1, x2, y2),
                        "age": None, "gender": None,
                        "conf": float(box.conf[0]),
                    })
            return {"faces": faces}

        # Pipeline B: DeepFace single-pass (detection + analysis)
        if self._deepface is not None:
            try:
                analyses = self._deepface.analyze(
                    frame, actions=["age", "gender"],
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
                        "age": a.get("age"),
                        "gender": a.get("dominant_gender"),
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
                label_parts = []
                if f.get("age") is not None:
                    label_parts.append(f"Age:{f['age']}")
                if f.get("gender"):
                    label_parts.append(f"{f['gender']}")
                label = " | ".join(label_parts) if label_parts else f"face {f['conf']:.2f}"
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            return vis
        return output[0].plot()
