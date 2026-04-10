"""Modern v2 pipeline — Celebrity Face Recognition.

Replaces: Keras model + hardcoded label dict
Uses:     YOLO face detector for face ROI + InsightFace ArcFace embeddings

Pipeline: YOLO face detector → crop faces → InsightFace ArcFace embeddings
          Compare embeddings against gallery for open-set recognition.
Fallback: InsightFace single-pass (detection + embeddings) if no YOLO face weights
          DeepFace as secondary fallback

The original implementation is preserved in test.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


@register("celebrity_face_recognition")
class CelebrityFaceModern(CVProject):
    project_type = "detection"
    description = "Face detection + ArcFace embedding extraction for ID lookup"
    legacy_tech = "Keras CNN + hardcoded 105-class labels"
    modern_tech = "YOLO face detector + InsightFace ArcFace embeddings"

    CONF_THRESHOLD = 0.4
    _face_detector = None
    _insightface = None
    _deepface = None
    _backend = "none"
    _det_backend = None  # "yolo_face" or None

    def load(self):
        # Face ROI: custom YOLO face detector
        try:
            from models.registry import resolve
            from utils.yolo import load_yolo
            weights, ver, is_default = resolve("celebrity_face_recognition", "face_detect")
            w_path = Path(weights) if Path(weights).is_absolute() else Path(__file__).resolve().parents[2] / weights
            if w_path.exists():
                self._face_detector = load_yolo(str(w_path))
                self._det_backend = "yolo_face"
        except Exception:
            pass

        # Embedding extraction: InsightFace (primary)
        try:
            from insightface.app import FaceAnalysis
            self._insightface = FaceAnalysis(
                name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._insightface.prepare(ctx_id=0, det_size=(640, 640))
            self._backend = "insightface"
            if self._det_backend == "yolo_face":
                print("  [celebrity] YOLO face detector + InsightFace ArcFace embeddings")
            else:
                print("  [celebrity] InsightFace single-pass (detection + ArcFace embeddings)")
            return
        except (ImportError, Exception) as exc:
            pass

        # Fallback: DeepFace
        try:
            import deepface as _df
            self._deepface = _df
            self._backend = "deepface"
            if self._det_backend == "yolo_face":
                print("  [celebrity] YOLO face detector + DeepFace ArcFace embeddings")
            else:
                print("  [celebrity] DeepFace single-pass (detection + ArcFace embeddings)")
            return
        except ImportError:
            pass

        print("  [celebrity] No face-analysis library — install insightface or deepface")

    def predict(self, input_data):
        frame = input_data if isinstance(input_data, np.ndarray) else cv2.imread(str(input_data))

        # Pipeline A: YOLO face detect → InsightFace embeddings on crops
        if self._det_backend == "yolo_face" and self._insightface:
            det_results = self._face_detector(frame, verbose=False, conf=self.CONF_THRESHOLD)
            faces = []
            for box in det_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop_faces = self._insightface.get(crop)
                emb = crop_faces[0].embedding.tolist() if crop_faces and crop_faces[0].embedding is not None else None
                face_info = {
                    "box": (x1, y1, x2, y2),
                    "conf": float(box.conf[0]),
                }
                if emb:
                    face_info["embedding"] = emb
                    face_info["embedding_model"] = "ArcFace"
                faces.append(face_info)
            return {"faces": faces, "backend": "yolo_face+insightface"}

        # Pipeline B: InsightFace single-pass
        if self._backend == "insightface":
            results = self._insightface.get(frame)
            faces = []
            for face in results:
                box = face.bbox.astype(int)
                face_info = {
                    "box": (int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                    "conf": float(face.det_score),
                }
                if face.embedding is not None:
                    face_info["embedding"] = face.embedding.tolist()
                    face_info["embedding_model"] = "ArcFace"
                faces.append(face_info)
            return {"faces": faces, "backend": "insightface"}

        # Pipeline C: DeepFace
        if self._backend == "deepface":
            try:
                representations = self._deepface.DeepFace.represent(
                    frame, model_name="ArcFace", enforce_detection=True,
                )
                if not isinstance(representations, list):
                    representations = [representations]
                faces = []
                for rep in representations:
                    r = rep.get("facial_area", {})
                    x, y, w, h = r.get("x", 0), r.get("y", 0), r.get("w", 0), r.get("h", 0)
                    face_info = {
                        "box": (x, y, x + w, y + h),
                        "conf": rep.get("face_confidence", 0.0),
                    }
                    if "embedding" in rep:
                        face_info["embedding"] = rep["embedding"]
                        face_info["embedding_model"] = "ArcFace"
                    faces.append(face_info)
                return {"faces": faces, "backend": "deepface"}
            except Exception:
                return {"faces": [], "backend": "deepface"}

        return {"faces": [], "backend": "none"}

    def visualize(self, input_data, output):
        if isinstance(output, dict):
            frame = input_data if isinstance(input_data, np.ndarray) else cv2.imread(str(input_data))
            vis = frame.copy()
            backend = output.get("backend", "?")
            for f in output.get("faces", []):
                x1, y1, x2, y2 = f["box"]
                has_emb = "embedding" in f
                color = (0, 255, 0) if has_emb else (0, 165, 255)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                label = f"face {f['conf']:.2f}"
                if has_emb:
                    label += f" [{f.get('embedding_model', 'ArcFace')}]"
                cv2.putText(vis, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            n = len(output.get("faces", []))
            cv2.putText(vis, f"Faces: {n} [{backend}]", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            return vis
        return output[0].plot()
