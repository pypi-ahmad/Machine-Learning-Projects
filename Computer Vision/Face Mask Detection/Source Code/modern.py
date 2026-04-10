"""Modern v2 pipeline — Face Mask Detection.

Replaces: Notebook-only Keras/TF pipeline
Uses:     Custom YOLO face-mask detector (mask / no_mask / improper classes)

Pipeline: YOLO face-mask detector → face-level boxes with mask labels
Training: Train on face-level mask dataset (mask / no_mask / improper_mask)
          python train_detection.py --data face_mask.yaml --weights yolo26m.pt

Registry: resolve("face_mask_detection", "detect") → weights/face_mask_yolo26m.pt

Merged: Absorbs "Project 47 - Face Mask Detection" (identical pipeline).
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


@register("face_mask_detection")
@register("face_mask_detection_v2")
class FaceMaskDetectionModern(CVProject):
    project_type = "detection"
    description = "Face-level mask detection via custom YOLO detector"
    legacy_tech = "Keras/TF CNN (notebook)"
    modern_tech = "YOLO face-mask detector (mask / no_mask / improper)"

    CONF_THRESHOLD = 0.4

    def load(self):
        weights, ver, fallback = resolve("face_mask_detection", "detect")
        w_path = Path(weights) if Path(weights).is_absolute() else Path(__file__).resolve().parents[2] / weights
        self.model = load_yolo(str(w_path) if w_path.exists() else weights)
        if fallback and not w_path.exists():
            print(f"  [face_mask] Using COCO pretrained ({weights}) — train custom mask weights:")
            print("  [face_mask]   python train_detection.py --data face_mask.yaml --weights yolo26m.pt")
        else:
            print(f"  [face_mask] Custom face-mask detector: {weights}")

    def predict(self, input_data):
        frame = input_data if isinstance(input_data, np.ndarray) else cv2.imread(str(input_data))
        results = self.model(frame, verbose=False, conf=self.CONF_THRESHOLD)
        faces = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            cls_id = int(box.cls[0])
            label = results[0].names.get(cls_id, f"class_{cls_id}")
            faces.append({
                "box": (x1, y1, x2, y2),
                "label": label,
                "conf": float(box.conf[0]),
            })
        return {"faces": faces, "backend": "yolo"}

    def visualize(self, input_data, output):
        if isinstance(output, dict) and "faces" in output:
            frame = input_data if isinstance(input_data, np.ndarray) else cv2.imread(str(input_data))
            vis = frame.copy()
            colors = {"mask": (0, 255, 0), "no_mask": (0, 0, 255), "improper_mask": (0, 165, 255)}
            for f in output["faces"]:
                x1, y1, x2, y2 = f["box"]
                label = f["label"]
                color = colors.get(label, (255, 200, 0))
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, f"{label} {f['conf']:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            n = len(output["faces"])
            cv2.putText(vis, f"Faces: {n}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            return vis
        return output[0].plot()
