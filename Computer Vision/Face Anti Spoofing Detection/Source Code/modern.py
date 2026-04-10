"""Modern v2 pipeline — Face Anti-Spoofing Detection.

Replaces: Haar cascade face detector + Keras liveness classifier
Uses:     YOLO face detector for face ROI + LBP texture analysis (primary)
          Haar cascade + LBP as fallback

Pipeline: YOLO face detector → crop face ROI → LBP variance → classify
          Real face textures have higher variance than printed/screen replays.
Note: LBP texture variance is a naive baseline.  Swap in a dedicated
      FAS model (e.g. MiniFASNet, CDCN) for production anti-spoofing.
      Register the model at registry key "face_antispoof".
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


def _lbp_variance(gray_crop: np.ndarray) -> float:
    """Compute Local Binary Pattern variance as a simple texture measure.

    Real face textures have higher variance than printed/screen replays.
    This is a naive baseline — replace with a trained FAS classifier.
    """
    if gray_crop.size < 100:
        return 0.0
    h, w = gray_crop.shape
    center = gray_crop[1:h - 1, 1:w - 1].astype(np.int16)
    lbp = np.zeros_like(center, dtype=np.uint8)
    for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]:
        neighbor = gray_crop[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx].astype(np.int16)
        lbp = (lbp << 1) | (neighbor >= center).astype(np.uint8)
    return float(np.var(lbp))


@register("face_anti_spoofing")
class FaceAntiSpoofingModern(CVProject):
    project_type = "detection"
    description = "Face-level liveness detection via YOLO face detector + LBP texture"
    legacy_tech = "Haar cascade + Keras anti-spoofing CNN"
    modern_tech = "YOLO face detector + LBP texture variance (baseline)"

    LBP_THRESHOLD = 800.0  # Tunable: lower → more sensitive
    _face_detector = None
    _haar = None
    _backend = "none"

    def load(self):
        # Priority 1: YOLO custom face detector for accurate face ROI
        try:
            from models.registry import resolve
            from utils.yolo import load_yolo
            weights, ver, is_default = resolve("face_anti_spoofing", "face_detect")
            w_path = Path(weights) if Path(weights).is_absolute() else Path(__file__).resolve().parents[2] / weights
            if w_path.exists():
                self._face_detector = load_yolo(str(w_path))
                self._backend = "yolo_face"
                print(f"  [face_anti_spoofing] YOLO face detector + LBP texture analysis ({weights})")
                return
        except Exception:
            pass

        # Priority 2: Haar cascade fallback
        self._haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._backend = "haar"
        print("  [face_anti_spoofing] Haar cascade + LBP texture analysis")

    def predict(self, input_data):
        frame = input_data if isinstance(input_data, np.ndarray) else cv2.imread(str(input_data))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = []

        if self._backend == "yolo_face":
            results = self._face_detector(frame, verbose=False)
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                crop = gray[y1:y2, x1:x2]
                lbp_var = _lbp_variance(crop)
                faces.append({
                    "box": (x1, y1, x2, y2),
                    "lbp_var": lbp_var,
                    "live": lbp_var > self.LBP_THRESHOLD,
                    "conf": float(box.conf[0]),
                })
        else:
            haar_rects = self._haar.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in haar_rects:
                crop = gray[y:y + h, x:x + w]
                lbp_var = _lbp_variance(crop)
                faces.append({
                    "box": (x, y, x + w, y + h),
                    "lbp_var": lbp_var,
                    "live": lbp_var > self.LBP_THRESHOLD,
                    "conf": 1.0,
                })

        return {"faces": faces, "backend": self._backend}

    def visualize(self, input_data, output):
        frame = input_data if isinstance(input_data, np.ndarray) else cv2.imread(str(input_data))
        vis = frame.copy()
        backend = output.get("backend", "?")
        for f in output.get("faces", []):
            x1, y1, x2, y2 = f["box"]
            color = (0, 255, 0) if f["live"] else (0, 0, 255)
            label = "LIVE" if f["live"] else "SPOOF"
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"{label} ({f['lbp_var']:.0f})", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(vis, f"[{backend}]", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        return vis
