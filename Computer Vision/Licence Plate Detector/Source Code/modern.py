"""Modern v2 pipeline — Licence Plate Detector.
"""Modern v2 pipeline — Licence Plate Detector.

Uses:     Custom YOLO model for plate detection + PaddleOCR for recognition

Pipeline: detect plates → crop + rectify ROI → OCR recognition (PaddleOCR)
Fallback: EasyOCR if PaddleOCR not installed

Merged: Absorbs "Project 37 - Number Plate Detection" (same pipeline
without plate rectification — rectification is now always applied).
"""
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register
from utils.yolo import load_yolo
from utils.paths import PathResolver
from models.registry import resolve

paths = PathResolver()
_custom_model = paths.models("licence_plate_detector") / "best.pt"


def _rectify_plate(crop: np.ndarray) -> np.ndarray:
    """Basic plate rectification: deskew via minimum-area rotated rect."""
    if crop.size == 0:
        return crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 5:
        return crop
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle += 90
    h, w = crop.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated


@register("licence_plate_detector")
@register("number_plate_detection_v2")
class LicencePlateDetectorModern(CVProject):
    project_type = "detection"
    description = "Licence plate detection + crop rectification + PaddleOCR"
    legacy_tech = "YOLOv5 (subprocess) + EasyOCR"
    modern_tech = "YOLO26 (custom weights) + plate rectification + PaddleOCR"

    _reader = None
    _ocr_backend = None  # "paddle" | "easyocr" | None

    def load(self):
        weights, ver, fallback = resolve("licence_plate_detector", "detect")
        if not fallback:
            self.model = load_yolo(weights)
            print(f"Using model for licence_plate_detector: version={ver} weights={weights} pretrained_fallback=False")
        elif _custom_model.exists():
            try:
                from ultralytics import YOLO
                self.model = YOLO(str(_custom_model))
                print(f"Using model for licence_plate_detector: version=legacy weights={_custom_model} pretrained_fallback=False")
            except Exception:
                self.model = load_yolo(weights)
                print(f"Using model for licence_plate_detector: version={ver} weights={weights} pretrained_fallback={fallback}")
        else:
            self.model = load_yolo(weights)
            print(f"Using model for licence_plate_detector: version={ver} weights={weights} pretrained_fallback={fallback}")

        # Priority 1: PaddleOCR (better accuracy, separates detection/recognition)
        try:
            from paddleocr import PaddleOCR
            self._reader = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=True, show_log=False)
            self._ocr_backend = "paddle"
            print("  [licence_plate] PaddleOCR loaded for plate text recognition")
            return
        except ImportError:
            pass

        # Priority 2: EasyOCR fallback
        try:
            import easyocr
            self._reader = easyocr.Reader(["en"], gpu=True)
            self._ocr_backend = "easyocr"
            print("  [licence_plate] PaddleOCR not installed -- using EasyOCR fallback")
            return
        except ImportError:
            pass

        print("  [licence_plate] No OCR engine installed -- detection only")

    def _ocr_read(self, crop: np.ndarray) -> str:
        """Read text from a plate crop using the available OCR engine."""
        if self._reader is None:
            return ""
        if self._ocr_backend == "paddle":
            result = self._reader.ocr(crop, cls=True)
            if result and result[0]:
                return " ".join(line[1][0] for line in result[0])
            return ""
        # easyocr
        ocr_result = self._reader.readtext(crop)
        return " ".join(r[1] for r in ocr_result) if ocr_result else ""

    def predict(self, input_data):
        results = self.model(input_data, verbose=False)
        if self._reader is None:
            return results

        frame = input_data if isinstance(input_data, np.ndarray) else cv2.imread(str(input_data))
        plates = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            crop = frame[y1:y2, x1:x2]
            rectified = _rectify_plate(crop)
            text = self._ocr_read(rectified)
            plates.append({
                "box": (x1, y1, x2, y2),
                "text": text.strip(),
                "conf": float(box.conf[0]),
            })
        return {"yolo_results": results, "plates": plates}

    def visualize(self, input_data, output):
        if isinstance(output, dict):
            frame = input_data if isinstance(input_data, np.ndarray) else cv2.imread(str(input_data))
            vis = frame.copy()
            for p in output.get("plates", []):
                x1, y1, x2, y2 = p["box"]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = p["text"] or f"plate {p['conf']:.2f}"
                cv2.putText(vis, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            return vis
        return output[0].plot()
