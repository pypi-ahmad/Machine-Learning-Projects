"""YOLO-based license plate detector.

Uses YOLO26m to detect license plates in images/frames and returns
structured :class:`PlateDetection` results with bounding boxes and
confidence scores.

Usage::

    from plate_detector import PlateDetector
    from config import PlateConfig

    detector = PlateDetector(PlateConfig())
    detections = detector.detect(frame)
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("plate_reader.detector")


@dataclass
class PlateDetection:
    """A single detected license plate."""

    x1: int
    y1: int
    x2: int
    y2: int
    det_confidence: float
    crop: np.ndarray             # raw crop from frame
    rectified: np.ndarray | None = None  # rectified crop (if enabled)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


class PlateDetector:
    """Detect and optionally rectify license plates using YOLO26m."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._model = None

    def _init_model(self) -> None:
        try:
            from models.registry import resolve
            weights, ver, fallback = resolve("number_plate_reader_pro", "detect")
            log.info(
                "Resolved weights: version=%s, weights=%s, fallback=%s",
                ver, weights, fallback,
            )
        except Exception:
            weights = self.cfg.det_model
            log.info("Using default weights: %s", weights)

        from utils.yolo import load_yolo
        self._model = load_yolo(weights)
        log.info("YOLO plate detector loaded: %s", weights)

    def detect(self, frame: np.ndarray) -> list[PlateDetection]:
        """Detect license plates in *frame* (BGR)."""
        if self._model is None:
            self._init_model()

        results = self._model(
            frame,
            verbose=False,
            conf=self.cfg.det_confidence,
            iou=self.cfg.det_iou,
            imgsz=self.cfg.det_imgsz,
        )

        h, w = frame.shape[:2]
        detections: list[PlateDetection] = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop_w = x2 - x1
            crop_h = y2 - y1
            if crop_w < self.cfg.min_crop_width or crop_h < 5:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            rectified = self._rectify(crop) if self.cfg.rectify else None

            detections.append(PlateDetection(
                x1=x1, y1=y1, x2=x2, y2=y2,
                det_confidence=float(box.conf[0]),
                crop=crop,
                rectified=rectified,
            ))

        detections.sort(key=lambda d: d.det_confidence, reverse=True)
        log.debug("Detected %d plates", len(detections))
        return detections

    # ------------------------------------------------------------------
    # Rectification
    # ------------------------------------------------------------------

    def _rectify(self, crop: np.ndarray) -> np.ndarray:
        """Upscale and convert to grayscale for better OCR accuracy."""
        h, w = crop.shape[:2]

        # Convert to grayscale
        grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Upscale small crops
        if w < self.cfg.upscale_threshold:
            scale = self.cfg.upscale_threshold / w
            grey = cv2.resize(
                grey, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )

        # Adaptive threshold for cleaner character edges
        processed = cv2.adaptiveThreshold(
            grey, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )

        # Resize to standard target size
        processed = cv2.resize(
            processed,
            (self.cfg.target_width, self.cfg.target_height),
            interpolation=cv2.INTER_LINEAR,
        )

        return processed
