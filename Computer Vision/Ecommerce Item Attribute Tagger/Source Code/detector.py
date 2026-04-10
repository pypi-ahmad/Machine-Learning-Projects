"""Ecommerce Item Attribute Tagger — item detection / isolation.

For clean product images (studio shots) the full image is used.
When ``use_detector=True``, YOLO is used to detect the primary item
and crop it before attribute prediction.

Detection/isolation is fully separated from attribute prediction.

Usage::

    from detector import ItemDetector

    det = ItemDetector(cfg)
    det.load()
    crop, box = det.isolate(image)
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from config import TaggerConfig

log = logging.getLogger("attribute_tagger.detector")


class ItemDetector:
    """Detect and crop the primary item from a product image."""

    def __init__(self, cfg: TaggerConfig) -> None:
        self.cfg = cfg
        self._model = None

    def load(self) -> None:
        """Load YOLO model if detector is enabled."""
        if not self.cfg.use_detector:
            log.info("Detector disabled — using full image")
            return

        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from utils.yolo import load_yolo

        self._model = load_yolo(
            self.cfg.detector_model, device=self.cfg.device,
        )
        log.info("Item detector loaded: %s", self.cfg.detector_model)

    def isolate(
        self, image: np.ndarray,
    ) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
        """Isolate the primary item in the image.

        Returns
        -------
        tuple[np.ndarray, tuple | None]
            ``(cropped_image, (x1, y1, x2, y2))`` or
            ``(original_image, None)`` when no detection is used.
        """
        if self._model is None or not self.cfg.use_detector:
            return image, None

        results = self._model(
            image, verbose=False, conf=self.cfg.detector_conf,
        )

        best_box = None
        best_area = 0

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best_box = (x1, y1, x2, y2)

        if best_box is None:
            return image, None

        x1, y1, x2, y2 = best_box
        h, w = image.shape[:2]
        # Pad slightly
        pad = 10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return image, None

        return crop, (x1, y1, x2, y2)

    def detect_all(
        self, image: np.ndarray,
    ) -> list[dict]:
        """Detect all items in the image (for multi-item scenes).

        Returns
        -------
        list[dict]
            Each dict has: box, confidence, class_name, crop.
        """
        if self._model is None:
            return [{"box": None, "confidence": 1.0, "class_name": "full_image",
                     "crop": image}]

        results = self._model(
            image, verbose=False, conf=self.cfg.detector_conf,
        )
        items = []
        for result in results:
            if result.boxes is None:
                continue
            names = result.names
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                h, w = image.shape[:2]
                crop = image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if crop.size == 0:
                    continue
                items.append({
                    "box": (x1, y1, x2, y2),
                    "confidence": conf,
                    "class_name": names.get(cls_id, str(cls_id)),
                    "crop": crop,
                })

        if not items:
            return [{"box": None, "confidence": 1.0, "class_name": "full_image",
                     "crop": image}]

        return sorted(items, key=lambda x: x["confidence"], reverse=True)
