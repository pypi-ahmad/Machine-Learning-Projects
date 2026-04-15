"""Logo Retrieval Brand Match — optional logo detector / cropper.
"""Logo Retrieval Brand Match — optional logo detector / cropper.

Uses YOLO to detect and crop logo regions from scene images.
If no logo is detected, the full image is returned.
Detection and retrieval are kept separate — this module is optional.
"""
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DetectionResult:
    """Result of logo detection in an image."""

    crops: list[np.ndarray]            # cropped logo regions (BGR)
    boxes: list[tuple[int, int, int, int]]  # x1, y1, x2, y2
    confidences: list[float]
    full_image: np.ndarray

    @property
    def found(self) -> bool:
        return len(self.crops) > 0


class LogoDetector:
    """Optional YOLO-based logo region detector."""

    def __init__(
        self,
        model_name: str = "yolo26n.pt",
        confidence: float = 0.25,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._confidence = confidence
        self._device = device
        self._model = None

    def load(self) -> None:
        from ultralytics import YOLO
        self._model = YOLO(self._model_name)

    def detect(self, image_bgr: np.ndarray) -> DetectionResult:
        """Detect logo-like regions and return crops."""
        if self._model is None:
            self.load()

        results = self._model(
            image_bgr,
            conf=self._confidence,
            device=self._device,
            verbose=False,
        )

        crops, boxes, confs = [], [], []
        for r in results:
            if r.boxes is None:
                continue
            for box, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(image_bgr.shape[1], x2)
                y2 = min(image_bgr.shape[0], y2)
                crop = image_bgr[y1:y2, x1:x2]
                if crop.size > 0:
                    crops.append(crop)
                    boxes.append((x1, y1, x2, y2))
                    confs.append(float(conf))

        return DetectionResult(
            crops=crops,
            boxes=boxes,
            confidences=confs,
            full_image=image_bgr,
        )

    def close(self) -> None:
        self._model = None


def centre_crop(image_bgr: np.ndarray, ratio: float = 0.8) -> np.ndarray:
    """Simple centre crop -- useful when logos are centred in the image."""
    h, w = image_bgr.shape[:2]
    dh, dw = int(h * (1 - ratio) / 2), int(w * (1 - ratio) / 2)
    return image_bgr[dh:h - dh, dw:w - dw]
