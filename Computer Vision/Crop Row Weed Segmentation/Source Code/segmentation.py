"""Crop Row & Weed Segmentation — YOLO26m-seg multi-class extraction.
"""Crop Row & Weed Segmentation — YOLO26m-seg multi-class extraction.

Wraps Ultralytics YOLO26m-seg to produce per-instance masks with class
labels, confidences, and bounding boxes.
"""
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


@dataclass
class InstanceMask:
    """One detected instance (crop, weed, etc.)."""

    mask: np.ndarray                # binary uint8 (H, W) -- 255 = instance
    confidence: float
    class_id: int
    class_name: str
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    area_px: int


@dataclass
class SegmentationResult:
    """Output of a single-image segmentation pass."""

    instances: list[InstanceMask] = field(default_factory=list)
    class_masks: dict[str, np.ndarray] = field(default_factory=dict)
    image_hw: tuple[int, int] = (0, 0)

    @property
    def count(self) -> int:
        return len(self.instances)


class CropWeedSegmenter:
    """YOLO26m-seg wrapper for multi-class crop/weed segmentation."""

    def __init__(
        self,
        model_name: str = "yolo26m-seg.pt",
        confidence: float = 0.30,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        class_names: list[str] | None = None,
    ) -> None:
        self._model_name = model_name
        self._confidence = confidence
        self._iou = iou_threshold
        self._imgsz = imgsz
        self._class_names = class_names or []
        self._model = None

    def load(self) -> None:
        """Load model weights (via resolve -> load_yolo)."""
        from models.registry import resolve
        from utils.yolo import load_yolo

        weights, ver, fallback = resolve("crop_row_weed_segmentation", "seg")
        if fallback:
            print(
                f"[segmentation] Using pretrained fallback: {weights}  "
                "(fine-tune on crop/weed data for best results)"
            )
        else:
            print(f"[segmentation] Custom weights: {weights}  version={ver}")
        self._model = load_yolo(weights)

        # Update class names from model if available and user didn't override
        if not self._class_names and hasattr(self._model, "names"):
            self._class_names = list(self._model.names.values())

    def segment(self, image: np.ndarray) -> SegmentationResult:
        """Segment crops, weeds, and other classes in *image* (BGR, uint8)."""
        if self._model is None:
            self.load()

        results = self._model(
            image,
            verbose=False,
            conf=self._confidence,
            iou=self._iou,
            imgsz=self._imgsz,
            retina_masks=True,
        )
        return self._extract(results[0], image.shape[:2])

    def close(self) -> None:
        self._model = None

    # ── internals ──────────────────────────────────────────
    def _extract(self, result, hw: tuple[int, int]) -> SegmentationResult:
        h, w = hw
        instances: list[InstanceMask] = []
        # Per-class combined masks
        class_accum: dict[str, np.ndarray] = {}

        if result.masks is not None and len(result.masks):
            masks_data = result.masks.data.cpu().numpy()
            boxes = result.boxes
            model_names = result.names if hasattr(result, "names") else {}

            for i in range(len(masks_data)):
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                cls_name = model_names.get(cls_id, f"class_{cls_id}")

                m = masks_data[i]
                if m.shape != (h, w):
                    m = cv2.resize(
                        m.astype(np.float32), (w, h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                binary = (m > 0.5).astype(np.uint8) * 255
                area = int((binary > 0).sum())

                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))

                instances.append(InstanceMask(
                    mask=binary,
                    confidence=round(conf, 4),
                    class_id=cls_id,
                    class_name=cls_name,
                    bbox=bbox,
                    area_px=area,
                ))

                if cls_name not in class_accum:
                    class_accum[cls_name] = np.zeros((h, w), dtype=np.uint8)
                class_accum[cls_name] = cv2.bitwise_or(
                    class_accum[cls_name], binary,
                )

        return SegmentationResult(
            instances=instances,
            class_masks=class_accum,
            image_hw=(h, w),
        )
