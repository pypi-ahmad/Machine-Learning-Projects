"""Wound Area Measurement — YOLO26m-seg wound region extraction.

Wraps Ultralytics YOLO26m-seg to produce per-instance wound masks
from a single image.
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
class WoundInstance:
    """One detected wound region."""

    mask: np.ndarray                # binary uint8 (H, W) — 255 = wound
    confidence: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    area_px: int
    class_id: int = 0


@dataclass
class SegmentationResult:
    """Output of a single-image wound segmentation pass."""

    instances: list[WoundInstance] = field(default_factory=list)
    combined_mask: np.ndarray = field(default_factory=lambda: np.empty(0))
    image_hw: tuple[int, int] = (0, 0)

    @property
    def count(self) -> int:
        return len(self.instances)

    @property
    def total_area_px(self) -> int:
        return int((self.combined_mask > 127).sum()) if self.combined_mask.size else 0


class WoundSegmenter:
    """YOLO26m-seg wrapper for wound region extraction."""

    def __init__(
        self,
        model_name: str = "yolo26m-seg.pt",
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
    ) -> None:
        self._model_name = model_name
        self._confidence = confidence
        self._iou = iou_threshold
        self._imgsz = imgsz
        self._model = None

    def load(self) -> None:
        """Load model weights (via resolve → load_yolo)."""
        from models.registry import resolve
        from utils.yolo import load_yolo

        weights, ver, fallback = resolve("wound_area_measurement", "seg")
        if fallback:
            print(
                f"[segmentation] Using pretrained fallback: {weights}  "
                "(fine-tune on wound data for best results)"
            )
        else:
            print(f"[segmentation] Custom weights: {weights}  version={ver}")
        self._model = load_yolo(weights)

    def segment(self, image: np.ndarray) -> SegmentationResult:
        """Extract wound regions from *image* (BGR, uint8)."""
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
        combined = np.zeros((h, w), dtype=np.uint8)
        instances: list[WoundInstance] = []

        if result.masks is not None and len(result.masks):
            masks_data = result.masks.data.cpu().numpy()
            boxes = result.boxes

            for i in range(len(masks_data)):
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])

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

                instances.append(WoundInstance(
                    mask=binary,
                    confidence=round(conf, 4),
                    bbox=bbox,
                    area_px=area,
                    class_id=cls_id,
                ))
                combined = cv2.bitwise_or(combined, binary)

        return SegmentationResult(
            instances=instances,
            combined_mask=combined,
            image_hw=(h, w),
        )
