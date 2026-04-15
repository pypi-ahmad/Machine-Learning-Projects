"""Fire Area Segmentation — YOLO26m-seg fire/smoke extraction.
"""Fire Area Segmentation — YOLO26m-seg fire/smoke extraction.

Wraps Ultralytics YOLO26m-seg to produce per-instance fire (and
optionally smoke) masks from a single image.
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
class FireInstance:
    """One detected fire or smoke region."""

    mask: np.ndarray                # binary uint8 (H, W) -- 255 = region
    confidence: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    area_px: int
    class_id: int
    class_name: str                 # "fire" | "smoke"


@dataclass
class SegmentationResult:
    """Output of a single-image fire/smoke segmentation pass."""

    instances: list[FireInstance] = field(default_factory=list)
    fire_mask: np.ndarray = field(default_factory=lambda: np.empty(0))
    smoke_mask: np.ndarray = field(default_factory=lambda: np.empty(0))
    image_hw: tuple[int, int] = (0, 0)

    @property
    def fire_count(self) -> int:
        return sum(1 for i in self.instances if i.class_name == "fire")

    @property
    def smoke_count(self) -> int:
        return sum(1 for i in self.instances if i.class_name == "smoke")

    @property
    def total_count(self) -> int:
        return len(self.instances)

    @property
    def fire_area_px(self) -> int:
        return int((self.fire_mask > 127).sum()) if self.fire_mask.size else 0

    @property
    def smoke_area_px(self) -> int:
        return int((self.smoke_mask > 127).sum()) if self.smoke_mask.size else 0


class FireSegmenter:
    """YOLO26m-seg wrapper for fire / smoke region extraction."""

    def __init__(
        self,
        model_name: str = "yolo26m-seg.pt",
        confidence: float = 0.30,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        class_names: tuple[str, ...] = ("fire", "smoke"),
        enable_smoke: bool = True,
    ) -> None:
        self._model_name = model_name
        self._confidence = confidence
        self._iou = iou_threshold
        self._imgsz = imgsz
        self._class_names = class_names
        self._enable_smoke = enable_smoke
        self._model = None

    def load(self) -> None:
        """Load model weights (via resolve -> load_yolo)."""
        from models.registry import resolve
        from utils.yolo import load_yolo

        weights, ver, fallback = resolve("fire_area_segmentation", "seg")
        if fallback:
            print(
                f"[segmentation] Using pretrained fallback: {weights}  "
                "(fine-tune on fire/smoke data for best results)"
            )
        else:
            print(f"[segmentation] Custom weights: {weights}  version={ver}")
        self._model = load_yolo(weights)

    def segment(self, image: np.ndarray) -> SegmentationResult:
        """Extract fire/smoke regions from *image* (BGR, uint8)."""
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

    def _class_label(self, cls_id: int) -> str:
        """Map YOLO class-id to a human label."""
        if cls_id < len(self._class_names):
            return self._class_names[cls_id]
        return f"class_{cls_id}"

    def _extract(self, result, hw: tuple[int, int]) -> SegmentationResult:
        h, w = hw
        fire_combined = np.zeros((h, w), dtype=np.uint8)
        smoke_combined = np.zeros((h, w), dtype=np.uint8)
        instances: list[FireInstance] = []

        if result.masks is not None and len(result.masks):
            masks_data = result.masks.data.cpu().numpy()
            boxes = result.boxes

            for i in range(len(masks_data)):
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                label = self._class_label(cls_id)

                # Skip smoke instances when disabled
                if label == "smoke" and not self._enable_smoke:
                    continue

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

                instances.append(FireInstance(
                    mask=binary,
                    confidence=round(conf, 4),
                    bbox=bbox,
                    area_px=area,
                    class_id=cls_id,
                    class_name=label,
                ))

                if label == "fire":
                    fire_combined = cv2.bitwise_or(fire_combined, binary)
                else:
                    smoke_combined = cv2.bitwise_or(smoke_combined, binary)

        return SegmentationResult(
            instances=instances,
            fire_mask=fire_combined,
            smoke_mask=smoke_combined,
            image_hw=(h, w),
        )
