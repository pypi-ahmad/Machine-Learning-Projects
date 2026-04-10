"""Building Footprint Change Detector — YOLO-seg building extraction.

Wraps Ultralytics YOLO26m-seg to produce a single binary mask per image
indicating building footprint coverage.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


@dataclass
class SegmentationResult:
    """Output of a single-image building segmentation pass."""

    mask: np.ndarray         # binary uint8 (H, W) — 255 = building
    instance_count: int      # number of instance masks merged
    confidences: list[float] # per-instance confidence scores


class BuildingSegmenter:
    """YOLO26m-seg wrapper for building footprint extraction."""

    def __init__(
        self,
        model_name: str = "yolo26m-seg.pt",
        confidence: float = 0.30,
        iou_threshold: float = 0.45,
        imgsz: int = 1024,
        use_all_classes: bool = True,
    ) -> None:
        self._model_name = model_name
        self._confidence = confidence
        self._iou = iou_threshold
        self._imgsz = imgsz
        self._use_all = use_all_classes
        self._model = None

    # ── lazy load ──────────────────────────────────────────
    def load(self) -> None:
        """Load model weights (via resolve → load_yolo)."""
        from models.registry import resolve
        from utils.yolo import load_yolo

        weights, ver, fallback = resolve(
            "building_footprint_change_detector", "seg",
        )
        if fallback:
            print(
                f"[segmentation] Using pretrained fallback: {weights}  "
                "(fine-tune on building data for best results)"
            )
        else:
            print(f"[segmentation] Custom weights: {weights}  version={ver}")
        self._model = load_yolo(weights)

    # ── public API ─────────────────────────────────────────
    def segment(self, image: np.ndarray) -> SegmentationResult:
        """Extract a binary building mask from *image* (BGR, uint8)."""
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
        return self._extract_mask(results[0], image.shape[:2])

    def close(self) -> None:
        self._model = None

    # ── internals ──────────────────────────────────────────
    def _extract_mask(self, result, hw: tuple[int, int]) -> SegmentationResult:
        h, w = hw
        combined = np.zeros((h, w), dtype=np.uint8)
        confs: list[float] = []
        count = 0

        if result.masks is not None and len(result.masks):
            masks_data = result.masks.data.cpu().numpy()  # (N, mh, mw)
            boxes = result.boxes

            for i in range(len(masks_data)):
                conf = float(boxes.conf[i])
                # Resize mask to original image size if needed
                m = masks_data[i]
                if m.shape != (h, w):
                    m = cv2.resize(
                        m.astype(np.float32), (w, h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                binary = (m > 0.5).astype(np.uint8) * 255
                combined = cv2.bitwise_or(combined, binary)
                confs.append(round(conf, 4))
                count += 1

        return SegmentationResult(
            mask=combined,
            instance_count=count,
            confidences=confs,
        )
