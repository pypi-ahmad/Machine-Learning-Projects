"""Layout detector — document block detection and region extraction.

Core detection module for the Document Layout Block Detector project.

Features
--------
* Detects document-layout blocks: titles, paragraphs, tables, figures,
  stamps, headers, footers, lists, captions, page numbers.
* Returns structured ``LayoutBlock`` / ``PageResult`` dataclasses.
* Optional region cropping for downstream OCR integration.

Usage::

    from detector import LayoutDetector
    from config import load_config

    cfg = load_config("layout_config.yaml")
    det = LayoutDetector(cfg)
    result = det.process(page_image)
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import LayoutConfig
from utils.yolo import load_yolo


@dataclass
class LayoutBlock:
    """Single detected document block."""

    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    area: int
    block_id: int = 0                # sequential ID within page


@dataclass
class PageResult:
    """Aggregated result for a single page image."""

    blocks: list[LayoutBlock] = field(default_factory=list)
    class_counts: dict[str, int] = field(default_factory=dict)
    total_blocks: int = 0
    page_idx: int = 0
    image_shape: tuple[int, int] = (0, 0)  # (height, width)


class LayoutDetector:
    """Detect document-layout blocks in page images."""

    def __init__(self, cfg: LayoutConfig) -> None:
        self.cfg = cfg
        self.model = load_yolo(cfg.model, device=cfg.device or None)
        self._target_lower: set[str] | None = None
        if cfg.target_classes:
            self._target_lower = {c.lower() for c in cfg.target_classes}

    def process(self, image: np.ndarray, *, page_idx: int = 0) -> PageResult:
        """Detect layout blocks in a page image.

        Parameters
        ----------
        image : np.ndarray
            BGR page image (OpenCV convention).
        page_idx : int
            Page index for multi-page documents.

        Returns
        -------
        PageResult
        """
        h, w = image.shape[:2]
        results = self.model(
            image,
            conf=self.cfg.conf_threshold,
            iou=self.cfg.iou_threshold,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device or None,
            verbose=False,
        )

        blocks: list[LayoutBlock] = []
        counts: Counter[str] = Counter()
        block_id = 0

        for det in results:
            boxes = det.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
                if self._target_lower and cls_name.lower() not in self._target_lower:
                    continue

                area = (x2 - x1) * (y2 - y1)
                blocks.append(LayoutBlock(
                    class_name=cls_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    area=area,
                    block_id=block_id,
                ))
                counts[cls_name] += 1
                block_id += 1

        # Sort blocks top-to-bottom, left-to-right (reading order)
        blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
        for i, blk in enumerate(blocks):
            blk.block_id = i

        return PageResult(
            blocks=blocks,
            class_counts=dict(counts),
            total_blocks=len(blocks),
            page_idx=page_idx,
            image_shape=(h, w),
        )

    def crop_blocks(self, image: np.ndarray, result: PageResult) -> list[np.ndarray]:
        """Extract cropped regions for each detected block.

        Returns a list of BGR images in the same order as
        ``result.blocks``.
        """
        crops = []
        for blk in result.blocks:
            x1, y1, x2, y2 = blk.bbox
            crop = image[max(0, y1):y2, max(0, x1):x2].copy()
            crops.append(crop)
        return crops
