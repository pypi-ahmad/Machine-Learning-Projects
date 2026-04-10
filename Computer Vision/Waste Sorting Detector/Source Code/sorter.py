"""Waste sorter — per-class counting and bin-zone validation.

Core detection module for the Waste Sorting Detector project.

Features
--------
* Per-class counting of detected waste items (plastic, paper, cardboard,
  metal, glass, trash).
* Optional **bin-zone validation**: checks whether each detection centre
  falls inside the correct zone polygon.  Misplaced items are flagged.
* Lightweight dataclass ``WasteDetection`` carries results for downstream
  visualisation and export.

Usage::

    from sorter import WasteSorter
    from config import load_config

    cfg = load_config("waste.yaml")
    sorter = WasteSorter(cfg)
    frame_result = sorter.process(frame)
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# Repo root for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import BinZoneConfig, WasteConfig
from utils.yolo import load_yolo


# ---------------------------------------------------------------------------
# Data carriers
# ---------------------------------------------------------------------------

@dataclass
class WasteItem:
    """Single detected waste item."""

    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    centre: tuple[int, int]
    zone_name: str | None = None
    misplaced: bool = False


@dataclass
class FrameResult:
    """Aggregated result for a single frame."""

    items: list[WasteItem] = field(default_factory=list)
    class_counts: dict[str, int] = field(default_factory=dict)
    misplaced_items: list[WasteItem] = field(default_factory=list)
    total_items: int = 0
    frame_idx: int = 0


# ---------------------------------------------------------------------------
# Sorter
# ---------------------------------------------------------------------------

class WasteSorter:
    """Detect waste items and optionally validate bin-zone placement."""

    def __init__(self, cfg: WasteConfig) -> None:
        self.cfg = cfg
        self.model = load_yolo(cfg.model)
        self._zone_polys: list[tuple[BinZoneConfig, np.ndarray]] | None = None
        if cfg.bin_zones:
            self._zone_polys = [
                (z, np.array(z.polygon, dtype=np.int32)) for z in cfg.bin_zones
            ]

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def process(self, frame: np.ndarray, *, frame_idx: int = 0) -> FrameResult:
        """Run detection + classification on a single frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (OpenCV convention).
        frame_idx : int
            Monotonically increasing frame counter (for export).

        Returns
        -------
        FrameResult
        """
        results = self.model(
            frame,
            conf=self.cfg.conf_threshold,
            iou=self.cfg.iou_threshold,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
            verbose=False,
        )

        items: list[WasteItem] = []
        class_counts: Counter[str] = Counter()
        misplaced: list[WasteItem] = []

        for det in results:
            boxes = det.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                class_name = self.model.names.get(cls_id, f"class_{cls_id}")
                # Filter to configured waste classes (case-insensitive)
                if self.cfg.waste_classes:
                    lower_classes = {c.lower() for c in self.cfg.waste_classes}
                    if class_name.lower() not in lower_classes:
                        continue

                item = WasteItem(
                    class_name=class_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    centre=(cx, cy),
                )

                # Bin-zone validation
                if self._zone_polys:
                    self._validate_zone(item)
                    if item.misplaced:
                        misplaced.append(item)

                items.append(item)
                class_counts[class_name] += 1

        return FrameResult(
            items=items,
            class_counts=dict(class_counts),
            misplaced_items=misplaced,
            total_items=len(items),
            frame_idx=frame_idx,
        )

    # -----------------------------------------------------------------
    # Bin-zone helpers
    # -----------------------------------------------------------------

    def _validate_zone(self, item: WasteItem) -> None:
        """Check if item centre falls in an appropriate bin zone."""
        import cv2

        cx, cy = item.centre
        for zone_cfg, poly in self._zone_polys:  # type: ignore[union-attr]
            inside = cv2.pointPolygonTest(poly, (float(cx), float(cy)), False)
            if inside >= 0:
                item.zone_name = zone_cfg.name
                accepted = {c.lower() for c in zone_cfg.accepted_classes}
                if item.class_name.lower() not in accepted:
                    item.misplaced = True
                return

        # Not inside any zone — mark zone as "unzoned"
        item.zone_name = None
