"""OBB detector — oriented bounding-box detection for aerial imagery.

Core detection module for the Drone Ship OBB Detector project.

Features
--------
* Runs YOLO-OBB inference and parses rotated bounding-box outputs
  (4 corner points + class + confidence).
* Per-class counting.
* Lightweight ``OBBDetection`` / ``FrameResult`` dataclasses for
  downstream visualisation and export.

Usage::

    from detector import OBBDetector
    from config import load_config

    cfg = load_config("obb_config.yaml")
    det = OBBDetector(cfg)
    result = det.process(frame)
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OBBConfig
from utils.yolo import load_yolo


@dataclass
class OBBDetection:
    """Single oriented bounding-box detection."""

    class_name: str
    confidence: float
    corners: np.ndarray          # shape (4, 2) — four corner points in pixel coords
    centre: tuple[int, int]      # centroid
    angle_deg: float             # rotation angle in degrees


@dataclass
class FrameResult:
    """Aggregated result for a single frame."""

    detections: list[OBBDetection] = field(default_factory=list)
    class_counts: dict[str, int] = field(default_factory=dict)
    total: int = 0
    frame_idx: int = 0


class OBBDetector:
    """Run YOLO-OBB inference on aerial imagery."""

    def __init__(self, cfg: OBBConfig) -> None:
        self.cfg = cfg
        self.model = load_yolo(cfg.model, device=cfg.device or None)
        self._target_lower: set[str] | None = None
        if cfg.target_classes:
            self._target_lower = {c.lower() for c in cfg.target_classes}

    def process(self, frame: np.ndarray, *, frame_idx: int = 0) -> FrameResult:
        """Detect oriented bounding boxes in *frame*.

        Parameters
        ----------
        frame : np.ndarray
            BGR image.
        frame_idx : int
            Monotonic frame counter for export.

        Returns
        -------
        FrameResult
        """
        results = self.model(
            frame,
            conf=self.cfg.conf_threshold,
            iou=self.cfg.iou_threshold,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device or None,
            verbose=False,
        )

        detections: list[OBBDetection] = []
        counts: Counter[str] = Counter()

        for det in results:
            obb = det.obb
            if obb is None or len(obb) == 0:
                continue

            # obb.xyxyxyxy: (N, 4, 2) — four corners per box
            # obb.cls: (N,)   obb.conf: (N,)
            corners_all = obb.xyxyxyxy.cpu().numpy()   # (N, 4, 2)
            cls_ids = obb.cls.cpu().numpy().astype(int)
            confs = obb.conf.cpu().numpy()

            for i in range(len(cls_ids)):
                cls_name = self.model.names.get(cls_ids[i], f"class_{cls_ids[i]}")

                # Filter by target classes
                if self._target_lower and cls_name.lower() not in self._target_lower:
                    continue

                corners = corners_all[i]  # (4, 2)
                cx = int(corners[:, 0].mean())
                cy = int(corners[:, 1].mean())
                angle = _compute_angle(corners)

                detections.append(OBBDetection(
                    class_name=cls_name,
                    confidence=float(confs[i]),
                    corners=corners,
                    centre=(cx, cy),
                    angle_deg=angle,
                ))
                counts[cls_name] += 1

        return FrameResult(
            detections=detections,
            class_counts=dict(counts),
            total=len(detections),
            frame_idx=frame_idx,
        )


def _compute_angle(corners: np.ndarray) -> float:
    """Compute the orientation angle (degrees) from the first edge of the OBB.

    Takes the vector from corner[0] to corner[1] and returns the
    angle w.r.t. the positive x-axis, in [-90, 90).
    """
    dx = corners[1][0] - corners[0][0]
    dy = corners[1][1] - corners[0][1]
    angle = np.degrees(np.arctan2(dy, dx))
    # Normalise to [-90, 90)
    while angle >= 90:
        angle -= 180
    while angle < -90:
        angle += 180
    return round(float(angle), 1)
