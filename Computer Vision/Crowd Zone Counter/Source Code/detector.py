"""Person detector — YOLO-based person detection for crowd counting.

Wraps YOLO inference and filters to "person" class only.
Returns lightweight ``PersonDetection`` / ``FrameDetections`` dataclasses.

Usage::

    from detector import PersonDetector
    from config import load_config

    cfg = load_config("crowd_config.yaml")
    det = PersonDetector(cfg)
    dets = det.detect(frame)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CrowdConfig
from utils.yolo import load_yolo


@dataclass
class PersonDetection:
    """Single detected person."""

    confidence: float
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2)
    centre: tuple[int, int]           # foot-centre used for zone test
    foot_point: tuple[int, int]       # bottom-centre of bbox


@dataclass
class FrameDetections:
    """All person detections for one frame."""

    persons: list[PersonDetection] = field(default_factory=list)
    total: int = 0
    frame_idx: int = 0


class PersonDetector:
    """YOLO person detector filtered to a single class."""

    def __init__(self, cfg: CrowdConfig) -> None:
        self.cfg = cfg
        self.model = load_yolo(cfg.model, device=cfg.device or None)

    def detect(self, frame: np.ndarray, *, frame_idx: int = 0) -> FrameDetections:
        """Run person detection on *frame*.

        The *foot_point* (bottom-centre of bbox) is used for zone
        assignment because it better represents where a person is
        standing than the bbox centre.
        """
        results = self.model(
            frame,
            conf=self.cfg.conf_threshold,
            iou=self.cfg.iou_threshold,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device or None,
            classes=[self.cfg.person_class_id],
            verbose=False,
        )

        persons: list[PersonDetection] = []

        for det in results:
            boxes = det.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                foot_x, foot_y = cx, y2  # bottom-centre

                persons.append(PersonDetection(
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    centre=(cx, cy),
                    foot_point=(foot_x, foot_y),
                ))

        return FrameDetections(
            persons=persons,
            total=len(persons),
            frame_idx=frame_idx,
        )
