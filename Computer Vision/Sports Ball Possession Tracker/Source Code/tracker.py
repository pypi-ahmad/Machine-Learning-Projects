"""Tracker — detection + multi-object tracking for players and the ball.

Wraps YOLO ``model.track()`` to produce per-frame tracked detections
with stable IDs across frames.

Usage::

    from tracker import SportTracker
    from config import load_config

    cfg = load_config("possession_config.yaml")
    tracker = SportTracker(cfg)
    frame_dets = tracker.update(frame, frame_idx=0)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PossessionConfig
from utils.yolo import load_yolo


@dataclass
class Detection:
    """Single tracked detection."""

    track_id: int
    class_name: str
    class_id: int
    confidence: float
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2)
    centre: tuple[int, int]


@dataclass
class FrameDetections:
    """All tracked detections for one frame."""

    players: list[Detection] = field(default_factory=list)
    balls: list[Detection] = field(default_factory=list)
    other: list[Detection] = field(default_factory=list)
    frame_idx: int = 0


class SportTracker:
    """YOLO detect-and-track wrapper for sports footage."""

    def __init__(self, cfg: PossessionConfig) -> None:
        self.cfg = cfg
        self.model = load_yolo(cfg.model, device=cfg.device or None)
        self._player_ids, self._ball_ids = self._resolve_class_ids()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray, *, frame_idx: int = 0) -> FrameDetections:
        """Run detection + tracking on a single frame.

        Returns
        -------
        FrameDetections
            Players, balls, and other objects with stable track IDs.
        """
        results = self.model.track(
            frame,
            conf=self.cfg.conf_threshold,
            iou=self.cfg.iou_threshold,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device or None,
            persist=True,
            tracker=f"{self.cfg.tracker_type}.yaml",
            verbose=False,
        )

        players: list[Detection] = []
        balls: list[Detection] = []
        other: list[Detection] = []

        for det in results:
            boxes = det.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                tid = int(box.id[0]) if box.id is not None else -1
                cls_name = self.model.names.get(cls_id, f"class_{cls_id}")

                d = Detection(
                    track_id=tid,
                    class_name=cls_name,
                    class_id=cls_id,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    centre=(cx, cy),
                )

                if cls_id in self._ball_ids:
                    if conf >= self.cfg.min_ball_conf:
                        balls.append(d)
                elif cls_id in self._player_ids:
                    players.append(d)
                else:
                    other.append(d)

        return FrameDetections(
            players=players,
            balls=balls,
            other=other,
            frame_idx=frame_idx,
        )

    # ------------------------------------------------------------------
    # Class-ID resolution
    # ------------------------------------------------------------------

    def _resolve_class_ids(self) -> tuple[set[int], set[int]]:
        """Map config class IDs or auto-detect from model names."""
        names = self.model.names  # {int: str}

        player_ids: set[int] = set()
        ball_ids: set[int] = set()

        if self.cfg.player_class_id >= 0:
            player_ids.add(self.cfg.player_class_id)
        if self.cfg.ball_class_id >= 0:
            ball_ids.add(self.cfg.ball_class_id)

        # Auto-detect from common class names
        player_keywords = {"person", "player", "goalkeeper", "referee"}
        ball_keywords = {"ball", "sports ball", "football", "soccer ball",
                         "basketball", "tennis ball"}

        for cid, cname in names.items():
            lower = cname.lower().strip()
            if not player_ids and lower in player_keywords:
                player_ids.add(cid)
            if not ball_ids and lower in ball_keywords:
                ball_ids.add(cid)

        # Fallback: COCO class 0 = person, 32 = sports ball
        if not player_ids:
            if 0 in names:
                player_ids.add(0)
        if not ball_ids:
            if 32 in names:
                ball_ids.add(32)

        return player_ids, ball_ids
