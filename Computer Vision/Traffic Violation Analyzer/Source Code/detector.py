"""Traffic Violation Analyzer — detection data structures.

Provides the shared :class:`Detection` dataclass used throughout the
project (tracker, rule engine, visualiser, exporter).

Usage::

    from detector import Detection
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Detection:
    """A single detection from the YOLO model."""

    box: tuple[int, int, int, int]    # (x1, y1, x2, y2)
    center: tuple[int, int]           # (cx, cy)
    class_name: str
    confidence: float
    class_id: int = -1
    track_id: int | None = None       # filled when tracking is active

    @property
    def area(self) -> int:
        return max(0, self.box[2] - self.box[0]) * max(0, self.box[3] - self.box[1])
