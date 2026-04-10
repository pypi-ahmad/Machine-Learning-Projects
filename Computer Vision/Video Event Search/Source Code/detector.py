"""Video Event Search — detection dataclass.

Provides :class:`Detection` used throughout the pipeline (tracker,
event generator, visualiser, exporter).
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
    track_id: int | None = None

    @property
    def area(self) -> int:
        return max(0, self.box[2] - self.box[0]) * max(0, self.box[3] - self.box[1])
