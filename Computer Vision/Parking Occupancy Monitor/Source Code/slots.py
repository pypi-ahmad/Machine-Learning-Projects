"""Parking Occupancy Monitor — detection helpers & slot evaluator.

Provides the shared :class:`Detection` dataclass and the
:class:`SlotEvaluator` which checks whether each parking slot polygon
contains a detected vehicle.

Usage::

    from slots import SlotEvaluator, Detection

    evaluator = SlotEvaluator(cfg)
    result = evaluator.evaluate(detections)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from config import ParkingConfig, SlotConfig


# ---------------------------------------------------------------------------
# Detection data class (shared with visualize / export)
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    """A single detection from the YOLO model."""

    box: tuple[int, int, int, int]    # (x1, y1, x2, y2)
    center: tuple[int, int]           # (cx, cy)
    class_name: str
    confidence: float
    class_id: int = -1

    @property
    def area(self) -> int:
        return max(0, self.box[2] - self.box[0]) * max(0, self.box[3] - self.box[1])


# ---------------------------------------------------------------------------
# Slot status
# ---------------------------------------------------------------------------
@dataclass
class SlotStatus:
    """Occupancy status of a single parking slot."""

    name: str
    polygon: list[tuple[int, int]]
    occupied: bool = False
    vehicle: Detection | None = None    # the vehicle that claims this slot


@dataclass
class FrameResult:
    """Aggregate occupancy result for one frame."""

    slot_statuses: list[SlotStatus] = field(default_factory=list)
    total_slots: int = 0
    occupied_count: int = 0
    free_count: int = 0
    vehicle_detections: list[Detection] = field(default_factory=list)
    other_detections: list[Detection] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def point_in_polygon(
    point: tuple[int, int],
    polygon: Sequence[tuple[int, int]],
) -> bool:
    """Ray-casting algorithm — True if *point* is inside *polygon*."""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _polygon_bbox(polygon: Sequence[tuple[int, int]]) -> tuple[int, int, int, int]:
    """Axis-aligned bounding box of a polygon → (x1, y1, x2, y2)."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection-over-union for two (x1, y1, x2, y2) boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / (area_a + area_b - inter)


# ---------------------------------------------------------------------------
# Slot evaluator
# ---------------------------------------------------------------------------

class SlotEvaluator:
    """Determine which parking slots are occupied by detected vehicles."""

    def __init__(self, cfg: ParkingConfig) -> None:
        self.cfg = cfg
        self._slot_bboxes = [_polygon_bbox(s.polygon) for s in cfg.slots]

    def evaluate(self, detections: list[Detection]) -> FrameResult:
        """Check every slot against detected vehicles.

        A slot is marked *occupied* when:
        1. A vehicle's center falls inside the slot polygon, **or**
        2. The vehicle box overlaps the slot bounding-box with IoU ≥
           ``occupancy_iou_threshold``.

        Each slot is assigned the highest-confidence matching vehicle.
        """
        vehicle_classes = set(self.cfg.vehicle_classes)
        vehicles: list[Detection] = []
        others: list[Detection] = []

        for det in detections:
            if det.class_name in vehicle_classes:
                vehicles.append(det)
            else:
                others.append(det)

        statuses: list[SlotStatus] = []

        for idx, slot_cfg in enumerate(self.cfg.slots):
            best_vehicle: Detection | None = None
            best_score: float = 0.0
            slot_bbox = self._slot_bboxes[idx]

            for v in vehicles:
                # Check 1: center-in-polygon
                if point_in_polygon(v.center, slot_cfg.polygon):
                    if v.confidence > best_score:
                        best_score = v.confidence
                        best_vehicle = v
                    continue

                # Check 2: IoU overlap
                iou = _box_iou(v.box, slot_bbox)
                if iou >= self.cfg.occupancy_iou_threshold and v.confidence > best_score:
                    best_score = v.confidence
                    best_vehicle = v

            statuses.append(SlotStatus(
                name=slot_cfg.name,
                polygon=slot_cfg.polygon,
                occupied=best_vehicle is not None,
                vehicle=best_vehicle,
            ))

        occupied = sum(1 for s in statuses if s.occupied)
        return FrameResult(
            slot_statuses=statuses,
            total_slots=len(statuses),
            occupied_count=occupied,
            free_count=len(statuses) - occupied,
            vehicle_detections=vehicles,
            other_detections=others,
        )
