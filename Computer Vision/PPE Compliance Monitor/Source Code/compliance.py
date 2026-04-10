"""PPE Compliance Monitor — compliance rule engine.

Associates detected PPE items with the nearest person using bounding-box
overlap, then evaluates each person's compliance against the required PPE
list for their zone.

Usage::

    from compliance import ComplianceChecker, Detection

    checker = ComplianceChecker(required_ppe=["helmet", "safety_vest"],
                                ppe_iou_threshold=0.20)
    results = checker.evaluate(detections)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


# ---------------------------------------------------------------------------
# Detection data class (shared with zones / visualize)
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    """A single detection from the YOLO model."""

    box: tuple[int, int, int, int]         # (x1, y1, x2, y2)
    center: tuple[int, int]                # (cx, cy)
    class_name: str
    confidence: float
    class_id: int = -1

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)


@dataclass
class PersonCompliance:
    """Compliance result for one person."""

    person: Detection
    ppe_items: dict[str, Detection]       # class_name → Detection
    missing_items: list[str]
    is_compliant: bool
    zone_name: str = ""                   # filled by zone logic


@dataclass
class FrameCompliance:
    """Aggregate compliance result for one frame."""

    persons: list[PersonCompliance]
    total_persons: int = 0
    compliant_count: int = 0
    violation_count: int = 0
    non_person_detections: list[Detection] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Association helpers
# ---------------------------------------------------------------------------

def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
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


def _containment(child: tuple[int, int, int, int],
                 parent: tuple[int, int, int, int]) -> float:
    """Fraction of *child* box area that falls inside *parent* box."""
    x1 = max(child[0], parent[0])
    y1 = max(child[1], parent[1])
    x2 = min(child[2], parent[2])
    y2 = min(child[3], parent[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    child_area = max(1, (child[2] - child[0]) * (child[3] - child[1]))
    return inter / child_area


# ---------------------------------------------------------------------------
# Compliance checker
# ---------------------------------------------------------------------------

class ComplianceChecker:
    """Evaluate PPE compliance for detected persons."""

    def __init__(
        self,
        required_ppe: Sequence[str],
        person_class: str = "person",
        ppe_iou_threshold: float = 0.20,
    ) -> None:
        self.required_ppe = list(required_ppe)
        self.person_class = person_class
        self.ppe_iou_threshold = ppe_iou_threshold

    # ---- public API --------------------------------------------------------

    def evaluate(self, detections: list[Detection]) -> FrameCompliance:
        """Run compliance evaluation on a set of detections.

        * Associate each PPE item with its best-matching person using
          containment (PPE typically *inside* the person box).
        * Check each person against the required PPE list.
        """
        persons: list[Detection] = []
        ppe_items: list[Detection] = []

        for det in detections:
            if det.class_name == self.person_class:
                persons.append(det)
            else:
                ppe_items.append(det)

        # Greedy association: each PPE item → best-overlapping person
        person_ppe: dict[int, dict[str, Detection]] = {
            i: {} for i in range(len(persons))
        }

        for ppe in ppe_items:
            best_idx = -1
            best_score = self.ppe_iou_threshold
            for idx, person in enumerate(persons):
                score = _containment(ppe.box, person.box)
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx >= 0:
                cls = ppe.class_name
                # Keep highest-confidence PPE per class
                existing = person_ppe[best_idx].get(cls)
                if existing is None or ppe.confidence > existing.confidence:
                    person_ppe[best_idx][cls] = ppe

        # Build compliance results
        results: list[PersonCompliance] = []
        for idx, person in enumerate(persons):
            assigned = person_ppe[idx]
            missing = [r for r in self.required_ppe if r not in assigned]
            results.append(PersonCompliance(
                person=person,
                ppe_items=assigned,
                missing_items=missing,
                is_compliant=len(missing) == 0,
            ))

        compliant = sum(1 for r in results if r.is_compliant)
        return FrameCompliance(
            persons=results,
            total_persons=len(results),
            compliant_count=compliant,
            violation_count=len(results) - compliant,
            non_person_detections=ppe_items,
        )
