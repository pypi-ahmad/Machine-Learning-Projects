"""Conveyor Part Defect Detector — detection & inspection logic.

Provides the shared :class:`Detection` dataclass and the
:class:`Inspector` which evaluates a frame's detections into a
pass/fail verdict with per-defect details.

Usage::

    from inspector import Inspector, Detection

    inspector = Inspector(cfg)
    result = inspector.evaluate(detections)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import InspectionConfig


# ---------------------------------------------------------------------------
# Detection data class
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
# Frame result
# ---------------------------------------------------------------------------
@dataclass
class FrameResult:
    """Inspection result for a single frame / image."""

    passed: bool = True
    defect_count: int = 0
    defects: list[Detection] = field(default_factory=list)
    all_detections: list[Detection] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return "PASS" if self.passed else "FAIL"


# ---------------------------------------------------------------------------
# Inspector
# ---------------------------------------------------------------------------
class Inspector:
    """Evaluate detections against defect rules and produce pass/fail."""

    def __init__(self, cfg: InspectionConfig) -> None:
        self.cfg = cfg
        self._defect_set = set(cfg.defect_classes)

    def evaluate(self, detections: list[Detection]) -> FrameResult:
        """Classify detections into defects vs. non-defects and determine
        the pass/fail verdict.

        If ``all_classes_are_defects`` is ``True`` in the config, every
        detection counts as a defect regardless of its class name.
        """
        defects: list[Detection] = []

        for det in detections:
            if self.cfg.all_classes_are_defects:
                defects.append(det)
            elif det.class_name in self._defect_set:
                defects.append(det)

        passed = len(defects) < self.cfg.fail_threshold
        return FrameResult(
            passed=passed,
            defect_count=len(defects),
            defects=defects,
            all_detections=detections,
        )
