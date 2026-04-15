"""High-level pipeline for Number Plate Reader Pro.

Orchestrates detection → rectification → OCR → cleanup → dedup
into a single :class:`PlateReadResult`.

Usage::

    from parser import PlateReaderPipeline

    pipeline = PlateReaderPipeline(cfg)
    result = pipeline.process_frame(frame)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PlateConfig
from ocr_engine import OCREngine, OCRResult
from plate_cleaner import PlateCleaner
from plate_detector import PlateDetection, PlateDetector
from tracker import PlateTracker


@dataclass
class PlateRead:
    """A single plate read from one detection."""

    plate_text: str
    raw_text: str
    det_confidence: float
    ocr_confidence: float
    is_new: bool
    is_valid: bool
    box: tuple[int, int, int, int]
    crop: np.ndarray
    rectified: np.ndarray | None = None


@dataclass
class PlateReadResult:
    """Result from processing a single frame."""

    reads: list[PlateRead] = field(default_factory=list)
    num_detections: int = 0
    num_valid: int = 0
    num_new: int = 0
    frame_index: int = 0


class PlateReaderPipeline:
    """Full plate reading pipeline: detect -> OCR -> clean -> dedup."""

    def __init__(self, cfg: PlateConfig) -> None:
        self.cfg = cfg
        self._detector = PlateDetector(cfg)
        self._engine = OCREngine(cfg)
        self._cleaner = PlateCleaner(cfg)
        self._tracker = PlateTracker(cfg)
        self._frame_count = 0

    def process_frame(self, frame: np.ndarray) -> PlateReadResult:
        """Process a single frame through the full pipeline."""
        self._frame_count += 1
        detections = self._detector.detect(frame)

        reads: list[PlateRead] = []
        for det in detections:
            # OCR on rectified crop (or raw crop)
            ocr_input = det.rectified if det.rectified is not None else det.crop
            ocr_result = self._engine.read_plate(ocr_input)

            # Clean plate text
            plate_text = self._cleaner.clean(ocr_result.raw_text)
            is_valid = self._cleaner.is_valid(plate_text)

            # Dedup
            is_new = self._tracker.is_new(plate_text)

            reads.append(PlateRead(
                plate_text=plate_text,
                raw_text=ocr_result.raw_text,
                det_confidence=det.det_confidence,
                ocr_confidence=ocr_result.confidence,
                is_new=is_new,
                is_valid=is_valid,
                box=det.box,
                crop=det.crop,
                rectified=det.rectified,
            ))

        return PlateReadResult(
            reads=reads,
            num_detections=len(reads),
            num_valid=sum(1 for r in reads if r.is_valid),
            num_new=sum(1 for r in reads if r.is_new),
            frame_index=self._frame_count,
        )

    def reset_tracker(self) -> None:
        """Clear the duplicate suppression tracker."""
        self._tracker.reset()
