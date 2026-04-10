"""Modern v2 pipeline — Conveyor Part Defect Detector.

Uses:     YOLO26m detection for industrial defect classes
Pipeline: YOLO detect → inspector (pass/fail) → overlay → export

Delegates inspection logic to ``inspector.py``, visualisation to
``visualize.py``, and I/O to ``export.py``.  This file is the thin
CVProject adapter that plugs into the repo's global registry.
"""

import sys
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_DIR))
sys.path.insert(0, str(_PROJECT_DIR.parents[1]))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register
from utils.yolo import load_yolo

from config import InspectionConfig, load_config, default_sample_config
from export import EventExporter
from inspector import Inspector, Detection
from visualize import OverlayRenderer


@register("conveyor_part_defect_detector")
class ConveyorDefectModern(CVProject):
    project_type = "detection"
    description = "Industrial conveyor-belt defect detection with pass/fail verdict and defect crops"
    legacy_tech = "N/A (new project)"
    modern_tech = "YOLO26m detection + pass/fail inspector + defect crop export"

    def __init__(self, config: InspectionConfig | None = None) -> None:
        super().__init__()
        self._cfg = config or default_sample_config()
        self._inspector = Inspector(self._cfg)
        self._renderer = OverlayRenderer()
        self._exporter = EventExporter(self._cfg)
        self.model = None
        self._frame_idx = 0

    # ── CVProject interface ────────────────────────────────

    def load(self) -> None:
        from models.registry import resolve
        weights, ver, fallback = resolve("conveyor_part_defect_detector", "detect")
        self.model = load_yolo(weights)
        print(
            f"Using model for conveyor_part_defect_detector: version={ver} "
            f"weights={weights} pretrained_fallback={fallback}"
        )

    def predict(self, input_data) -> dict:
        frame = input_data if isinstance(input_data, np.ndarray) else cv2.imread(str(input_data))
        results = self.model(
            frame, verbose=False,
            conf=self._cfg.conf_threshold,
            iou=self._cfg.iou_threshold,
        )

        # Parse YOLO boxes into Detection objects
        detections: list[Detection] = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cls_id = int(box.cls[0])
            cls_name = self.model.names.get(cls_id, str(cls_id))
            detections.append(Detection(
                box=(x1, y1, x2, y2),
                center=(cx, cy),
                class_name=cls_name,
                confidence=float(box.conf[0]),
                class_id=cls_id,
            ))

        # Inspection
        result = self._inspector.evaluate(detections)

        # Log
        self._exporter.log_frame(self._frame_idx, result, frame)
        self._frame_idx += 1

        return {
            "detections": detections,
            "result": result,
            "verdict": result.verdict,
            "defect_count": result.defect_count,
            "passed": result.passed,
            "_frame": frame,
        }

    def visualize(self, input_data, output) -> np.ndarray:
        frame = output.get("_frame")
        if frame is None:
            frame = input_data if isinstance(input_data, np.ndarray) else cv2.imread(str(input_data))
        return self._renderer.draw(frame, output["result"])

    # ── Project-specific API ───────────────────────────────

    def set_config(self, cfg: InspectionConfig) -> None:
        """Hot-swap configuration."""
        self._cfg = cfg
        self._inspector = Inspector(cfg)

    def export_events(self) -> None:
        """Flush accumulated events to disk."""
        self._exporter.flush()
