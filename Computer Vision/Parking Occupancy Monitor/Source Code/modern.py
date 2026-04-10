"""Modern v2 pipeline — Parking Occupancy Monitor.

Uses:     YOLO26m detection for vehicles (car, truck, bus, motorcycle, …)
Pipeline: YOLO detect → slot evaluation → overlay → export

Delegates slot logic to ``slots.py``, visualisation to ``visualize.py``,
and I/O to ``export.py``.  This file is the thin CVProject adapter that
plugs into the repo's global registry.
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

from config import ParkingConfig, load_config, default_sample_config
from export import EventExporter
from slots import SlotEvaluator, Detection
from visualize import OverlayRenderer


@register("parking_occupancy_monitor")
class ParkingOccupancyModern(CVProject):
    project_type = "detection"
    description = "Parking lot vehicle detection with per-slot occupancy monitoring"
    legacy_tech = "N/A (new project)"
    modern_tech = "YOLO26m detection + slot polygon evaluation + CSV/JSON export"

    def __init__(self, config: ParkingConfig | None = None) -> None:
        super().__init__()
        self._cfg = config or default_sample_config()
        self._evaluator = SlotEvaluator(self._cfg)
        self._renderer = OverlayRenderer()
        self._exporter = EventExporter(self._cfg)
        self.model = None
        self._frame_idx = 0

    # ── CVProject interface ────────────────────────────────

    def load(self) -> None:
        from models.registry import resolve
        weights, ver, fallback = resolve("parking_occupancy_monitor", "detect")
        self.model = load_yolo(weights)
        print(
            f"Using model for parking_occupancy_monitor: version={ver} "
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

        # Slot evaluation
        frame_result = self._evaluator.evaluate(detections)

        # Log events
        self._exporter.log_frame(self._frame_idx, frame_result)
        self._frame_idx += 1

        return {
            "detections": detections,
            "frame_result": frame_result,
            "total_slots": frame_result.total_slots,
            "occupied": frame_result.occupied_count,
            "free": frame_result.free_count,
            "slot_statuses": frame_result.slot_statuses,
            "_frame": frame,
        }

    def visualize(self, input_data, output) -> np.ndarray:
        frame = output.get("_frame")
        if frame is None:
            frame = input_data if isinstance(input_data, np.ndarray) else cv2.imread(str(input_data))
        return self._renderer.draw(frame, output["frame_result"])

    # ── Project-specific API ───────────────────────────────

    def set_config(self, cfg: ParkingConfig) -> None:
        """Hot-swap configuration (slots, thresholds, export settings)."""
        self._cfg = cfg
        self._evaluator = SlotEvaluator(cfg)

    def set_slots(self, slots: list[dict]) -> None:
        """Set slot polygons from raw dicts.

        Each slot: ``{"name": "A1", "polygon": [(x1,y1), ...]}``
        """
        from config import SlotConfig
        slot_configs = [
            SlotConfig(
                name=s["name"],
                polygon=[tuple(pt) for pt in s["polygon"]],
            )
            for s in slots
        ]
        self._cfg.slots = slot_configs
        self._evaluator = SlotEvaluator(self._cfg)

    def export_events(self) -> None:
        """Flush accumulated events to disk."""
        self._exporter.flush()
