"""Modern v2 pipeline — PPE Compliance Monitor.

Uses:     YOLO detection for person + PPE items (helmet, vest, …)
Pipeline: YOLO detect → compliance check → zone monitoring → export

Delegates compliance logic to ``compliance.py``, zone tracking to
``zones.py``, visualisation to ``visualize.py``, and I/O to ``export.py``.
This file is the thin CVProject adapter that plugs into the repo's global
registry.
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

from compliance import ComplianceChecker, Detection
from config import PPEConfig, load_config, default_sample_config
from export import EventExporter
from visualize import OverlayRenderer
from zones import ZoneMonitor


@register("ppe_compliance_monitor")
class PPEComplianceModern(CVProject):
    project_type = "detection"
    description = "PPE compliance detection with zone monitoring and violation alerts"
    legacy_tech = "N/A (new project)"
    modern_tech = "YOLO detection + person-PPE association + zone compliance + CSV/JSON export"

    def __init__(self, config: PPEConfig | None = None) -> None:
        super().__init__()
        self._cfg = config or default_sample_config()
        self._checker = ComplianceChecker(
            required_ppe=self._cfg.required_ppe,
            person_class=self._cfg.person_class,
            ppe_iou_threshold=self._cfg.ppe_iou_threshold,
        )
        self._monitor = ZoneMonitor(self._cfg)
        self._renderer = OverlayRenderer()
        self._exporter = EventExporter(self._cfg)
        self.model = None
        self._frame_idx = 0

    # ── CVProject interface ────────────────────────────────

    def load(self) -> None:
        from models.registry import resolve
        weights, ver, fallback = resolve("ppe_compliance_monitor", "detect")
        self.model = load_yolo(weights)
        print(
            f"Using model for ppe_compliance_monitor: version={ver} "
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

        # Compliance evaluation
        compliance = self._checker.evaluate(detections)

        # Zone monitoring
        zone_statuses, alerts = self._monitor.process_frame(compliance)

        # Log events
        self._exporter.log_frame(self._frame_idx, zone_statuses, alerts, frame)
        self._frame_idx += 1

        return {
            "detections": detections,
            "compliance": compliance,
            "zone_statuses": zone_statuses,
            "alerts": alerts,
            "total_persons": compliance.total_persons,
            "compliant": compliance.compliant_count,
            "violations": compliance.violation_count,
            "_frame": frame,
        }

    def visualize(self, input_data, output) -> np.ndarray:
        frame = output.get("_frame")
        if frame is None:
            frame = input_data if isinstance(input_data, np.ndarray) else cv2.imread(str(input_data))
        return self._renderer.draw(frame, output["zone_statuses"], output["alerts"])

    # ── Project-specific API ───────────────────────────────

    def set_config(self, cfg: PPEConfig) -> None:
        """Hot-swap configuration."""
        self._cfg = cfg
        self._checker = ComplianceChecker(
            required_ppe=cfg.required_ppe,
            person_class=cfg.person_class,
            ppe_iou_threshold=cfg.ppe_iou_threshold,
        )
        self._monitor = ZoneMonitor(cfg)

    def export_events(self) -> None:
        """Flush accumulated events to disk."""
        self._exporter.flush()
