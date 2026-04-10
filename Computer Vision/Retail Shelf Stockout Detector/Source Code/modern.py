"""Modern v2 pipeline — Retail Shelf Stockout Detector.

Uses:     YOLO26m detection for shelf product objects
Pipeline: YOLO detect → per-zone counting → low-stock alerts → export

Delegates zone logic to ``zones.py``, visualisation to ``visualize.py``,
and I/O to ``export.py``.  This file is the thin CVProject adapter that
plugs into the repo's global registry.
"""

import sys
from pathlib import Path

# Add project root for local module imports
_PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_DIR))
sys.path.insert(0, str(_PROJECT_DIR.parents[1]))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register
from utils.yolo import load_yolo

from config import StockoutConfig, ZoneConfig, load_config, default_sample_config
from export import EventExporter
from visualize import OverlayRenderer
from zones import Detection, ZoneCounter


@register("retail_shelf_stockout")
class RetailShelfStockoutModern(CVProject):
    project_type = "detection"
    description = "Retail shelf product detection with zone counting and low-stock alerts"
    legacy_tech = "N/A (new project)"
    modern_tech = "YOLO26m detection + zone polygons + CSV/JSON event export"

    def __init__(self, config: StockoutConfig | None = None) -> None:
        super().__init__()
        self._cfg = config or default_sample_config()
        self._counter = ZoneCounter(
            self._cfg.zones,
            default_threshold=self._cfg.default_low_stock_threshold,
        )
        self._renderer = OverlayRenderer(self._counter)
        self._exporter = EventExporter(
            output_dir=self._cfg.export_dir,
            save_csv=self._cfg.save_events_csv,
            save_json=self._cfg.save_events_json,
            save_snapshots=self._cfg.save_alert_snapshots,
            snapshot_cooldown=self._cfg.snapshot_cooldown_sec,
        )
        self.model = None

    # ── CVProject interface ────────────────────────────────

    def load(self) -> None:
        from models.registry import resolve
        weights, ver, fallback = resolve("retail_shelf_stockout", "detect")
        self.model = load_yolo(weights)
        print(
            f"Using model for retail_shelf_stockout: version={ver} "
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

        # Zone counting
        frame_result = self._counter.update(detections)

        # Log events
        self._exporter.log_frame(frame, frame_result)

        return {
            "detections": detections,
            "total_count": frame_result.total_count,
            "zone_statuses": frame_result.zone_statuses,
            "zone_counts": {zs.name: zs.count for zs in frame_result.zone_statuses},
            "alerts": frame_result.alerts,
            "_frame_result": frame_result,
            "_frame": frame,
        }

    def visualize(self, input_data, output) -> np.ndarray:
        frame = output.get("_frame")
        if frame is None:
            frame = input_data if isinstance(input_data, np.ndarray) else cv2.imread(str(input_data))
        return self._renderer.draw(frame, output["_frame_result"])

    # ── Project-specific API ───────────────────────────────

    def set_config(self, cfg: StockoutConfig) -> None:
        """Hot-swap configuration (zones, thresholds, export settings)."""
        self._cfg = cfg
        self._counter = ZoneCounter(cfg.zones, default_threshold=cfg.default_low_stock_threshold)
        self._renderer = OverlayRenderer(self._counter)

    def set_zones(self, zones: list[dict]) -> None:
        """Set shelf-zone polygons from raw dicts (backward-compatible).

        Each zone: ``{"name": "Aisle-1", "polygon": [(x1,y1), ...], "low_stock_threshold": 3}``
        """
        zone_configs = [
            ZoneConfig(
                name=z["name"],
                polygon=[tuple(pt) for pt in z["polygon"]],
                low_stock_threshold=z.get("low_stock_threshold", self._cfg.default_low_stock_threshold),
                classes=z.get("classes"),
            )
            for z in zones
        ]
        self._cfg.zones = zone_configs
        self._counter = ZoneCounter(zone_configs, default_threshold=self._cfg.default_low_stock_threshold)
        self._renderer = OverlayRenderer(self._counter)

    def export_events(self) -> dict[str, str]:
        """Flush accumulated events to disk and return file paths."""
        return self._exporter.flush()

    def export_events_csv(self, path: str) -> None:
        """Export events to a specific CSV path (legacy compat)."""
        import csv
        events = self._exporter._events
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "zone", "count", "threshold", "event_type"])
            writer.writeheader()
            writer.writerows(events)

    def export_snapshot(self, frame: np.ndarray, output: dict, path: str) -> None:
        """Save annotated snapshot as image."""
        vis = self.visualize(frame, output)
        cv2.imwrite(path, vis)
