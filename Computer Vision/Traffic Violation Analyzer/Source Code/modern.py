"""Modern v2 pipeline — Traffic Violation Analyzer.

Uses:     YOLO26m detection + built-in ByteTrack/BoT-SORT tracker
Pipeline: YOLO detect+track → rule engine (line crossing / wrong-way) → export

Delegates tracking to ``tracker.py``, rule evaluation to ``rules.py``,
visualisation to ``visualize.py``, and I/O to ``export.py``.
This file is the thin CVProject adapter that plugs into the repo's
global registry.
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

from config import TrafficConfig, load_config, default_sample_config
from detector import Detection
from export import EventExporter
from rules import RuleEngine
from tracker import TrackManager
from visualize import OverlayRenderer


@register("traffic_violation_analyzer")
class TrafficViolationModern(CVProject):
    project_type = "tracking"
    description = "Traffic violation detection with line crossing counts and wrong-way alerts"
    legacy_tech = "N/A (new project)"
    modern_tech = "YOLO26m detection + ByteTrack + rule engine + CSV/JSON export"

    def __init__(self, config: TrafficConfig | None = None) -> None:
        super().__init__()
        self._cfg = config or default_sample_config()
        self._tm = TrackManager(max_history=60)
        self._engine = RuleEngine(self._cfg)
        self._renderer = OverlayRenderer()
        self._exporter = EventExporter(self._cfg)
        self.model = None
        self._frame_idx = 0

    # ── CVProject interface ────────────────────────────────

    def load(self) -> None:
        from models.registry import resolve
        weights, ver, fallback = resolve("traffic_violation_analyzer", "detect")
        self.model = load_yolo(weights)
        print(
            f"Using model for traffic_violation_analyzer: version={ver} "
            f"weights={weights} pretrained_fallback={fallback}"
        )

    def predict(self, input_data) -> dict:
        frame = input_data if isinstance(input_data, np.ndarray) else cv2.imread(str(input_data))

        # Run tracking
        try:
            results = self.model.track(
                frame, persist=True, verbose=False,
                conf=self._cfg.conf_threshold,
                iou=self._cfg.iou_threshold,
                tracker=self._cfg.tracker,
            )
        except Exception:
            results = self.model(
                frame, verbose=False,
                conf=self._cfg.conf_threshold,
                iou=self._cfg.iou_threshold,
            )

        dets = self._tm.update(results, self._cfg.conf_threshold)
        fe = self._engine.evaluate(dets, self._tm, self._frame_idx)
        trails = self._tm.get_all_trails()

        self._exporter.log_events(fe)
        self._frame_idx += 1

        return {
            "detections": dets,
            "frame_events": fe,
            "trails": trails,
            "line_counts": fe.line_counts,
            "wrong_way_count": fe.wrong_way_count,
            "_frame": frame,
        }

    def visualize(self, input_data, output) -> np.ndarray:
        frame = output.get("_frame")
        if frame is None:
            frame = input_data if isinstance(input_data, np.ndarray) else cv2.imread(str(input_data))
        return self._renderer.draw(
            frame,
            output["detections"],
            output["frame_events"],
            output["trails"],
            self._cfg,
        )

    # ── Project-specific API ───────────────────────────────

    def set_config(self, cfg: TrafficConfig) -> None:
        """Hot-swap configuration."""
        self._cfg = cfg
        self._engine = RuleEngine(cfg)

    def export_events(self) -> None:
        """Flush accumulated events to disk."""
        self._exporter.flush()
