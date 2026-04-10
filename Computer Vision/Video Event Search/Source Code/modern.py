"""Modern v2 pipeline — Video Event Search.

Uses:     YOLO26m detection + built-in ByteTrack tracker
Pipeline: YOLO detect+track → event generator (zone/line/dwell/appear/disappear) → event store → query

Delegates tracking to ``tracker.py``, event generation to
``event_generator.py``, storage to ``event_store.py``, queries to
``query.py``, and visualisation to ``visualize.py``.
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

from config import EventSearchConfig, load_config, default_sample_config
from detector import Detection
from event_generator import EventGenerator
from event_store import EventStore
from query import EventQuery
from tracker import TrackManager
from visualize import OverlayRenderer


@register("video_event_search")
class VideoEventSearchModern(CVProject):
    project_type = "tracking"
    description = "Video event detection with zone entry/exit, line crossing, dwell time, and searchable event logs"
    legacy_tech = "N/A (new project)"
    modern_tech = "YOLO26m detection + ByteTrack + event generator + JSON/CSV event store + query engine"

    def __init__(self, config: EventSearchConfig | None = None) -> None:
        super().__init__()
        self._cfg = config or default_sample_config()
        self._tm = TrackManager(max_history=120)
        self._gen: EventGenerator | None = None
        self._store: EventStore | None = None
        self._renderer = OverlayRenderer()
        self.model = None
        self._frame_idx = 0
        self._recent_events = []
        self._fps = 25.0

    # ── CVProject interface ────────────────────────────────

    def load(self) -> None:
        from models.registry import resolve
        weights, ver, fallback = resolve("video_event_search", "detect")
        self.model = load_yolo(weights)
        print(
            f"Using model for video_event_search: version={ver} "
            f"weights={weights} pretrained_fallback={fallback}"
        )

    def predict(self, input_data) -> dict:
        frame = input_data if isinstance(input_data, np.ndarray) else cv2.imread(str(input_data))

        # Lazy init event generator and store
        if self._gen is None:
            self._gen = EventGenerator(self._cfg, fps=self._fps)
        if self._store is None:
            out_dir = Path(self._cfg.export_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            self._store = EventStore(out_dir / "events.json")

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
        events = self._gen.process(dets, self._tm, self._frame_idx)
        self._store.add_batch(events)
        self._recent_events = (self._recent_events + events)[-10:]
        self._frame_idx += 1

        return {
            "detections": dets,
            "events": events,
            "recent_events": self._recent_events,
            "trails": self._tm.get_all_trails(),
            "event_count": self._store.count,
            "_frame": frame,
        }

    def visualize(self, input_data, output) -> np.ndarray:
        frame = output.get("_frame")
        if frame is None:
            frame = input_data if isinstance(input_data, np.ndarray) else cv2.imread(str(input_data))
        return self._renderer.draw(
            frame,
            output["detections"],
            output["recent_events"],
            output["trails"],
            self._cfg,
        )

    # ── Project-specific API ───────────────────────────────

    def set_config(self, cfg: EventSearchConfig) -> None:
        """Hot-swap configuration."""
        self._cfg = cfg
        self._gen = EventGenerator(cfg, fps=self._fps)

    def set_fps(self, fps: float) -> None:
        """Update video FPS for timestamp calculations."""
        self._fps = fps
        if self._gen:
            self._gen.fps = fps

    def flush_events(self) -> None:
        """Write accumulated events to disk."""
        if self._store:
            self._store.flush()

    def query_events(self, **kwargs) -> list[dict]:
        """Search stored events. See :class:`query.EventQuery.search`."""
        if self._store:
            self._store.flush()
        q = EventQuery(Path(self._cfg.export_dir) / "events.json")
        return q.search(**kwargs)

    def event_summary(self) -> dict:
        """Return event store summary."""
        if self._store:
            self._store.flush()
        q = EventQuery(Path(self._cfg.export_dir) / "events.json")
        return q.summary()
