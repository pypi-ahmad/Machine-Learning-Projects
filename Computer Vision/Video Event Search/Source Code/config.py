"""Video Event Search — configuration and event schema.

Provides :class:`EventSearchConfig` with all tunables: YOLO model
settings, zone / line definitions, dwell-time thresholds, event-store
paths, and display options.

Defines the event types and :class:`Event` dataclass used throughout
the project.

Usage::

    from config import EventSearchConfig, load_config, EventType

    cfg = load_config("event_search.yaml")
    cfg = EventSearchConfig()          # defaults
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class EventType(str, enum.Enum):
    """Structured event types emitted by the event generator."""

    APPEAR = "appear"
    DISAPPEAR = "disappear"
    ZONE_ENTER = "zone_enter"
    ZONE_EXIT = "zone_exit"
    LINE_CROSS = "line_cross"
    DWELL = "dwell"


# ---------------------------------------------------------------------------
# Event data class
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """A single structured event emitted during video analysis."""

    event_type: str          # EventType value
    track_id: int
    class_name: str
    frame_idx: int
    timestamp_sec: float     # seconds into the video
    confidence: float = 0.0
    center: tuple[int, int] = (0, 0)
    zone_or_line: str = ""   # name of zone/line involved
    direction: str = ""      # crossing direction (for LINE_CROSS)
    dwell_seconds: float = 0.0  # (for DWELL events)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "track_id": self.track_id,
            "class_name": self.class_name,
            "frame_idx": self.frame_idx,
            "timestamp_sec": round(self.timestamp_sec, 3),
            "confidence": round(self.confidence, 3),
            "center_x": self.center[0],
            "center_y": self.center[1],
            "zone_or_line": self.zone_or_line,
            "direction": self.direction,
            "dwell_seconds": round(self.dwell_seconds, 2),
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Zone / line configuration
# ---------------------------------------------------------------------------

@dataclass
class LineConfig:
    """A virtual counting line."""

    name: str
    pt1: tuple[int, int]
    pt2: tuple[int, int]
    direction: str = "any"        # allowed direction: up/down/left/right/any


@dataclass
class ZoneConfig:
    """A named polygon zone."""

    name: str
    polygon: list[list[int]]     # [[x1,y1], [x2,y2], ...]


# ---------------------------------------------------------------------------
# Top-level configuration
# ---------------------------------------------------------------------------

@dataclass
class EventSearchConfig:
    """Top-level project configuration."""

    # Detection model
    model: str = "yolo26m.pt"
    conf_threshold: float = 0.30
    iou_threshold: float = 0.45
    imgsz: int = 640
    device: str | None = None
    tracker: str = "bytetrack.yaml"

    # Classes to track (empty = all COCO classes)
    target_classes: list[str] = field(default_factory=list)

    # Lines and zones
    lines: list[LineConfig] = field(default_factory=list)
    zones: list[ZoneConfig] = field(default_factory=list)

    # Dwell-time threshold (seconds)
    dwell_threshold: float = 5.0

    # Disappearance threshold — frames without detection before DISAPPEAR
    disappear_frames: int = 30

    # Event store
    event_store_path: str = "outputs/events.json"
    export_csv: bool = True
    export_json: bool = True
    export_dir: str = "outputs"

    # Display
    show_display: bool = True
    save_video: bool = False
    output_fps: int = 25

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if k == "lines":
                d[k] = [{"name": ln.name, "pt1": list(ln.pt1),
                          "pt2": list(ln.pt2), "direction": ln.direction}
                         for ln in v]
            elif k == "zones":
                d[k] = [{"name": z.name, "polygon": z.polygon} for z in v]
            else:
                d[k] = v
        return d


# ---------------------------------------------------------------------------
# Default sample configuration (demo on pedestrian dataset)
# ---------------------------------------------------------------------------

def default_sample_config() -> EventSearchConfig:
    """Return a demo config suitable for the pedestrian crosswalk video.

    Defines one counting line across the crosswalk and one zone.
    Coordinates are approximate for a 640-wide frame.
    """
    return EventSearchConfig(
        lines=[
            LineConfig(name="crosswalk_line", pt1=(50, 300), pt2=(590, 300)),
        ],
        zones=[
            ZoneConfig(name="crosswalk_zone", polygon=[
                [40, 200], [600, 200], [600, 400], [40, 400],
            ]),
        ],
        target_classes=["person"],
        dwell_threshold=3.0,
    )


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str | Path | None = None) -> EventSearchConfig:
    """Load configuration from YAML or JSON, falling back to defaults."""
    if path is None:
        return default_sample_config()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        import yaml
        raw = yaml.safe_load(text) or {}
    else:
        raw = json.loads(text)

    return _dict_to_config(raw)


def _dict_to_config(d: dict[str, Any]) -> EventSearchConfig:
    cfg = EventSearchConfig()
    lines_raw = d.pop("lines", [])
    zones_raw = d.pop("zones", [])

    for key, val in d.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)

    for ln in lines_raw:
        cfg.lines.append(LineConfig(
            name=ln["name"],
            pt1=tuple(ln["pt1"]),
            pt2=tuple(ln["pt2"]),
            direction=ln.get("direction", "any"),
        ))

    for z in zones_raw:
        cfg.zones.append(ZoneConfig(
            name=z["name"],
            polygon=z["polygon"],
        ))

    return cfg
