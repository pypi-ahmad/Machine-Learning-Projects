"""Traffic Violation Analyzer — configuration loader.

Loads virtual lines, zone definitions, model parameters, rule-engine
settings, and export options from a YAML or JSON config file.

Usage::

    from config import load_config, TrafficConfig

    cfg = load_config("traffic.yaml")
    # or
    cfg = TrafficConfig()  # defaults
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Vehicle class names the detector should look for
# ---------------------------------------------------------------------------
DEFAULT_VEHICLE_CLASSES: list[str] = [
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class LineConfig:
    """A virtual counting / violation line."""

    name: str
    pt1: tuple[int, int]                   # (x, y) start
    pt2: tuple[int, int]                   # (x, y) end
    direction: str = "any"                 # "up", "down", "left", "right", "any"
    # direction specifies the *allowed* crossing direction.
    # Crossings opposite to this are flagged as wrong-way.

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "pt1": list(self.pt1),
            "pt2": list(self.pt2),
            "direction": self.direction,
        }


@dataclass
class ZoneConfig:
    """A polygon zone for restricted-area monitoring."""

    name: str
    polygon: list[tuple[int, int]]

    def to_dict(self) -> dict:
        return {"name": self.name, "polygon": self.polygon}


@dataclass
class TrafficConfig:
    """Top-level project configuration."""

    # Model
    model: str = "yolo26m.pt"
    conf_threshold: float = 0.30
    iou_threshold: float = 0.45
    imgsz: int = 640
    device: str | None = None
    tracker: str = "bytetrack.yaml"        # Ultralytics tracker config

    # Detection
    vehicle_classes: list[str] = field(
        default_factory=lambda: list(DEFAULT_VEHICLE_CLASSES),
    )

    # Lines & zones
    lines: list[LineConfig] = field(default_factory=list)
    zones: list[ZoneConfig] = field(default_factory=list)

    # Export
    export_dir: str = "outputs"
    save_events_csv: bool = True
    save_events_json: bool = True

    # Inference / display
    show_display: bool = True
    save_video: bool = False
    output_fps: int = 25

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k not in ("lines", "zones")}
        d["lines"] = [ln.to_dict() for ln in self.lines]
        d["zones"] = [z.to_dict() for z in self.zones]
        return d


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _parse_lines(raw: list[dict]) -> list[LineConfig]:
    lines: list[LineConfig] = []
    for entry in raw:
        lines.append(LineConfig(
            name=entry["name"],
            pt1=tuple(entry["pt1"]),
            pt2=tuple(entry["pt2"]),
            direction=entry.get("direction", "any"),
        ))
    return lines


def _parse_zones(raw: list[dict]) -> list[ZoneConfig]:
    zones: list[ZoneConfig] = []
    for entry in raw:
        polygon = [tuple(pt) for pt in entry["polygon"]]
        zones.append(ZoneConfig(name=entry["name"], polygon=polygon))
    return zones


def load_config(path: str | Path) -> TrafficConfig:
    """Load a ``TrafficConfig`` from a YAML or JSON file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
            data = yaml.safe_load(text) or {}
        except ImportError:
            raise ImportError("PyYAML required: pip install pyyaml")
    elif path.suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")

    cfg = TrafficConfig()

    # Model
    cfg.model = data.get("model", cfg.model)
    cfg.conf_threshold = data.get("conf_threshold", cfg.conf_threshold)
    cfg.iou_threshold = data.get("iou_threshold", cfg.iou_threshold)
    cfg.imgsz = data.get("imgsz", cfg.imgsz)
    cfg.device = data.get("device", cfg.device)
    cfg.tracker = data.get("tracker", cfg.tracker)

    # Detection
    cfg.vehicle_classes = data.get("vehicle_classes", cfg.vehicle_classes)

    # Lines & zones
    if "lines" in data:
        cfg.lines = _parse_lines(data["lines"])
    if "zones" in data:
        cfg.zones = _parse_zones(data["zones"])

    # Export
    cfg.export_dir = data.get("export_dir", cfg.export_dir)
    cfg.save_events_csv = data.get("save_events_csv", cfg.save_events_csv)
    cfg.save_events_json = data.get("save_events_json", cfg.save_events_json)

    # Inference
    cfg.show_display = data.get("show_display", cfg.show_display)
    cfg.save_video = data.get("save_video", cfg.save_video)
    cfg.output_fps = data.get("output_fps", cfg.output_fps)

    return cfg


def save_config(cfg: TrafficConfig, path: str | Path) -> None:
    """Persist a config to disk (JSON)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")


def default_sample_config() -> TrafficConfig:
    """Return a demo config with two counting lines and one zone."""
    cfg = TrafficConfig()
    cfg.lines = [
        LineConfig(
            name="Line-North",
            pt1=(100, 300),
            pt2=(700, 300),
            direction="up",
        ),
        LineConfig(
            name="Line-South",
            pt1=(100, 450),
            pt2=(700, 450),
            direction="down",
        ),
    ]
    cfg.zones = [
        ZoneConfig(
            name="Intersection",
            polygon=[(200, 200), (600, 200), (600, 500), (200, 500)],
        ),
    ]
    return cfg
