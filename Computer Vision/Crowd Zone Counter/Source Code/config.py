"""Configuration dataclasses for Crowd Zone Counter.

Provides :class:`ZoneConfig` and :class:`CrowdConfig` with all tunables
for the zone-based people counting pipeline: model, thresholds,
zone polygons, overcrowding limits, export paths, and display options.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


@dataclass
class ZoneConfig:
    """A named counting zone with an overcrowding threshold."""

    name: str
    polygon: list[list[int]]        # [[x1,y1], [x2,y2], ...]
    max_capacity: int = 0            # 0 = no limit (alert disabled)
    colour: list[int] | None = None  # BGR override; None = auto


@dataclass
class CrowdConfig:
    """Top-level project configuration."""

    # Detection
    model: str = "yolo26m.pt"
    conf_threshold: float = 0.30
    iou_threshold: float = 0.45
    imgsz: int = 640
    device: str = ""

    # Only detect "person" (COCO class 0)
    person_class_id: int = 0

    # Zones
    zones: list[ZoneConfig] = field(default_factory=list)

    # Overcrowding
    alert_cooldown_frames: int = 30   # suppress repeated alerts

    # Export
    export_json: str = ""
    export_csv: str = ""

    # Display
    show_display: bool = True
    show_zone_fill: bool = True
    zone_alpha: float = 0.25
    line_width: int = 2
    show_counts: bool = True
    show_alerts: bool = True

    # Save
    save_video: bool = False
    output_path: str = "output/crowd_output.mp4"


def load_config(path: str | Path | None) -> CrowdConfig:
    """Load configuration from YAML or JSON, falling back to defaults."""
    if path is None:
        return CrowdConfig()

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


def _dict_to_config(d: dict[str, Any]) -> CrowdConfig:
    cfg = CrowdConfig()
    zones_raw = d.pop("zones", [])
    for key, val in d.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)

    for z in zones_raw:
        cfg.zones.append(ZoneConfig(
            name=z.get("name", "Zone"),
            polygon=z.get("polygon", []),
            max_capacity=z.get("max_capacity", 0),
            colour=z.get("colour"),
        ))
    return cfg


def default_sample_zones() -> list[ZoneConfig]:
    """Return demo zones for a 1280×720 frame."""
    return [
        ZoneConfig(
            name="Entrance",
            polygon=[[50, 50], [400, 50], [400, 400], [50, 400]],
            max_capacity=15,
        ),
        ZoneConfig(
            name="Main-Area",
            polygon=[[450, 50], [900, 50], [900, 650], [450, 650]],
            max_capacity=40,
        ),
        ZoneConfig(
            name="Exit",
            polygon=[[950, 50], [1230, 50], [1230, 400], [950, 400]],
            max_capacity=10,
        ),
    ]
