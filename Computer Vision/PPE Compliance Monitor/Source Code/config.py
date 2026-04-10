"""PPE Compliance Monitor — configuration loader.

Loads zone definitions, required PPE items, alert settings, model
parameters, and export options from a YAML or JSON config file.

Usage::

    from config import load_config, PPEConfig

    cfg = load_config("ppe_config.yaml")
    # or
    cfg = PPEConfig()  # defaults
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# PPE class names — canonical set
# ---------------------------------------------------------------------------
PERSON_CLASS = "person"

# Required PPE items (extend this list for gloves, goggles, boots, etc.)
DEFAULT_REQUIRED_PPE: list[str] = ["helmet", "safety_vest"]

# All known PPE-related class names the detector may see
ALL_PPE_CLASSES: list[str] = [
    "helmet",
    "safety_vest",
    "gloves",
    "goggles",
    "boots",
    "mask",
    "ear_protection",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ZoneConfig:
    """A monitored zone polygon."""

    name: str
    polygon: list[tuple[int, int]]
    required_ppe: list[str] | None = None  # None → use global default_required_ppe

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "polygon": self.polygon,
            "required_ppe": self.required_ppe,
        }


@dataclass
class PPEConfig:
    """Top-level project configuration."""

    # Model
    model: str = "yolo26m.pt"
    conf_threshold: float = 0.30
    iou_threshold: float = 0.45
    imgsz: int = 640
    device: str | None = None

    # Compliance
    required_ppe: list[str] = field(default_factory=lambda: list(DEFAULT_REQUIRED_PPE))
    person_class: str = PERSON_CLASS
    ppe_iou_threshold: float = 0.20  # min overlap to associate PPE with person

    # Zones (optional — if empty, whole frame is one zone)
    zones: list[ZoneConfig] = field(default_factory=list)

    # Alerts
    alert_cooldown_sec: float = 10.0

    # Export
    export_dir: str = "outputs"
    save_events_csv: bool = True
    save_events_json: bool = True
    save_violation_snapshots: bool = True
    snapshot_cooldown_sec: float = 10.0

    # Inference
    show_display: bool = True
    save_video: bool = False
    output_fps: int = 25

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "zones"}
        d["zones"] = [z.to_dict() for z in self.zones]
        return d


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _parse_zones(raw: list[dict]) -> list[ZoneConfig]:
    zones: list[ZoneConfig] = []
    for entry in raw:
        polygon = [tuple(pt) for pt in entry["polygon"]]
        zones.append(ZoneConfig(
            name=entry["name"],
            polygon=polygon,
            required_ppe=entry.get("required_ppe"),
        ))
    return zones


def load_config(path: str | Path) -> PPEConfig:
    """Load a ``PPEConfig`` from a YAML or JSON file."""
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

    cfg = PPEConfig()

    # Model
    cfg.model = data.get("model", cfg.model)
    cfg.conf_threshold = data.get("conf_threshold", cfg.conf_threshold)
    cfg.iou_threshold = data.get("iou_threshold", cfg.iou_threshold)
    cfg.imgsz = data.get("imgsz", cfg.imgsz)
    cfg.device = data.get("device", cfg.device)

    # Compliance
    cfg.required_ppe = data.get("required_ppe", cfg.required_ppe)
    cfg.person_class = data.get("person_class", cfg.person_class)
    cfg.ppe_iou_threshold = data.get("ppe_iou_threshold", cfg.ppe_iou_threshold)

    # Zones
    if "zones" in data:
        cfg.zones = _parse_zones(data["zones"])

    # Alerts
    cfg.alert_cooldown_sec = data.get("alert_cooldown_sec", cfg.alert_cooldown_sec)

    # Export
    cfg.export_dir = data.get("export_dir", cfg.export_dir)
    cfg.save_events_csv = data.get("save_events_csv", cfg.save_events_csv)
    cfg.save_events_json = data.get("save_events_json", cfg.save_events_json)
    cfg.save_violation_snapshots = data.get("save_violation_snapshots", cfg.save_violation_snapshots)
    cfg.snapshot_cooldown_sec = data.get("snapshot_cooldown_sec", cfg.snapshot_cooldown_sec)

    # Inference
    cfg.show_display = data.get("show_display", cfg.show_display)
    cfg.save_video = data.get("save_video", cfg.save_video)
    cfg.output_fps = data.get("output_fps", cfg.output_fps)

    return cfg


def save_config(cfg: PPEConfig, path: str | Path) -> None:
    """Persist a config to disk (JSON)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")


def default_sample_config() -> PPEConfig:
    """Return a demo config with two example zones."""
    cfg = PPEConfig()
    cfg.zones = [
        ZoneConfig(
            name="Entry-Gate",
            polygon=[(50, 50), (400, 50), (400, 450), (50, 450)],
            required_ppe=["helmet", "safety_vest"],
        ),
        ZoneConfig(
            name="Loading-Dock",
            polygon=[(420, 50), (800, 50), (800, 450), (420, 450)],
            required_ppe=["helmet", "safety_vest", "gloves"],
        ),
    ]
    return cfg
