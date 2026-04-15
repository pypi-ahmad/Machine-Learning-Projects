"""Waste Sorting Detector — configuration loader.

Loads waste class definitions, bin-zone polygons, model parameters,
and export options from a YAML or JSON config file.

Usage::

    from config import load_config, WasteConfig

    cfg = load_config("waste.yaml")
    # or
    cfg = WasteConfig()  # defaults
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Default waste class names
# ---------------------------------------------------------------------------
DEFAULT_WASTE_CLASSES: list[str] = [
    "plastic",
    "paper",
    "cardboard",
    "metal",
    "glass",
    "trash",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class BinZoneConfig:
    """A bin zone polygon that accepts specific waste classes."""

    name: str
    polygon: list[tuple[int, int]]
    accepted_classes: list[str]        # waste classes this bin accepts

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "polygon": self.polygon,
            "accepted_classes": self.accepted_classes,
        }


@dataclass
class WasteConfig:
    """Top-level project configuration."""

    # Model
    model: str = "yolo26m.pt"
    conf_threshold: float = 0.30
    iou_threshold: float = 0.45
    imgsz: int = 640
    device: str | None = None

    # Detection
    waste_classes: list[str] = field(
        default_factory=lambda: list(DEFAULT_WASTE_CLASSES),
    )

    # Bin zones (optional — if empty, no zone validation)
    bin_zones: list[BinZoneConfig] = field(default_factory=list)

    # Export
    export_dir: str = "outputs"
    export_csv: str = ""
    export_json: str = ""
    save_events_csv: bool = True
    save_events_json: bool = True

    # Inference / display
    show_display: bool = True
    show_counts: bool = True
    show_zones: bool = True
    save_video: bool = False
    output_fps: int = 25

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "bin_zones"}
        d["bin_zones"] = [b.to_dict() for b in self.bin_zones]
        return d


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _parse_bin_zones(raw: list[dict]) -> list[BinZoneConfig]:
    zones: list[BinZoneConfig] = []
    for entry in raw:
        polygon = [tuple(pt) for pt in entry["polygon"]]
        zones.append(BinZoneConfig(
            name=entry["name"],
            polygon=polygon,
            accepted_classes=entry.get("accepted_classes", []),
        ))
    return zones


def load_config(path: str | Path) -> WasteConfig:
    """Load a ``WasteConfig`` from a YAML or JSON file."""
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

    cfg = WasteConfig()

    # Model
    cfg.model = data.get("model", cfg.model)
    cfg.conf_threshold = data.get("conf_threshold", cfg.conf_threshold)
    cfg.iou_threshold = data.get("iou_threshold", cfg.iou_threshold)
    cfg.imgsz = data.get("imgsz", cfg.imgsz)
    cfg.device = data.get("device", cfg.device)

    # Detection
    cfg.waste_classes = data.get("waste_classes", cfg.waste_classes)

    # Bin zones
    if "bin_zones" in data:
        cfg.bin_zones = _parse_bin_zones(data["bin_zones"])

    # Export
    cfg.export_dir = data.get("export_dir", cfg.export_dir)
    cfg.export_csv = data.get("export_csv", cfg.export_csv)
    cfg.export_json = data.get("export_json", cfg.export_json)
    cfg.save_events_csv = data.get("save_events_csv", cfg.save_events_csv)
    cfg.save_events_json = data.get("save_events_json", cfg.save_events_json)

    # Inference
    cfg.show_display = data.get("show_display", cfg.show_display)
    cfg.show_counts = data.get("show_counts", cfg.show_counts)
    cfg.show_zones = data.get("show_zones", cfg.show_zones)
    cfg.save_video = data.get("save_video", cfg.save_video)
    cfg.output_fps = data.get("output_fps", cfg.output_fps)

    return cfg


def save_config(cfg: WasteConfig, path: str | Path) -> None:
    """Persist a config to disk (JSON)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")


def default_sample_config() -> WasteConfig:
    """Return a demo config with three bin zones."""
    cfg = WasteConfig()
    cfg.bin_zones = [
        BinZoneConfig(
            name="Recyclables",
            polygon=[(30, 300), (250, 300), (250, 550), (30, 550)],
            accepted_classes=["plastic", "paper", "cardboard", "metal", "glass"],
        ),
        BinZoneConfig(
            name="General-Waste",
            polygon=[(280, 300), (500, 300), (500, 550), (280, 550)],
            accepted_classes=["trash"],
        ),
        BinZoneConfig(
            name="Glass-Only",
            polygon=[(530, 300), (750, 300), (750, 550), (530, 550)],
            accepted_classes=["glass"],
        ),
    ]
    return cfg
