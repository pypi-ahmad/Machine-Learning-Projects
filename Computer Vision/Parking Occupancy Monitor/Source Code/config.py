"""Parking Occupancy Monitor — configuration loader.

Loads parking-slot polygon definitions, model parameters, alert settings,
and export options from a YAML or JSON config file.

Usage::

    from config import load_config, ParkingConfig

    cfg = load_config("slots.yaml")
    # or
    cfg = ParkingConfig()  # defaults
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
class SlotConfig:
    """A single parking slot polygon."""

    name: str
    polygon: list[tuple[int, int]]  # [(x1,y1), (x2,y2), …]

    def to_dict(self) -> dict:
        return {"name": self.name, "polygon": self.polygon}


@dataclass
class ParkingConfig:
    """Top-level project configuration."""

    # Model
    model: str = "yolo26m.pt"
    model_live: str = "yolo26s.pt"          # lighter model for webcam mode
    conf_threshold: float = 0.30
    iou_threshold: float = 0.45
    imgsz: int = 640
    device: str | None = None

    # Detection
    vehicle_classes: list[str] = field(
        default_factory=lambda: list(DEFAULT_VEHICLE_CLASSES),
    )
    occupancy_iou_threshold: float = 0.15   # min overlap to mark a slot occupied

    # Slots (if empty → no per-slot analytics, just vehicle counting)
    slots: list[SlotConfig] = field(default_factory=list)

    # Export
    export_dir: str = "outputs"
    save_events_csv: bool = True
    save_events_json: bool = True

    # Inference / display
    show_display: bool = True
    save_video: bool = False
    output_fps: int = 25

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "slots"}
        d["slots"] = [s.to_dict() for s in self.slots]
        return d


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _parse_slots(raw: list[dict]) -> list[SlotConfig]:
    slots: list[SlotConfig] = []
    for entry in raw:
        polygon = [tuple(pt) for pt in entry["polygon"]]
        slots.append(SlotConfig(name=entry["name"], polygon=polygon))
    return slots


def load_config(path: str | Path) -> ParkingConfig:
    """Load a ``ParkingConfig`` from a YAML or JSON file."""
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

    cfg = ParkingConfig()

    # Model
    cfg.model = data.get("model", cfg.model)
    cfg.model_live = data.get("model_live", cfg.model_live)
    cfg.conf_threshold = data.get("conf_threshold", cfg.conf_threshold)
    cfg.iou_threshold = data.get("iou_threshold", cfg.iou_threshold)
    cfg.imgsz = data.get("imgsz", cfg.imgsz)
    cfg.device = data.get("device", cfg.device)

    # Detection
    cfg.vehicle_classes = data.get("vehicle_classes", cfg.vehicle_classes)
    cfg.occupancy_iou_threshold = data.get("occupancy_iou_threshold",
                                           cfg.occupancy_iou_threshold)

    # Slots
    if "slots" in data:
        cfg.slots = _parse_slots(data["slots"])

    # Export
    cfg.export_dir = data.get("export_dir", cfg.export_dir)
    cfg.save_events_csv = data.get("save_events_csv", cfg.save_events_csv)
    cfg.save_events_json = data.get("save_events_json", cfg.save_events_json)

    # Inference
    cfg.show_display = data.get("show_display", cfg.show_display)
    cfg.save_video = data.get("save_video", cfg.save_video)
    cfg.output_fps = data.get("output_fps", cfg.output_fps)

    return cfg


def save_config(cfg: ParkingConfig, path: str | Path) -> None:
    """Persist a config to disk (JSON)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")


def default_sample_config() -> ParkingConfig:
    """Return a demo config with eight example parking slots."""
    cfg = ParkingConfig()
    # Two rows of four slots (typical small lot view)
    slot_w, slot_h = 90, 130
    for row, y0 in enumerate([60, 220]):
        for col in range(4):
            x0 = 50 + col * 110
            cfg.slots.append(SlotConfig(
                name=f"R{row + 1}-S{col + 1}",
                polygon=[
                    (x0, y0),
                    (x0 + slot_w, y0),
                    (x0 + slot_w, y0 + slot_h),
                    (x0, y0 + slot_h),
                ],
            ))
    return cfg
