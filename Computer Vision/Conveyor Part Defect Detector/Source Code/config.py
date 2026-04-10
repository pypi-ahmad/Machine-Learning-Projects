"""Conveyor Part Defect Detector — configuration loader.

Loads defect class definitions, pass/fail thresholds, model parameters,
crop settings, and export options from a YAML or JSON config file.

Usage::

    from config import load_config, InspectionConfig

    cfg = load_config("inspection.yaml")
    # or
    cfg = InspectionConfig()  # defaults
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Default defect class names
# ---------------------------------------------------------------------------
DEFAULT_DEFECT_CLASSES: list[str] = [
    "defect",
    "scratch",
    "dent",
    "crack",
    "missing_part",
    "discoloration",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class InspectionConfig:
    """Top-level project configuration."""

    # Model
    model: str = "yolo26m.pt"
    conf_threshold: float = 0.30
    iou_threshold: float = 0.45
    imgsz: int = 640
    device: str | None = None

    # Detection
    defect_classes: list[str] = field(
        default_factory=lambda: list(DEFAULT_DEFECT_CLASSES),
    )
    # If True, any detection from the model is treated as a defect
    # (useful when the dataset only contains defect classes, no "OK" class).
    all_classes_are_defects: bool = True

    # Pass / fail
    fail_threshold: int = 1        # ≥ this many defects → FAIL

    # Crop settings
    save_crops: bool = True
    crop_padding: int = 10         # pixels added around each defect crop

    # Export
    export_dir: str = "outputs"
    save_events_csv: bool = True
    save_events_json: bool = True

    # Inference / display
    show_display: bool = True
    save_video: bool = False
    output_fps: int = 25

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_config(path: str | Path) -> InspectionConfig:
    """Load an ``InspectionConfig`` from a YAML or JSON file."""
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

    cfg = InspectionConfig()

    # Model
    cfg.model = data.get("model", cfg.model)
    cfg.conf_threshold = data.get("conf_threshold", cfg.conf_threshold)
    cfg.iou_threshold = data.get("iou_threshold", cfg.iou_threshold)
    cfg.imgsz = data.get("imgsz", cfg.imgsz)
    cfg.device = data.get("device", cfg.device)

    # Detection
    cfg.defect_classes = data.get("defect_classes", cfg.defect_classes)
    cfg.all_classes_are_defects = data.get("all_classes_are_defects",
                                           cfg.all_classes_are_defects)

    # Pass / fail
    cfg.fail_threshold = data.get("fail_threshold", cfg.fail_threshold)

    # Crop
    cfg.save_crops = data.get("save_crops", cfg.save_crops)
    cfg.crop_padding = data.get("crop_padding", cfg.crop_padding)

    # Export
    cfg.export_dir = data.get("export_dir", cfg.export_dir)
    cfg.save_events_csv = data.get("save_events_csv", cfg.save_events_csv)
    cfg.save_events_json = data.get("save_events_json", cfg.save_events_json)

    # Inference
    cfg.show_display = data.get("show_display", cfg.show_display)
    cfg.save_video = data.get("save_video", cfg.save_video)
    cfg.output_fps = data.get("output_fps", cfg.output_fps)

    return cfg


def save_config(cfg: InspectionConfig, path: str | Path) -> None:
    """Persist a config to disk (JSON)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")


def default_sample_config() -> InspectionConfig:
    """Return a demo config with sensible defaults."""
    return InspectionConfig()
