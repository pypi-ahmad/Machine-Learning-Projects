"""Configuration dataclasses for Drone Ship OBB Detector.

Provides :class:`OBBConfig` with all tunables for the oriented
bounding-box detection pipeline: model, thresholds, class filters,
export paths, and display settings.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Default classes for the ShipRSImageNet / DOTAv1.5 ship subset.
# Adjusted at runtime to match whatever the trained model emits.
DEFAULT_CLASSES: list[str] = [
    "ship",
    "large-vehicle",
    "small-vehicle",
    "plane",
    "helicopter",
    "harbor",
    "storage-tank",
    "container-crane",
]


@dataclass
class OBBConfig:
    """Top-level project configuration."""

    model: str = "yolo26m-obb.pt"
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    imgsz: int = 1024
    device: str = ""

    # Class filter — empty list means "report all model classes"
    target_classes: list[str] = field(default_factory=list)

    # Export
    export_json: str = ""
    export_txt: str = ""

    # Display
    show_display: bool = True
    line_width: int = 2
    label_font_scale: float = 0.45
    show_conf: bool = True
    show_angle: bool = True

    # Save
    save_video: bool = False
    output_fps: int = 25


def load_config(path: str | Path | None) -> OBBConfig:
    """Load configuration from YAML or JSON, falling back to defaults."""
    if path is None:
        return OBBConfig()

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


def _dict_to_config(d: dict[str, Any]) -> OBBConfig:
    cfg = OBBConfig()
    for key, val in d.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg
