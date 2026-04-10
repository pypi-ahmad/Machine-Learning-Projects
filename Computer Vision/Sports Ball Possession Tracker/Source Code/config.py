"""Configuration dataclasses for Sports Ball Possession Tracker.

Provides :class:`PossessionConfig` with all tunables for the detection,
tracking, and possession-estimation pipeline.
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
class PossessionConfig:
    """Top-level project configuration."""

    # Detection
    model: str = "yolo26m.pt"
    conf_threshold: float = 0.30
    iou_threshold: float = 0.45
    imgsz: int = 1280
    device: str = ""

    # Class mapping — indices into model.names for player and ball.
    # Set to -1 to auto-detect from model class names at runtime.
    player_class_id: int = -1
    ball_class_id: int = -1

    # Tracking
    tracker_type: str = "bytetrack"   # "bytetrack" or "botsort"
    track_high_thresh: float = 0.3
    track_low_thresh: float = 0.1
    track_buffer: int = 30

    # Possession
    possession_radius_px: int = 120   # max pixel distance ball→player for possession
    possession_hold_frames: int = 5   # keep possession for N frames after losing proximity
    min_ball_conf: float = 0.20       # minimum confidence for ball detections

    # Export
    export_json: str = ""
    export_csv: str = ""

    # Display
    show_display: bool = True
    show_trails: bool = True
    trail_length: int = 30
    show_possession_bar: bool = True
    line_width: int = 2

    # Save
    save_video: bool = False
    output_path: str = "output/possession_output.mp4"


def load_config(path: str | Path | None) -> PossessionConfig:
    """Load configuration from YAML or JSON, falling back to defaults."""
    if path is None:
        return PossessionConfig()

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


def _dict_to_config(d: dict[str, Any]) -> PossessionConfig:
    cfg = PossessionConfig()
    for key, val in d.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg
