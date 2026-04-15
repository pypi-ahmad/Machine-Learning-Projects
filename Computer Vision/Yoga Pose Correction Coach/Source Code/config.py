"""Yoga Pose Correction Coach -- configuration dataclass."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any


# Supported yoga poses
YOGA_POSES: list[str] = [
    "mountain",       # Tadasana
    "warrior_i",      # Virabhadrasana I
    "warrior_ii",     # Virabhadrasana II
    "tree",           # Vrksasana
    "downward_dog",   # Adho Mukha Svanasana
]


@dataclasses.dataclass
class YogaConfig:
    """All tunables for the yoga pose analysis pipeline."""

    # --- MediaPipe Pose ---
    model_complexity: int = 2          # 0, 1, or 2 (2 = highest accuracy)
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5
    static_image_mode: bool = False

    # --- Classification ---
    min_visibility: float = 0.5        # per-landmark visibility gate
    confidence_threshold: float = 0.4  # minimum classification score

    # --- Smoothing ---
    enable_smoothing: bool = True
    vote_window: int = 7               # majority-vote window for pose label

    # --- Correction hints ---
    angle_tolerance: float = 15.0      # degrees of acceptable deviation
    max_hints: int = 3                 # max correction hints per frame

    # --- Display ---
    show_skeleton: bool = True
    show_angles: bool = True
    show_pose_label: bool = True
    show_confidence: bool = True
    show_corrections: bool = True

    # --- Output ---
    output_dir: str = "output"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> YogaConfig:
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})


def load_config(path: str | Path) -> YogaConfig:
    """Load config from YAML or JSON, falling back to defaults."""
    p = Path(path)
    if not p.exists():
        return YogaConfig()
    text = p.read_text(encoding="utf-8")
    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml

            data = yaml.safe_load(text) or {}
        except Exception:
            data = {}
    else:
        data = json.loads(text)
    return YogaConfig.from_dict(data)
