"""Exercise Rep Counter — configuration dataclass."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any


@dataclasses.dataclass
class ExerciseConfig:
    """All tunables for the exercise rep-counting pipeline."""

    # --- MediaPipe Pose ---
    model_complexity: int = 1        # 0, 1, or 2
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5
    static_image_mode: bool = False

    # --- Exercise selection ---
    exercise: str = "squat"  # "squat", "pushup", "bicep_curl"

    # --- Stage detection thresholds (degrees) ---
    # Squat: hip-knee-ankle angle
    squat_down_angle: float = 90.0    # below → "down"
    squat_up_angle: float = 160.0     # above → "up"

    # Push-up: shoulder-elbow-wrist angle
    pushup_down_angle: float = 90.0
    pushup_up_angle: float = 160.0

    # Bicep curl: shoulder-elbow-wrist angle
    curl_down_angle: float = 160.0    # arm extended → "down"
    curl_up_angle: float = 40.0       # arm curled → "up"

    # --- Smoothing ---
    enable_smoothing: bool = True
    ema_alpha: float = 0.4            # EMA on raw angles
    stable_frames: int = 2            # consecutive same-stage frames to confirm

    # --- Display ---
    show_skeleton: bool = True
    show_angles: bool = True
    show_rep_count: bool = True
    show_stage: bool = True
    show_exercise: bool = True

    # --- Output ---
    output_dir: str = "output"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExerciseConfig:
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})


def load_config(path: str | Path) -> ExerciseConfig:
    """Load config from YAML or JSON, falling back to defaults."""
    p = Path(path)
    if not p.exists():
        return ExerciseConfig()
    text = p.read_text(encoding="utf-8")
    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml

            data = yaml.safe_load(text) or {}
        except Exception:
            data = {}
    else:
        data = json.loads(text)
    return ExerciseConfig.from_dict(data)
