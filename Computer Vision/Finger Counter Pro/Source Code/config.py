"""Finger Counter Pro -- configuration dataclass."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any


@dataclasses.dataclass
class FingerCounterConfig:
    """All tunables for the finger-counting pipeline."""

    # --- MediaPipe Hand Landmarker ---
    max_num_hands: int = 2
    model_complexity: int = 1
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5

    # --- Finger detection ---
    finger_up_margin: float = 0.02  # y-gap above PIP to consider extended

    # --- Smoothing ---
    enable_smoothing: bool = True
    ema_alpha: float = 0.35       # 0->sluggish, 1->raw (per-hand count)
    vote_window: int = 5          # majority-vote window size

    # --- Display ---
    show_landmarks: bool = True
    show_finger_state: bool = True
    show_count: bool = True
    show_handedness: bool = True

    # --- Export / output ---
    output_dir: str = "output"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FingerCounterConfig:
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})


def load_config(path: str | Path) -> FingerCounterConfig:
    """Load config from YAML or JSON, falling back to defaults."""
    p = Path(path)
    if not p.exists():
        return FingerCounterConfig()
    text = p.read_text(encoding="utf-8")
    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml
            data = yaml.safe_load(text) or {}
        except Exception:
            data = {}
    else:
        data = json.loads(text)
    return FingerCounterConfig.from_dict(data)
