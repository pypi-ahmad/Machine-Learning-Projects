"""Configuration dataclass for Gesture Controlled Slideshow.

Provides :class:`GestureConfig` with tunables for MediaPipe Hand
Landmarker, gesture recognition, debouncing, slideshow behaviour,
keyboard fallback, export, and display settings.
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
class GestureConfig:
    """Top-level project configuration."""

    # ── MediaPipe Hands ────────────────────────────────────
    max_num_hands: int = 1
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5
    model_complexity: int = 1           # 0 = lite, 1 = full

    # ── Gesture recognition ────────────────────────────────
    # Finger-up detection: tip must be above PIP by this pixel
    # margin (relative to hand bbox height) to count as "up"
    finger_up_margin: float = 0.02

    # ── Gesture → action mapping ───────────────────────────
    # Keys: gesture names, values: slideshow actions
    gesture_map: dict[str, str] = field(default_factory=lambda: {
        "OPEN_PALM": "next",
        "FIST": "previous",
        "PEACE": "pause",
        "POINTING": "pointer",
        "THUMBS_UP": "resume",
    })

    # ── Debouncing ─────────────────────────────────────────
    debounce_sec: float = 0.8          # min time between actions
    confidence_threshold: float = 0.6   # gesture confidence floor
    stable_frames: int = 3             # consecutive same-gesture frames

    # ── Slideshow ──────────────────────────────────────────
    slide_dir: str = ""                 # folder of slide images
    auto_advance_sec: float = 0.0       # 0 = manual only
    loop: bool = True

    # ── Keyboard fallback ──────────────────────────────────
    enable_keyboard: bool = True
    key_next: int = ord("n")            # 'n' key
    key_prev: int = ord("p")            # 'p' key
    key_pause: int = ord(" ")           # space
    key_pointer: int = ord("t")         # 't' for toggle pointer

    # ── Validation ─────────────────────────────────────────
    warn_no_hand: bool = True

    # ── Export ─────────────────────────────────────────────
    export_json: str = ""
    export_csv: str = ""

    # ── Display ────────────────────────────────────────────
    show_display: bool = True
    show_hand_landmarks: bool = True
    show_gesture_label: bool = True
    show_finger_state: bool = True
    show_slide_counter: bool = True
    show_action_banner: bool = True
    line_width: int = 2

    # ── Save ───────────────────────────────────────────────
    save_annotated: bool = False
    output_dir: str = "output"


def load_config(path: str | Path | None) -> GestureConfig:
    """Load config from YAML or JSON; falls back to defaults."""
    if path is None:
        return GestureConfig()

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


def _dict_to_config(d: dict[str, Any]) -> GestureConfig:
    cfg = GestureConfig()
    for key, val in d.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg
