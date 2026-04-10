"""Configuration dataclass for Gaze Direction Estimator.

Provides :class:`GazeConfig` with tunables for MediaPipe Face Mesh,
iris-based gaze classification, calibration, smoothing, export, and
display settings.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


@dataclass
class GazeConfig:
    """Top-level project configuration."""

    # ── MediaPipe Face Mesh ────────────────────────────────
    max_num_faces: int = 1
    refine_landmarks: bool = True       # required for iris landmarks
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # ── Gaze classification thresholds ─────────────────────
    # Iris horizontal ratio: 0.0 = fully left, 1.0 = fully right
    horiz_left_threshold: float = 0.38
    horiz_right_threshold: float = 0.62
    # Iris vertical ratio: 0.0 = fully up, 1.0 = fully down
    vert_up_threshold: float = 0.38
    vert_down_threshold: float = 0.62

    # ── Smoothing ──────────────────────────────────────────
    enable_smoothing: bool = True
    ema_alpha: float = 0.4              # EMA weight for new values
    vote_window: int = 5                # majority-vote window size

    # ── Calibration ────────────────────────────────────────
    enable_calibration: bool = False
    calibration_frames: int = 30        # frames per gaze position
    calibration_file: str = ""          # path to save/load calibration

    # ── Validation ─────────────────────────────────────────
    warn_no_face: bool = True

    # ── Export ─────────────────────────────────────────────
    export_json: str = ""
    export_csv: str = ""

    # ── Display ────────────────────────────────────────────
    show_display: bool = True
    show_iris_markers: bool = True
    show_eye_contours: bool = True
    show_gaze_label: bool = True
    show_ratios: bool = True
    show_stats_panel: bool = True
    line_width: int = 1

    # ── Save ───────────────────────────────────────────────
    save_annotated: bool = False
    output_dir: str = "output"


def load_config(path: str | Path | None) -> GazeConfig:
    """Load config from YAML or JSON; falls back to defaults."""
    if path is None:
        return GazeConfig()

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


def _dict_to_config(d: dict[str, Any]) -> GazeConfig:
    cfg = GazeConfig()
    for key, val in d.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg
