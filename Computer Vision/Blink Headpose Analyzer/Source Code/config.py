"""Configuration dataclass for Blink Headpose Analyzer.

Provides :class:`AnalyzerConfig` with tunables for MediaPipe Face
Mesh, blink counting (EAR), head-pose estimation, export, and
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
class AnalyzerConfig:
    """Top-level project configuration."""

    # ── MediaPipe Face Mesh ────────────────────────────────
    max_num_faces: int = 1
    refine_landmarks: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # ── Blink / EAR ───────────────────────────────────────
    ear_threshold: float = 0.21
    blink_consec_frames: int = 2

    # ── Head pose ──────────────────────────────────────────
    yaw_threshold: float = 30.0
    pitch_threshold: float = 25.0

    # ── Validation ─────────────────────────────────────────
    warn_no_face: bool = True

    # ── Export ─────────────────────────────────────────────
    export_json: str = ""
    export_csv: str = ""

    # ── Display ────────────────────────────────────────────
    show_display: bool = True
    show_eye_contours: bool = True
    show_ear_bar: bool = True
    show_pose_text: bool = True
    show_stats_panel: bool = True
    line_width: int = 1

    # ── Save ───────────────────────────────────────────────
    save_annotated: bool = False
    output_dir: str = "output"


def load_config(path: str | Path | None) -> AnalyzerConfig:
    """Load config from YAML or JSON; falls back to defaults."""
    if path is None:
        return AnalyzerConfig()

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


def _dict_to_config(d: dict[str, Any]) -> AnalyzerConfig:
    cfg = AnalyzerConfig()
    for key, val in d.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg
