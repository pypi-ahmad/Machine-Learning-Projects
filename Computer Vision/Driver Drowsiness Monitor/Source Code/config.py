"""Configuration dataclasses for Driver Drowsiness Monitor.

Provides :class:`DrowsinessConfig` with all tunables for the
MediaPipe landmark + rule-based drowsiness detection pipeline:
face mesh, EAR blink detection, yawn (MAR) detection, head-pose
distraction, alert cooldowns, export, and display settings.
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
class DrowsinessConfig:
    """Top-level project configuration."""

    # ── MediaPipe Face Mesh ────────────────────────────────
    max_num_faces: int = 1
    refine_landmarks: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # ── Blink / EAR ───────────────────────────────────────
    ear_threshold: float = 0.21         # EAR below this → eyes closed
    blink_consec_frames: int = 2        # frames below threshold → 1 blink
    drowsy_eye_frames: int = 15         # prolonged closure → drowsiness alert
    perclos_window_sec: float = 60.0    # PERCLOS calculation window
    perclos_threshold: float = 0.40     # fraction closed → drowsiness

    # ── Yawn / MAR (Mouth Aspect Ratio) ────────────────────
    mar_threshold: float = 0.55         # MAR above this → mouth open
    yawn_consec_frames: int = 10        # sustained opening → yawn
    yawn_cooldown_sec: float = 5.0      # min gap between yawn events

    # ── Head pose / distraction ────────────────────────────
    yaw_threshold: float = 30.0         # degrees off-center → distraction
    pitch_threshold: float = 25.0       # looking down/up → distraction
    distraction_consec_frames: int = 15 # sustained deviation → alert

    # ── Alert management ──────────────────────────────────
    alert_cooldown_sec: float = 10.0    # global cooldown between alerts
    log_dir: str = "drowsiness_logs"
    session_name: str = ""              # auto-generated if empty

    # ── Validation ─────────────────────────────────────────
    warn_no_face: bool = True
    warn_low_confidence: bool = True
    confidence_threshold: float = 0.35

    # ── Export ─────────────────────────────────────────────
    export_json: str = ""
    export_csv: str = ""

    # ── Display ────────────────────────────────────────────
    show_display: bool = True
    show_mesh: bool = False             # full face mesh overlay
    show_eye_contours: bool = True
    show_mouth_contours: bool = True
    show_ear_bar: bool = True
    show_mar_bar: bool = True
    show_alerts: bool = True
    show_stats_panel: bool = True
    line_width: int = 1

    # ── Save ───────────────────────────────────────────────
    save_annotated: bool = False
    output_dir: str = "output"


def load_config(path: str | Path | None) -> DrowsinessConfig:
    """Load configuration from YAML or JSON, falling back to defaults."""
    if path is None:
        return DrowsinessConfig()

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


def _dict_to_config(d: dict[str, Any]) -> DrowsinessConfig:
    cfg = DrowsinessConfig()
    for key, val in d.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg
