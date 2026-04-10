"""Configuration dataclasses for Face Verification Attendance System.

Provides :class:`FaceAttendanceConfig` with all tunables for the
InsightFace embedding + enrollment + verification pipeline: detection,
embedding extraction, enrollment, matching, attendance logging,
display, and export settings.
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
class FaceAttendanceConfig:
    """Top-level project configuration."""

    # ── Face detection ─────────────────────────────────────
    det_confidence: float = 0.4
    det_size: tuple[int, int] = (640, 640)
    use_yolo_detector: bool = True          # prefer YOLO face; else InsightFace

    # ── Embedding extraction ───────────────────────────────
    embedding_model: str = "buffalo_l"
    embedding_dim: int = 512
    providers: list[str] = field(default_factory=lambda: [
        "CUDAExecutionProvider", "CPUExecutionProvider",
    ])

    # ── Enrollment ─────────────────────────────────────────
    gallery_dir: str = "gallery"            # where to persist gallery
    min_enrollment_images: int = 1
    max_enrollment_images: int = 10         # cap per identity
    use_mean_embedding: bool = True         # average multiple images

    # ── Matching ───────────────────────────────────────────
    similarity_threshold: float = 0.45
    unknown_label: str = "Unknown"

    # ── Attendance logging ─────────────────────────────────
    log_dir: str = "attendance_logs"
    dedup_cooldown_sec: float = 300.0       # same person won't re-log within 5 min
    session_name: str = ""                  # auto-generated if empty

    # ── Validation ─────────────────────────────────────────
    warn_no_gallery: bool = True
    warn_no_faces: bool = True
    warn_low_confidence: bool = True
    confidence_threshold: float = 0.35

    # ── Export ─────────────────────────────────────────────
    export_json: str = ""
    export_csv: str = ""

    # ── Display ────────────────────────────────────────────
    show_display: bool = True
    show_boxes: bool = True
    show_labels: bool = True
    show_confidence: bool = True
    show_attendance_panel: bool = True
    line_width: int = 2

    # ── Save ───────────────────────────────────────────────
    save_annotated: bool = False
    output_dir: str = "output"


def load_config(path: str | Path | None) -> FaceAttendanceConfig:
    """Load configuration from YAML or JSON, falling back to defaults."""
    if path is None:
        return FaceAttendanceConfig()

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


def _dict_to_config(d: dict[str, Any]) -> FaceAttendanceConfig:
    cfg = FaceAttendanceConfig()
    for key, val in d.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg
