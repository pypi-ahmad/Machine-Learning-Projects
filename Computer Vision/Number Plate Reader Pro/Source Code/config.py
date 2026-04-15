"""Configuration dataclasses for Number Plate Reader Pro.

Provides :class:`PlateConfig` with all tunables for the YOLO +
PaddleOCR-first license-plate reading pipeline: detection, rectification,
OCR, regex cleanup, duplicate suppression, and export.
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
class PlateConfig:
    """Top-level project configuration."""

    # ── YOLO detection ─────────────────────────────────────
    det_model: str = "yolo26m.pt"
    det_confidence: float = 0.35
    det_iou: float = 0.45
    det_imgsz: int = 640

    # ── Plate rectification ────────────────────────────────
    rectify: bool = True
    target_width: int = 240            # standard plate crop width
    target_height: int = 80            # standard plate crop height
    min_crop_width: int = 30           # skip tiny crops
    upscale_threshold: int = 100       # upscale crop if width < this

    # ── PaddleOCR ──────────────────────────────────────────
    ocr_backend: str = "auto"
    ocr_lang: str = "en"
    use_gpu: bool = False
    det_db_thresh: float = 0.3
    rec_batch_num: int = 6

    # ── Regex cleanup ──────────────────────────────────────
    valid_plate_pattern: str = r"^[A-Z0-9\- ]{2,15}$"
    strip_chars: str = r"[^A-Z0-9\-\s]"

    # ── Duplicate suppression ──────────────────────────────
    dedup_enabled: bool = True
    dedup_cooldown: float = 5.0        # seconds between same plate reads
    dedup_max_entries: int = 200

    # ── Validation ─────────────────────────────────────────
    confidence_threshold: float = 0.50
    min_plate_length: int = 2
    max_plate_length: int = 15

    # ── Export ─────────────────────────────────────────────
    export_json: str = ""
    export_csv: str = ""

    # ── Display ────────────────────────────────────────────
    show_display: bool = True
    show_det_boxes: bool = True
    show_plate_text: bool = True
    show_confidence: bool = True
    line_width: int = 2

    # ── Save ───────────────────────────────────────────────
    save_annotated: bool = False
    save_crops: bool = False
    output_dir: str = "output"

    # ── Video / webcam ─────────────────────────────────────
    webcam_index: int = 0
    video_fps: int = 0                 # 0 = use source fps


def load_config(path: str | Path | None) -> PlateConfig:
    """Load configuration from YAML or JSON, falling back to defaults."""
    if path is None:
        return PlateConfig()

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


def _dict_to_config(d: dict[str, Any]) -> PlateConfig:
    cfg = PlateConfig()
    for key, val in d.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg
