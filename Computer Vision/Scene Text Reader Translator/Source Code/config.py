"""Configuration dataclasses for Scene Text Reader Translator.

Provides :class:`SceneTextConfig` with all tunables for the
scene-text reading pipeline: OCR settings, translation
hook, validation, export, and display.
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
class SceneTextConfig:
    """Top-level project configuration."""

    # ── PaddleOCR ──────────────────────────────────────────
    ocr_backend: str = "auto"
    ocr_lang: str = "en"
    use_gpu: bool = False
    det_db_thresh: float = 0.3
    rec_batch_num: int = 6
    use_angle_cls: bool = True

    # ── Translation hook ───────────────────────────────────
    translate_enabled: bool = False
    translate_target_lang: str = "en"
    translate_provider: str = ""       # reserved hook label; no bundled provider

    # ── Validation ─────────────────────────────────────────
    confidence_threshold: float = 0.40
    min_text_length: int = 1
    warn_low_confidence: bool = True

    # ── Export ─────────────────────────────────────────────
    export_json: str = ""
    export_csv: str = ""

    # ── Display ────────────────────────────────────────────
    show_display: bool = True
    show_ocr_boxes: bool = True
    show_text_labels: bool = True
    show_confidence: bool = True
    line_width: int = 2

    # ── Save ───────────────────────────────────────────────
    save_annotated: bool = False
    output_dir: str = "output"

    # ── Video / webcam ─────────────────────────────────────
    webcam_index: int = 0
    video_fps: int = 0                 # 0 = use source fps


def load_config(path: str | Path | None) -> SceneTextConfig:
    """Load configuration from YAML or JSON, falling back to defaults."""
    if path is None:
        return SceneTextConfig()

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


def _dict_to_config(d: dict[str, Any]) -> SceneTextConfig:
    cfg = SceneTextConfig()
    for key, val in d.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg
