"""Configuration dataclasses for Form OCR Checkbox Extractor.

Provides :class:`FormCheckboxConfig` with all tunables for the
checkbox/radio detection and OCR-based form extraction pipeline.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

STANDARD_FIELDS: list[str] = [
    "name",
    "date",
    "address",
    "phone",
    "email",
    "signature",
    "id_number",
]


@dataclass
class FormCheckboxConfig:
    """Top-level project configuration."""

    # ── Checkbox / radio detection (OpenCV) ────────────────
    checkbox_min_size: int = 12        # min side length (px)
    checkbox_max_size: int = 60        # max side length (px)
    checkbox_aspect_lo: float = 0.7    # min aspect ratio (w/h)
    checkbox_aspect_hi: float = 1.4    # max aspect ratio
    fill_threshold: float = 0.35       # pixel-fill ratio → checked
    radio_circularity: float = 0.75    # min circularity for radio buttons
    adaptive_block_size: int = 25      # adaptive-threshold block size
    adaptive_c: int = 10              # constant subtracted in adaptive thresh
    morph_kernel_size: int = 3         # morphological kernel size

    # ── OCR ────────────────────────────────────────────────
    ocr_lang: str = "en"
    use_gpu: bool = False
    det_db_thresh: float = 0.3
    rec_batch_num: int = 6

    # ── Label association ──────────────────────────────────
    label_max_distance: int = 300      # max px distance to associate label
    label_prefer_direction: str = "right"  # "right" | "left" | "nearest"

    # ── Parsing ────────────────────────────────────────────
    target_fields: list[str] = field(default_factory=lambda: list(STANDARD_FIELDS))

    # ── Validation ─────────────────────────────────────────
    warn_missing_fields: bool = True
    required_fields: list[str] = field(
        default_factory=lambda: ["name"],
    )
    confidence_threshold: float = 0.55

    # ── Export ─────────────────────────────────────────────
    export_json: str = ""
    export_csv: str = ""

    # ── Display ────────────────────────────────────────────
    show_display: bool = True
    show_checkboxes: bool = True
    show_ocr_boxes: bool = False
    highlight_fields: bool = True
    line_width: int = 2

    # ── Save ───────────────────────────────────────────────
    save_annotated: bool = False
    output_dir: str = "output"


def load_config(path: str | Path | None) -> FormCheckboxConfig:
    """Load configuration from YAML or JSON, falling back to defaults."""
    if path is None:
        return FormCheckboxConfig()

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


def _dict_to_config(d: dict[str, Any]) -> FormCheckboxConfig:
    cfg = FormCheckboxConfig()
    for key, val in d.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg
