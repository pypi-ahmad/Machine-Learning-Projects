"""Configuration dataclasses for Receipt Digitizer.

Provides :class:`ReceiptConfig` with all tunables for the OCR-based
receipt parsing pipeline: OCR engine settings, preprocessing options,
field extraction patterns, validation rules, export paths, and
display options.
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
    "merchant_name",
    "date",
    "time",
    "subtotal",
    "tax",
    "total",
    "payment_method",
    "currency",
]


@dataclass
class ReceiptConfig:
    """Top-level project configuration."""

    # OCR
    ocr_lang: str = "en"
    use_gpu: bool = False
    det_db_thresh: float = 0.3
    rec_batch_num: int = 6

    # Preprocessing
    denoise: bool = True
    deskew: bool = True
    sharpen: bool = False
    binarize: bool = False
    resize_max: int = 0  # 0 = no resize

    # Parsing
    target_fields: list[str] = field(default_factory=lambda: list(STANDARD_FIELDS))
    extract_line_items: bool = True

    # Validation
    warn_missing_fields: bool = True
    required_fields: list[str] = field(
        default_factory=lambda: ["total"]
    )

    # Export
    export_json: str = ""
    export_csv: str = ""

    # Display
    show_display: bool = True
    highlight_fields: bool = True
    show_ocr_boxes: bool = False
    line_width: int = 2

    # Save
    save_annotated: bool = False
    output_dir: str = "output"


def load_config(path: str | Path | None) -> ReceiptConfig:
    """Load configuration from YAML or JSON, falling back to defaults."""
    if path is None:
        return ReceiptConfig()

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


def _dict_to_config(d: dict[str, Any]) -> ReceiptConfig:
    cfg = ReceiptConfig()
    for key, val in d.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg
