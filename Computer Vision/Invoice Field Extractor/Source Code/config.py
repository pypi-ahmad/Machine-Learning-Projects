"""Configuration dataclasses for Invoice Field Extractor.

Provides :class:`InvoiceConfig` with all tunables for the OCR-based
invoice parsing pipeline: OCR engine settings, field extraction
patterns, validation rules, export paths, and display options.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Fields the extractor attempts to find
STANDARD_FIELDS: list[str] = [
    "vendor_name",
    "invoice_number",
    "invoice_date",
    "due_date",
    "subtotal",
    "tax",
    "total",
    "currency",
    "bill_to",
]


@dataclass
class InvoiceConfig:
    """Top-level project configuration."""

    # OCR
    ocr_lang: str = "en"
    use_gpu: bool = False
    det_db_thresh: float = 0.3
    rec_batch_num: int = 6

    # Parsing
    target_fields: list[str] = field(default_factory=lambda: list(STANDARD_FIELDS))
    extract_line_items: bool = True

    # Validation
    warn_missing_fields: bool = True
    required_fields: list[str] = field(
        default_factory=lambda: ["invoice_number", "total"]
    )

    # Export
    export_json: str = ""
    export_csv: str = ""

    # Display
    show_display: bool = True
    highlight_fields: bool = True
    show_ocr_boxes: bool = False
    line_width: int = 2

    # PDF
    pdf_dpi: int = 300

    # Save
    save_annotated: bool = False
    output_dir: str = "output"


def load_config(path: str | Path | None) -> InvoiceConfig:
    """Load configuration from YAML or JSON, falling back to defaults."""
    if path is None:
        return InvoiceConfig()

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


def _dict_to_config(d: dict[str, Any]) -> InvoiceConfig:
    cfg = InvoiceConfig()
    for key, val in d.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg
