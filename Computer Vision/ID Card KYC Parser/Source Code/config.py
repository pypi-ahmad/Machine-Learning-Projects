"""Configuration dataclasses for ID Card KYC Parser.

Provides :class:`IDCardConfig` with all tunables for the OCR-based
ID card parsing pipeline: card detection, perspective correction,
OCR engine settings, template selection, field extraction,
validation rules, export paths, and display options.
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
    "full_name",
    "date_of_birth",
    "id_number",
    "nationality",
    "gender",
    "expiry_date",
    "issue_date",
    "address",
    "document_type",
]


@dataclass
class IDCardConfig:
    """Top-level project configuration."""

    # Card detection
    detect_card: bool = True
    min_card_area_ratio: float = 0.05   # min card area as fraction of image
    canny_low: int = 30
    canny_high: int = 200
    approx_eps: float = 0.02           # contour approximation epsilon factor

    # Perspective correction
    rectify: bool = True
    target_width: int = 856            # standard ID card ratio ≈ 85.6 × 53.98 mm
    target_height: int = 540

    # OCR
    ocr_lang: str = "en"
    use_gpu: bool = False
    det_db_thresh: float = 0.3
    rec_batch_num: int = 6

    # Template
    template: str = "generic"          # "generic", "us_dl", "eu_id", "passport"

    # Parsing
    target_fields: list[str] = field(default_factory=lambda: list(STANDARD_FIELDS))

    # Validation
    warn_missing_fields: bool = True
    required_fields: list[str] = field(
        default_factory=lambda: ["full_name", "id_number"],
    )

    # Export
    export_json: str = ""
    export_csv: str = ""

    # Display
    show_display: bool = True
    highlight_fields: bool = True
    show_ocr_boxes: bool = False
    show_card_boundary: bool = True
    line_width: int = 2

    # Save
    save_annotated: bool = False
    save_rectified: bool = False
    output_dir: str = "output"


def load_config(path: str | Path | None) -> IDCardConfig:
    """Load configuration from YAML or JSON, falling back to defaults."""
    if path is None:
        return IDCardConfig()

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


def _dict_to_config(d: dict[str, Any]) -> IDCardConfig:
    cfg = IDCardConfig()
    for key, val in d.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg
