"""Configuration dataclasses for Prescription OCR Parser.

Provides :class:`PrescriptionConfig` with all tunables for the
PaddleOCR-based prescription reading pipeline: OCR settings,
medicine field extraction rules, validation, export, and display.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

MEDICINE_FIELDS: list[str] = [
    "medicine_name",
    "dosage",
    "frequency",
    "duration",
    "route",
    "instructions",
    "prescriber",
    "patient_name",
    "date",
]


@dataclass
class PrescriptionConfig:
    """Top-level project configuration."""

    # ── OCR ────────────────────────────────────────────────
    ocr_lang: str = "en"
    use_gpu: bool = False
    det_db_thresh: float = 0.3
    rec_batch_num: int = 6

    # ── Field extraction ───────────────────────────────────
    target_fields: list[str] = field(
        default_factory=lambda: list(MEDICINE_FIELDS),
    )
    dosage_patterns: list[str] = field(default_factory=lambda: [
        r"\d+\s*(?:mg|mcg|ml|g|iu|units?|tab(?:let)?s?|cap(?:sule)?s?)",
    ])
    frequency_keywords: list[str] = field(default_factory=lambda: [
        "once", "twice", "thrice",
        "daily", "bid", "tid", "qid", "qd",
        "every", "hourly", "hours", "hrs",
        "morning", "evening", "night", "bedtime",
        "before meals", "after meals", "with food",
        "prn", "as needed", "sos",
        "od", "bd", "tds", "qds",
    ])
    route_keywords: list[str] = field(default_factory=lambda: [
        "oral", "topical", "iv", "im", "sc", "sublingual",
        "inhaled", "rectal", "ophthalmic", "otic", "nasal",
        "by mouth", "po",
    ])
    duration_keywords: list[str] = field(default_factory=lambda: [
        "days", "weeks", "months", "day", "week", "month",
        "for", "x",
    ])
    instruction_keywords: list[str] = field(default_factory=lambda: [
        "take", "apply", "use", "inject", "inhale", "insert",
        "dissolve", "chew", "swallow", "avoid", "do not",
        "with water", "empty stomach", "discontinue", "refill",
        "as directed", "if needed",
    ])

    # ── Validation ─────────────────────────────────────────
    confidence_threshold: float = 0.50
    warn_low_confidence: bool = True
    required_fields: list[str] = field(
        default_factory=lambda: ["medicine_name"],
    )

    # ── Export ─────────────────────────────────────────────
    export_json: str = ""
    export_csv: str = ""

    # ── Display ────────────────────────────────────────────
    show_display: bool = True
    show_ocr_boxes: bool = True
    highlight_fields: bool = True
    line_width: int = 2

    # ── Save ───────────────────────────────────────────────
    save_annotated: bool = False
    output_dir: str = "output"


def load_config(path: str | Path | None) -> PrescriptionConfig:
    """Load configuration from YAML or JSON, falling back to defaults."""
    if path is None:
        return PrescriptionConfig()

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


def _dict_to_config(d: dict[str, Any]) -> PrescriptionConfig:
    cfg = PrescriptionConfig()
    for key, val in d.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg
