"""Configuration dataclasses for Exam Sheet Parser.

Provides :class:`ExamSheetConfig` with all tunables for the
layout-aware exam-sheet parsing pipeline: OCR settings,
layout detection, question extraction, validation, export, and display.
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
class ExamSheetConfig:
    """Top-level project configuration."""

    # ── OCR ────────────────────────────────────────────────
    ocr_backend: str = "auto"
    ocr_lang: str = "en"
    use_gpu: bool = False
    det_db_thresh: float = 0.3
    rec_batch_num: int = 6
    use_angle_cls: bool = True

    # ── Layout parsing ─────────────────────────────────────
    heading_min_height: int = 18       # px -- blocks taller treated as heading
    heading_font_ratio: float = 1.4    # if h > median_h * ratio -> heading
    question_number_pattern: str = r"^\s*(?:Q\.?\s*)?(\d{1,3})\s*[\.\)\:]"
    mcq_option_pattern: str = r"^\s*\(?([A-Ea-e])\s*[\.\)\:]"
    marks_pattern: str = r"\[?\(?\s*(\d{1,3})\s*(?:marks?|pts?|points?)\s*\)?\]?"
    section_keywords: list[str] = field(default_factory=lambda: [
        "section", "part", "instructions", "note",
        "answer", "total", "time",
    ])
    merge_y_tolerance: int = 12        # px -- blocks within tol on Y merged
    min_block_confidence: float = 0.25

    # ── Validation ─────────────────────────────────────────
    confidence_threshold: float = 0.40
    warn_low_confidence: bool = True
    warn_no_questions: bool = True

    # ── Export ─────────────────────────────────────────────
    export_json: str = ""
    export_csv: str = ""

    # ── Display ────────────────────────────────────────────
    show_display: bool = True
    show_ocr_boxes: bool = True
    show_labels: bool = True
    show_confidence: bool = True
    line_width: int = 2

    # ── Save ───────────────────────────────────────────────
    save_annotated: bool = False
    output_dir: str = "output"


def load_config(path: str | Path | None) -> ExamSheetConfig:
    """Load configuration from YAML or JSON, falling back to defaults."""
    if path is None:
        return ExamSheetConfig()

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


def _dict_to_config(d: dict[str, Any]) -> ExamSheetConfig:
    cfg = ExamSheetConfig()
    for key, val in d.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg
