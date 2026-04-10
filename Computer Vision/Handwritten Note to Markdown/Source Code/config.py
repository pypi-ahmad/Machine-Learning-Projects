"""Configuration dataclasses for Handwritten Note to Markdown.

Provides :class:`NoteConfig` with all tunables for the TrOCR-based
handwriting recognition pipeline: line segmentation, OCR engine,
markdown formatting, validation, and export options.
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
class NoteConfig:
    """Top-level project configuration."""

    # ── Line segmentation ──────────────────────────────────
    enable_segmentation: bool = True
    min_line_height: int = 20          # min height (px) for a text line
    merge_gap: int = 12               # merge rows within this gap
    projection_threshold: float = 0.02 # fraction of row width to count as ink
    padding: int = 8                   # extra px above/below each line crop

    # ── TrOCR OCR engine ──────────────────────────────────
    model_name: str = "microsoft/trocr-base-handwritten"
    use_gpu: bool = False
    max_new_tokens: int = 200
    num_beams: int = 4

    # ── Markdown formatting ────────────────────────────────
    detect_headers: bool = True
    header_height_ratio: float = 1.8   # line ≥ 1.8× median → header
    detect_lists: bool = True
    list_indent_px: int = 40           # x-offset ≥ this → list item
    paragraph_gap_ratio: float = 2.0   # vertical gap ≥ 2× median → new paragraph

    # ── Language / extensibility ───────────────────────────
    language: str = "en"               # placeholder for future multilingual

    # ── Validation ─────────────────────────────────────────
    confidence_threshold: float = 0.40
    warn_low_confidence: bool = True
    min_text_length: int = 1

    # ── Export ─────────────────────────────────────────────
    export_md: str = ""
    export_txt: str = ""
    export_json: str = ""

    # ── Display ────────────────────────────────────────────
    show_display: bool = True
    show_line_boxes: bool = True
    show_confidence: bool = True
    line_width: int = 2

    # ── Save ───────────────────────────────────────────────
    save_annotated: bool = False
    output_dir: str = "output"


def load_config(path: str | Path | None) -> NoteConfig:
    """Load configuration from YAML or JSON, falling back to defaults."""
    if path is None:
        return NoteConfig()

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


def _dict_to_config(d: dict[str, Any]) -> NoteConfig:
    cfg = NoteConfig()
    for key, val in d.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg
