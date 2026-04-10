"""Configuration dataclasses for Document Layout Block Detector.

Provides :class:`LayoutConfig` with all tunables for the document-layout
detection pipeline: model, thresholds, class filters, export paths,
crop settings, and display options.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_LAYOUT_CLASSES: list[str] = [
    "title",
    "text",
    "table",
    "figure",
    "list",
    "caption",
    "header",
    "footer",
    "page-number",
    "stamp",
]


@dataclass
class LayoutConfig:
    """Top-level project configuration."""

    model: str = "yolo26m.pt"
    conf_threshold: float = 0.30
    iou_threshold: float = 0.45
    imgsz: int = 1024
    device: str = ""

    # Class filter — empty list means "report all model classes"
    target_classes: list[str] = field(default_factory=list)

    # Export
    export_json: str = ""
    save_crops: bool = False
    crops_dir: str = "output/crops"

    # Display
    show_display: bool = True
    line_width: int = 2
    label_font_scale: float = 0.5
    show_conf: bool = True

    # PDF conversion
    pdf_dpi: int = 300

    # Save
    save_annotated: bool = False
    output_dir: str = "output"


def load_config(path: str | Path | None) -> LayoutConfig:
    """Load configuration from YAML or JSON, falling back to defaults."""
    if path is None:
        return LayoutConfig()

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


def _dict_to_config(d: dict[str, Any]) -> LayoutConfig:
    cfg = LayoutConfig()
    for key, val in d.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg
