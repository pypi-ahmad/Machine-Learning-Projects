"""Wound Area Measurement -- configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class WoundConfig:
    """All tuneable knobs for the wound segmentation pipeline."""

    # ── Model ──────────────────────────────────────────────
    model_name: str = "yolo26m-seg.pt"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    imgsz: int = 640

    # ── Visualisation (BGR) ───────────────────────────────
    wound_color: tuple[int, int, int] = (0, 0, 220)       # red
    boundary_color: tuple[int, int, int] = (0, 255, 255)   # yellow
    mask_alpha: float = 0.40

    # ── Output ────────────────────────────────────────────
    output_dir: str = "output"

    # ── helpers ────────────────────────────────────────────
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WoundConfig:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {}
        for k, v in d.items():
            if k not in valid:
                continue
            if k.endswith("_color") and isinstance(v, list):
                v = tuple(v)
            filtered[k] = v
        return cls(**filtered)


def load_config(path: str | Path | None) -> WoundConfig:
    """Load config from YAML or JSON file, falling back to defaults."""
    if path is None:
        return WoundConfig()
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix in {".yaml", ".yml"}:
        try:
            import yaml
            data = yaml.safe_load(text) or {}
        except ImportError:
            data = json.loads(text)
    else:
        data = json.loads(text)
    return WoundConfig.from_dict(data)
