"""Polyp Lesion Segmentation -- configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PolypConfig:
    """All tuneable knobs for the polyp segmentation pipeline."""

    # ── Model ──────────────────────────────────────────────
    model_name: str = "yolo26m-seg.pt"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    imgsz: int = 640

    # ── Backend ───────────────────────────────────────────
    #   "yolo" — YOLO26m-seg baseline (default)
    #   "medsam" — optional MedSAM comparison path
    backend: str = "yolo"

    # ── Visualisation (BGR) ───────────────────────────────
    polyp_color: tuple[int, int, int] = (0, 100, 255)      # orange
    boundary_color: tuple[int, int, int] = (0, 255, 0)     # green
    mask_alpha: float = 0.40

    # ── Output ────────────────────────────────────────────
    output_dir: str = "output"

    # ── helpers ────────────────────────────────────────────
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PolypConfig:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {}
        for k, v in d.items():
            if k not in valid:
                continue
            if k.endswith("_color") and isinstance(v, list):
                v = tuple(v)
            filtered[k] = v
        return cls(**filtered)


def load_config(path: str | Path | None) -> PolypConfig:
    """Load config from YAML or JSON file, falling back to defaults."""
    if path is None:
        return PolypConfig()
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
    return PolypConfig.from_dict(data)
