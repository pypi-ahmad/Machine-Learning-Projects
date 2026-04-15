"""Waterbody & Flood Extent Segmentation -- configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class FloodConfig:
    """All tuneable knobs for the water/flood segmentation pipeline."""

    # ── Model ──────────────────────────────────────────────
    model_name: str = "yolo26m-seg.pt"
    confidence_threshold: float = 0.30
    iou_threshold: float = 0.45
    imgsz: int = 640

    # ── Comparison mode ───────────────────────────────────
    #   morph_kernel_size   — cleanup kernel for before/after diff masks
    #   min_change_area     — ignore change blobs smaller (pixels)
    morph_kernel_size: int = 5
    min_change_area: int = 200

    # ── Visualisation (BGR) ───────────────────────────────
    water_color: tuple[int, int, int] = (230, 160, 30)    # blue-ish
    flood_new_color: tuple[int, int, int] = (0, 0, 220)   # red
    flood_receded_color: tuple[int, int, int] = (0, 200, 0)  # green
    mask_alpha: float = 0.40

    # ── Output ────────────────────────────────────────────
    output_dir: str = "output"

    # ── helpers ────────────────────────────────────────────
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FloodConfig:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {}
        for k, v in d.items():
            if k not in valid:
                continue
            if k.endswith("_color") and isinstance(v, list):
                v = tuple(v)
            filtered[k] = v
        return cls(**filtered)


def load_config(path: str | Path | None) -> FloodConfig:
    """Load config from YAML or JSON file, falling back to defaults."""
    if path is None:
        return FloodConfig()
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
    return FloodConfig.from_dict(data)
