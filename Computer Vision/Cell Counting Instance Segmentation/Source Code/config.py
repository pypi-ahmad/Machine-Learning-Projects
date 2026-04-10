"""Cell Counting Instance Segmentation — configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CellConfig:
    """All tuneable knobs for the cell counting / segmentation pipeline."""

    # ── Model ──────────────────────────────────────────────
    model_name: str = "yolo26m-seg.pt"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    imgsz: int = 640

    # ── Counting post-processing ──────────────────────────
    #   min_area_px : discard detections smaller than this
    #   merge_overlap : IoU above which two masks are merged
    #   watershed_split : attempt watershed split on large blobs
    min_area_px: int = 64
    merge_overlap: float = 0.60
    watershed_split: bool = True

    # ── Visualisation (BGR) ───────────────────────────────
    cell_color: tuple[int, int, int] = (0, 220, 100)       # green
    boundary_color: tuple[int, int, int] = (255, 180, 0)   # cyan-ish
    centroid_color: tuple[int, int, int] = (0, 0, 255)     # red
    mask_alpha: float = 0.35

    # ── Output ────────────────────────────────────────────
    output_dir: str = "output"

    # ── helpers ────────────────────────────────────────────
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CellConfig:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {}
        for k, v in d.items():
            if k not in valid:
                continue
            if k.endswith("_color") and isinstance(v, list):
                v = tuple(v)
            filtered[k] = v
        return cls(**filtered)


def load_config(path: str | Path | None) -> CellConfig:
    """Load config from YAML or JSON file, falling back to defaults."""
    if path is None:
        return CellConfig()
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
    return CellConfig.from_dict(data)
