"""Fire Area Segmentation -- configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class FireConfig:
    """All tuneable knobs for the fire/smoke segmentation pipeline."""

    # ── Model ──────────────────────────────────────────────
    model_name: str = "yolo26m-seg.pt"
    confidence_threshold: float = 0.30
    iou_threshold: float = 0.45
    imgsz: int = 640

    # ── Classes ───────────────────────────────────────────
    #   class_names maps YOLO class-id → human label.
    #   When using a pretrained fallback the model has no fire class;
    #   after fine-tuning, override via config file.
    class_names: tuple[str, ...] = ("fire", "smoke")
    enable_smoke: bool = True       # segment smoke in addition to fire

    # ── Trend window ──────────────────────────────────────
    trend_window: int = 30          # frames for rolling trend summary

    # ── Visualisation (BGR) ───────────────────────────────
    fire_color: tuple[int, int, int] = (0, 60, 255)       # orange-red
    smoke_color: tuple[int, int, int] = (180, 180, 180)    # grey
    mask_alpha: float = 0.45

    # ── Output ────────────────────────────────────────────
    output_dir: str = "output"

    # ── helpers ────────────────────────────────────────────
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FireConfig:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {}
        for k, v in d.items():
            if k not in valid:
                continue
            if k.endswith("_color") and isinstance(v, list):
                v = tuple(v)
            if k == "class_names" and isinstance(v, list):
                v = tuple(v)
            filtered[k] = v
        return cls(**filtered)


def load_config(path: str | Path | None) -> FireConfig:
    """Load config from YAML or JSON file, falling back to defaults."""
    if path is None:
        return FireConfig()
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
    return FireConfig.from_dict(data)
