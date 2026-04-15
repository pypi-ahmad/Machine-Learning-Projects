"""Industrial Scratch / Crack Segmentation -- configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DefectConfig:
    """All tuneable knobs for the surface-defect segmentation pipeline."""

    # ── Model ──────────────────────────────────────────────
    model_name: str = "yolo26m-seg.pt"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    imgsz: int = 640

    # ── Severity heuristics ───────────────────────────────
    #   min_area_px : discard detections below this area
    #   severity_thresholds : (low, medium, high) in fractional coverage
    min_area_px: int = 32
    severity_low: float = 0.005        # ≤ 0.5 %  -> low
    severity_medium: float = 0.02      # ≤ 2.0 %  -> medium  (else high)

    # ── Visualisation (BGR) ───────────────────────────────
    scratch_color: tuple[int, int, int] = (0, 0, 255)       # red
    crack_color: tuple[int, int, int] = (0, 140, 255)       # orange
    boundary_color: tuple[int, int, int] = (0, 255, 255)    # yellow
    mask_alpha: float = 0.40

    # ── Output ────────────────────────────────────────────
    output_dir: str = "output"

    # ── helpers ────────────────────────────────────────────
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DefectConfig:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {}
        for k, v in d.items():
            if k not in valid:
                continue
            if k.endswith("_color") and isinstance(v, list):
                v = tuple(v)
            filtered[k] = v
        return cls(**filtered)


def load_config(path: str | Path | None) -> DefectConfig:
    """Load config from YAML or JSON file, falling back to defaults."""
    if path is None:
        return DefectConfig()
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
    return DefectConfig.from_dict(data)
