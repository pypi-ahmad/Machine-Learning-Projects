"""Road Pothole Segmentation -- configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PotholeConfig:
    """All tuneable knobs for the pothole segmentation pipeline."""

    # ── Model ──────────────────────────────────────────────
    model_name: str = "yolo26m-seg.pt"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    imgsz: int = 640

    # ── Severity estimation ───────────────────────────────
    #   pixel_area_per_m2 controls the assumed ground-sample distance.
    #   Set to 0 to report area in pixels only.
    pixel_area_per_m2: float = 0.0
    severity_thresholds: tuple[int, int] = (1500, 6000)  # px: minor < t[0] < moderate < t[1] < severe

    # ── Visualisation (BGR) ───────────────────────────────
    mask_alpha: float = 0.45
    minor_color: tuple[int, int, int] = (0, 200, 255)    # yellow
    moderate_color: tuple[int, int, int] = (0, 140, 255)  # orange
    severe_color: tuple[int, int, int] = (0, 0, 220)      # red

    # ── Output ────────────────────────────────────────────
    output_dir: str = "output"

    # ── helpers ────────────────────────────────────────────
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PotholeConfig:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {}
        for k, v in d.items():
            if k not in valid:
                continue
            if k == "severity_thresholds" and isinstance(v, list):
                v = tuple(v)
            filtered[k] = v
        return cls(**filtered)


def load_config(path: str | Path | None) -> PotholeConfig:
    """Load config from YAML or JSON file, falling back to defaults."""
    if path is None:
        return PotholeConfig()
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
    return PotholeConfig.from_dict(data)
