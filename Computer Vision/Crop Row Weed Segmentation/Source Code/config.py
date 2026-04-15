"""Crop Row & Weed Segmentation -- configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Default class names — override after fine-tuning on your dataset.
DEFAULT_CLASSES: list[str] = ["crop", "weed", "soil"]

# Class colours (BGR) for visualisation
DEFAULT_COLOURS: dict[str, tuple[int, int, int]] = {
    "crop": (0, 200, 0),       # green
    "weed": (0, 0, 220),       # red
    "soil": (140, 120, 100),   # brown
}


@dataclass
class CropWeedConfig:
    """All tuneable knobs for the crop/weed segmentation pipeline."""

    # ── Model ──────────────────────────────────────────────
    model_name: str = "yolo26m-seg.pt"
    confidence_threshold: float = 0.30
    iou_threshold: float = 0.45
    imgsz: int = 640

    # ── Classes ────────────────────────────────────────────
    class_names: list[str] = field(default_factory=lambda: list(DEFAULT_CLASSES))
    class_colours: dict[str, tuple[int, int, int]] = field(
        default_factory=lambda: dict(DEFAULT_COLOURS),
    )

    # ── Visualisation ─────────────────────────────────────
    mask_alpha: float = 0.45

    # ── Output ────────────────────────────────────────────
    output_dir: str = "output"

    # ── helpers ────────────────────────────────────────────
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CropWeedConfig:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {}
        for k, v in d.items():
            if k not in valid:
                continue
            if k == "class_colours" and isinstance(v, dict):
                v = {name: tuple(c) for name, c in v.items()}
            filtered[k] = v
        return cls(**filtered)

    def colour_for(self, class_name: str) -> tuple[int, int, int]:
        """Return the BGR colour for a class, falling back to grey."""
        return self.class_colours.get(class_name, (160, 160, 160))


def load_config(path: str | Path | None) -> CropWeedConfig:
    """Load config from YAML or JSON file, falling back to defaults."""
    if path is None:
        return CropWeedConfig()
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
    return CropWeedConfig.from_dict(data)
