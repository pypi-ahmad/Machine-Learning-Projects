"""Building Footprint Change Detector — configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ChangeConfig:
    """All tuneable knobs for the change-detection pipeline."""

    # ── Model ──────────────────────────────────────────────
    model_name: str = "yolo26m-seg.pt"
    confidence_threshold: float = 0.30
    iou_threshold: float = 0.45
    imgsz: int = 1024

    # ── Segmentation mask extraction ──────────────────────
    #   When using COCO-pretrained weights there is no "building" class;
    #   set *use_all_classes* = True to merge every detected instance mask.
    #   After fine-tuning on building data the model has a single class.
    use_all_classes: bool = True

    # ── Diff / morphological cleanup ──────────────────────
    morph_kernel_size: int = 5
    min_change_area: int = 100  # pixels – ignore change blobs smaller

    # ── Visualisation colours (BGR) ───────────────────────
    new_color: tuple[int, int, int] = (0, 200, 0)       # green
    demolished_color: tuple[int, int, int] = (0, 0, 220)  # red
    unchanged_color: tuple[int, int, int] = (200, 140, 0)  # orange
    mask_alpha: float = 0.40

    # ── Output ────────────────────────────────────────────
    output_dir: str = "output"

    # ── helpers ────────────────────────────────────────────
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChangeConfig:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


def load_config(path: str | Path | None) -> ChangeConfig:
    """Load config from a YAML or JSON file, falling back to defaults."""
    if path is None:
        return ChangeConfig()
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
    return ChangeConfig.from_dict(data)
