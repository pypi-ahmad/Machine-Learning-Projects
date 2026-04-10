"""Ecommerce Item Attribute Tagger — configuration and attribute schema.

Defines attribute fields, label vocabularies, and all pipeline tunables.

The ``styles.csv`` from the Fashion Product Images dataset provides
these attribute columns:

- **gender** — Men, Women, Boys, Girls, Unisex
- **masterCategory** — Apparel, Accessories, Footwear, Personal Care, Free Items
- **subCategory** — Topwear, Bottomwear, Watches, Shoes, …
- **articleType** — Tshirts, Jeans, Shirts, Watches, …  (143 unique)
- **baseColour** — Black, White, Blue, Navy Blue, …  (46 unique)
- **season** — Summer, Fall, Winter, Spring
- **usage** — Casual, Ethnic, Formal, Sports, Smart Casual, …

Usage::

    from config import TaggerConfig, ATTRIBUTE_HEADS, load_config

    cfg = load_config("tagger.yaml")
    cfg = TaggerConfig()
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Attribute head definitions  (name → num_classes set during data bootstrap)
# ---------------------------------------------------------------------------
# These are the structured attributes we predict.
# Top-N values are used; rarer labels are mapped to "<other>".
ATTRIBUTE_HEADS: dict[str, dict[str, Any]] = {
    "masterCategory": {"max_classes": 7},
    "subCategory":    {"max_classes": 20},
    "articleType":    {"max_classes": 50},
    "baseColour":     {"max_classes": 20},
    "season":         {"max_classes": 5},
    "usage":          {"max_classes": 10},
    "gender":         {"max_classes": 6},
}


@dataclass
class TaggerConfig:
    """Top-level project configuration."""

    # ── Model ─────────────────────────────────────────────
    backbone: str = "resnet18"       # resnet18 | resnet50 | mobilenet_v2
    imgsz: int = 224
    device: str | None = None        # None → auto

    # ── Training ──────────────────────────────────────────
    epochs: int = 15
    batch_size: int = 64
    lr: float = 1e-3
    val_split: float = 0.15
    num_workers: int = 4
    min_class_samples: int = 20      # drop classes with fewer samples

    # ── Inference ─────────────────────────────────────────
    weights_path: str = "runs/attribute_tagger/best_model.pt"
    confidence_threshold: float = 0.3

    # ── Item detection (isolation) ────────────────────────
    use_detector: bool = False          # True = YOLO crop before tagging
    detector_model: str = "yolo26n.pt"
    detector_conf: float = 0.35

    # ── Attributes to predict ─────────────────────────────
    attribute_heads: list[str] = field(
        default_factory=lambda: list(ATTRIBUTE_HEADS.keys()),
    )

    # ── Visualisation ─────────────────────────────────────
    font_scale: float = 0.5
    text_color: tuple[int, int, int] = (255, 255, 255)
    bg_color: tuple[int, int, int] = (40, 40, 40)
    accent_color: tuple[int, int, int] = (0, 200, 120)
    grid_thumb_size: int = 160
    grid_cols: int = 5

    # ── Output ────────────────────────────────────────────
    output_dir: str = "output"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TaggerConfig:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {}
        for k, v in d.items():
            if k not in valid:
                continue
            if k.endswith("_color") and isinstance(v, list):
                v = tuple(v)
            filtered[k] = v
        return cls(**filtered)


def load_config(path: str | Path | None = None) -> TaggerConfig:
    """Load config from YAML or JSON, falling back to defaults."""
    if path is None:
        return TaggerConfig()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    text = p.read_text(encoding="utf-8")
    if p.suffix in {".yaml", ".yml"}:
        import yaml
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    return TaggerConfig.from_dict(data)
