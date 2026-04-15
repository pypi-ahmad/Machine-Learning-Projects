"""Food Freshness Grader -- configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── Explicit grading labels ───────────────────────────────
FRESHNESS_GRADES = ("fresh", "stale")

PRODUCE_TYPES = (
    "apple", "banana", "bitter_gourd", "capsicum", "orange", "tomato",
)

# Full 12-class label list (alphabetical, matching ImageFolder order)
CLASS_NAMES: list[str] = sorted(
    f"{grade}_{produce}"
    for grade in FRESHNESS_GRADES
    for produce in PRODUCE_TYPES
)


def parse_label(class_name: str) -> tuple[str, str]:
    """Split a class name into (freshness_grade, produce_type).
    """Split a class name into (freshness_grade, produce_type).

    >>> parse_label("fresh_apple")
    ('fresh', 'apple')
    >>> parse_label("stale_bitter_gourd")
    ('stale', 'bitter_gourd')
    """
    """
    for grade in FRESHNESS_GRADES:
        prefix = f"{grade}_"
        if class_name.startswith(prefix):
            return grade, class_name[len(prefix):]
    return "unknown", class_name


@dataclass
class FreshnessConfig:
    """All tuneable knobs for the food freshness grading pipeline."""

    # ── Model ─────────────────────────────────────────────
    model_name: str = "resnet18"        # resnet18|resnet50|efficientnet_b0|mobilenet_v2
    num_classes: int = 12               # 6 produce × 2 freshness
    imgsz: int = 224
    device: str | None = None           # None -> auto

    # ── Training ──────────────────────────────────────────
    epochs: int = 25
    batch_size: int = 32
    lr: float = 1e-3
    val_split: float = 0.2
    num_workers: int = 4

    # ── Inference ─────────────────────────────────────────
    weights_path: str = "runs/freshness_cls/best_model.pt"
    confidence_threshold: float = 0.3   # below -> "uncertain"

    # ── Visualisation ─────────────────────────────────────
    font_scale: float = 0.7
    fresh_color: tuple[int, int, int] = (80, 200, 80)     # green BGR
    stale_color: tuple[int, int, int] = (0, 0, 220)        # red BGR
    uncertain_color: tuple[int, int, int] = (0, 180, 255)  # orange BGR
    text_color: tuple[int, int, int] = (255, 255, 255)
    grid_thumb_size: int = 180
    grid_cols: int = 4

    # ── Output ────────────────────────────────────────────
    output_dir: str = "output"

    # ── Class names (auto-populated) ──────────────────────
    class_names: list[str] = field(default_factory=lambda: list(CLASS_NAMES))

    # ── Helpers ────────────────────────────────────────────
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FreshnessConfig:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {}
        for k, v in d.items():
            if k not in valid:
                continue
            if k.endswith("_color") and isinstance(v, list):
                v = tuple(v)
            filtered[k] = v
        return cls(**filtered)


def load_config(path: str | Path | None) -> FreshnessConfig:
    """Load config from JSON or YAML, falling back to defaults."""
    if path is None:
        return FreshnessConfig()
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
    return FreshnessConfig.from_dict(data)
