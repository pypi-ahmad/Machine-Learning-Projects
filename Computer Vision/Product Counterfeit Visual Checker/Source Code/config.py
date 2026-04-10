"""Product Counterfeit Visual Checker — configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CounterfeitConfig:
    """All tuneable knobs for the visual screening pipeline.

    NOTE: This tool performs *screening* — it flags visual mismatch risk
    relative to approved references.  It does **not** prove or disprove
    counterfeit status.
    """

    # ── Embedding backbone ────────────────────────────────
    backbone: str = "efficientnet_b0"   # torchvision model name
    embedding_dim: int = 1280           # depends on backbone
    imgsz: int = 224                    # input resize
    device: str | None = None           # None → auto

    # ── Reference store ───────────────────────────────────
    reference_path: str = "references/product_refs.npz"

    # ── Comparison ────────────────────────────────────────
    similarity_metric: str = "cosine"       # cosine | euclidean
    global_weight: float = 0.6              # weight of global embedding score
    region_weight: float = 0.25             # weight of region-patch score
    histogram_weight: float = 0.15          # weight of colour-histogram score
    region_grid: tuple[int, int] = (3, 3)   # rows × cols for region patches
    histogram_bins: int = 64                # bins per channel for histograms

    # ── Risk thresholds ───────────────────────────────────
    high_risk_threshold: float = 0.55       # below → high mismatch risk
    medium_risk_threshold: float = 0.75     # below → medium mismatch risk
    # above medium → low mismatch risk

    # ── Retrieval ─────────────────────────────────────────
    top_k: int = 3                          # reference matches to return
    min_similarity: float = 0.0

    # ── Visualisation ─────────────────────────────────────
    grid_cols: int = 4
    grid_thumb_size: int = 160
    border_color_ok: tuple[int, int, int] = (80, 200, 80)      # green
    border_color_warn: tuple[int, int, int] = (0, 180, 255)     # orange
    border_color_risk: tuple[int, int, int] = (0, 0, 220)       # red
    text_color: tuple[int, int, int] = (255, 255, 255)

    # ── Output ────────────────────────────────────────────
    output_dir: str = "output"

    # ── Helpers ────────────────────────────────────────────
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CounterfeitConfig:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {}
        for k, v in d.items():
            if k not in valid:
                continue
            if k.endswith("_color") and isinstance(v, list):
                v = tuple(v)
            elif k == "region_grid" and isinstance(v, list):
                v = tuple(v)
            filtered[k] = v
        return cls(**filtered)


def load_config(path: str | Path | None) -> CounterfeitConfig:
    """Load config from JSON or YAML, falling back to defaults."""
    if path is None:
        return CounterfeitConfig()
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
    return CounterfeitConfig.from_dict(data)
