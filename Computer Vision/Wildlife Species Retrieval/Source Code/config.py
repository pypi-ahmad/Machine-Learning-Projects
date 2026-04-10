"""Wildlife Species Retrieval — configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class WildlifeConfig:
    """All tuneable knobs for the wildlife retrieval pipeline."""

    # ── Embedding backbone ────────────────────────────────
    backbone: str = "efficientnet_b0"
    embedding_dim: int = 1280
    imgsz: int = 224
    device: str | None = None           # None → auto

    # ── Index ─────────────────────────────────────────────
    index_path: str = "index/wildlife_index.npz"
    similarity_metric: str = "cosine"   # cosine | euclidean

    # ── Retrieval ─────────────────────────────────────────
    top_k: int = 8
    min_similarity: float = 0.0

    # ── Classifier reranking ──────────────────────────────
    enable_rerank: bool = False
    classifier_weights: str = "runs/wildlife_cls/best_model.pt"
    classifier_model: str = "resnet18"
    num_classes: int = 90               # Animal-90 species
    rerank_weight: float = 0.4          # blend: (1-w)*sim + w*cls_match

    # ── Visualisation ─────────────────────────────────────
    grid_cols: int = 4
    grid_thumb_size: int = 160
    border_color: tuple[int, int, int] = (80, 200, 80)
    match_color: tuple[int, int, int] = (0, 220, 100)
    text_color: tuple[int, int, int] = (255, 255, 255)
    font_scale: float = 0.5

    # ── Output ────────────────────────────────────────────
    output_dir: str = "output"

    # ── Helpers ────────────────────────────────────────────
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WildlifeConfig:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {}
        for k, v in d.items():
            if k not in valid:
                continue
            if k.endswith("_color") and isinstance(v, list):
                v = tuple(v)
            filtered[k] = v
        return cls(**filtered)


def load_config(path: str | Path | None) -> WildlifeConfig:
    """Load config from JSON or YAML, falling back to defaults."""
    if path is None:
        return WildlifeConfig()
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
    return WildlifeConfig.from_dict(data)
