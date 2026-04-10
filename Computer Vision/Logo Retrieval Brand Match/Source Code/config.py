"""Logo Retrieval Brand Match — configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LogoConfig:
    """All tuneable knobs for the logo retrieval pipeline."""

    # ── Embedding model ───────────────────────────────────
    backbone: str = "efficientnet_b0"   # torchvision model name
    embedding_dim: int = 1280           # depends on backbone
    imgsz: int = 224                    # input resize
    device: str | None = None           # None → auto

    # ── Detection (optional cropping) ─────────────────────
    use_detector: bool = False
    detector_model: str = "yolo26n.pt"
    detector_conf: float = 0.25

    # ── Index ─────────────────────────────────────────────
    index_path: str = "index/logo_index.npz"
    similarity_metric: str = "cosine"   # cosine | euclidean

    # ── Retrieval ─────────────────────────────────────────
    top_k: int = 5
    min_similarity: float = 0.0         # filter out below this

    # ── Visualisation ─────────────────────────────────────
    grid_cols: int = 5
    grid_thumb_size: int = 128
    border_color: tuple[int, int, int] = (0, 200, 0)   # green BGR
    text_color: tuple[int, int, int] = (255, 255, 255)

    # ── Output ────────────────────────────────────────────
    output_dir: str = "output"

    # ── Helpers ────────────────────────────────────────────
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LogoConfig:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {}
        for k, v in d.items():
            if k not in valid:
                continue
            if k.endswith("_color") and isinstance(v, list):
                v = tuple(v)
            filtered[k] = v
        return cls(**filtered)


def load_config(path: str | Path | None) -> LogoConfig:
    """Load config from JSON or YAML, falling back to defaults."""
    if path is None:
        return LogoConfig()
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
    return LogoConfig.from_dict(data)
