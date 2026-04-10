"""Visual Anomaly Detector — configuration.

Provides :class:`AnomalyConfig` with all tunables for the anomaly
detection pipeline: backbone, thresholds, heatmap settings, and output.

Usage::

    from config import AnomalyConfig, load_config

    cfg = load_config("anomaly.yaml")
    cfg = AnomalyConfig()        # defaults
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AnomalyConfig:
    """Top-level project configuration."""

    # Feature extractor
    backbone: str = "resnet18"          # resnet18 | resnet50 | wide_resnet50_2
    feature_dim: int = 512              # auto-set during training
    imgsz: int = 224

    # Scoring
    scoring_method: str = "mahalanobis"  # mahalanobis | knn | combined
    knn_k: int = 5
    regularization: float = 1e-5        # cov regularization

    # Threshold
    anomaly_threshold: float = 3.0      # Mahalanobis distance threshold
    auto_threshold: bool = True         # auto-select from validation data
    auto_threshold_percentile: float = 95.0  # percentile on normal scores

    # Heatmap
    patch_size: int = 64
    patch_stride: int = 32
    heatmap_alpha: float = 0.4          # overlay transparency

    # Output
    output_dir: str = "output"
    model_save_path: str = "runs/anomaly_model.npz"

    # Display
    border_width: int = 4
    normal_color: tuple[int, int, int] = (0, 255, 0)      # green BGR
    anomaly_color: tuple[int, int, int] = (0, 0, 255)     # red BGR

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AnomalyConfig:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {}
        for k, v in d.items():
            if k not in valid:
                continue
            if k.endswith("_color") and isinstance(v, list):
                v = tuple(v)
            filtered[k] = v
        return cls(**filtered)


def load_config(path: str | Path | None = None) -> AnomalyConfig:
    """Load configuration from YAML or JSON, falling back to defaults."""
    if path is None:
        return AnomalyConfig()

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")

    text = p.read_text(encoding="utf-8")
    if p.suffix in {".yaml", ".yml"}:
        import yaml
        raw = yaml.safe_load(text) or {}
    else:
        raw = json.loads(text)

    return AnomalyConfig.from_dict(raw)
