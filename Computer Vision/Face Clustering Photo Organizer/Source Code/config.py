"""Configuration dataclasses for Face Clustering Photo Organizer.

Provides :class:`FaceClusterConfig` with all tunables for the
InsightFace embedding + clustering pipeline: detection, embedding,
clustering, collage, export, and display settings.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


@dataclass
class FaceClusterConfig:
    """Top-level project configuration."""

    # ── Face detection ─────────────────────────────────────
    det_confidence: float = 0.4
    det_size: tuple[int, int] = (640, 640)
    use_yolo_detector: bool = True

    # ── Embedding extraction ───────────────────────────────
    embedding_model: str = "buffalo_l"
    embedding_dim: int = 512
    providers: list[str] = field(default_factory=lambda: [
        "CUDAExecutionProvider", "CPUExecutionProvider",
    ])

    # ── Clustering ─────────────────────────────────────────
    algorithm: str = "agglomerative"        # agglomerative | dbscan
    distance_threshold: float = 0.85        # for agglomerative (cosine dist)
    dbscan_eps: float = 0.55                # for DBSCAN (cosine distance)
    dbscan_min_samples: int = 2
    min_cluster_size: int = 2               # drop clusters smaller than this
    merge_threshold: float = 0.75           # merge centroids closer than this

    # ── Collage ────────────────────────────────────────────
    collage_cols: int = 5
    collage_thumb_size: int = 112           # pixels per face thumbnail
    collage_max_faces: int = 25             # max faces per collage
    collage_border: int = 2

    # ── Organizer / export ────────────────────────────────
    output_dir: str = "output"
    copy_photos: bool = True                # copy source photos into cluster dirs
    symlink_photos: bool = False            # symlink instead of copy (Unix)
    export_json: str = ""
    export_csv: str = ""

    # ── Validation ─────────────────────────────────────────
    warn_no_faces: bool = True
    warn_single_face: bool = True
    confidence_threshold: float = 0.35

    # ── Display ────────────────────────────────────────────
    show_display: bool = True
    show_boxes: bool = True
    show_labels: bool = True
    line_width: int = 2

    # ── Save ───────────────────────────────────────────────
    save_collages: bool = True
    save_manifest: bool = True


def load_config(path: str | Path | None) -> FaceClusterConfig:
    """Load configuration from YAML or JSON, falling back to defaults."""
    if path is None:
        return FaceClusterConfig()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        import yaml
        raw = yaml.safe_load(text) or {}
    else:
        raw = json.loads(text)

    return _dict_to_config(raw)


def _dict_to_config(d: dict[str, Any]) -> FaceClusterConfig:
    cfg = FaceClusterConfig()
    for key, val in d.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg
