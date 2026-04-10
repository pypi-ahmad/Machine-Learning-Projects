"""Interactive Video Object Cutout Studio — configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CutoutConfig:
    """All tuneable knobs for the SAM 2 cutout pipeline."""

    # ── Model ──────────────────────────────────────────────
    model_id: str = "facebook/sam2.1-hiera-small"
    device: str | None = None  # None → auto-detect (cuda / cpu)

    # ── Prediction ─────────────────────────────────────────
    multimask_output: bool = True
    mask_threshold: float = 0.0   # logit threshold for binary mask

    # ── Video ──────────────────────────────────────────────
    max_frames: int = 0       # 0 → all frames
    frame_stride: int = 1     # extract every Nth frame

    # ── Visualisation (BGR) ────────────────────────────────
    overlay_alpha: float = 0.45
    overlay_color: tuple[int, int, int] = (255, 144, 30)   # dodger-blue BGR
    point_fg_color: tuple[int, int, int] = (0, 255, 0)     # green
    point_bg_color: tuple[int, int, int] = (0, 0, 255)     # red
    box_color: tuple[int, int, int] = (255, 200, 0)        # cyan-ish

    # ── Output ─────────────────────────────────────────────
    output_dir: str = "output"

    # ── Helpers ────────────────────────────────────────────
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CutoutConfig:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {}
        for k, v in d.items():
            if k not in valid:
                continue
            if k.endswith("_color") and isinstance(v, list):
                v = tuple(v)
            filtered[k] = v
        return cls(**filtered)


def load_config(path: str | Path | None) -> CutoutConfig:
    """Load config from JSON or YAML, falling back to defaults."""
    if path is None:
        return CutoutConfig()
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
    return CutoutConfig.from_dict(data)
