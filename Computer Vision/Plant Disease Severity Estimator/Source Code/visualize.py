"""Plant Disease Severity Estimator — visualisation utilities.
"""Plant Disease Severity Estimator — visualisation utilities.

Annotates leaf images with disease label, severity badge, confidence bar,
and optional lesion-ratio gauge.  Also builds thumbnail grids for batch
results.
"""
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from classifier import PredictionResult
from config import SEVERITY_NAMES, SeverityConfig


# ── Single-image annotation ──────────────────────────────

def annotate_image(
    image_bgr: np.ndarray,
    result: PredictionResult,
    cfg: SeverityConfig | None = None,
) -> np.ndarray:
    """Draw disease label, severity badge, and confidence bar on image."""
    cfg = cfg or SeverityConfig()
    out = image_bgr.copy()
    h, w = out.shape[:2]
    fs = cfg.font_scale
    thick = max(1, int(fs * 2))
    pad = 6

    # Severity colour
    sev_color = cfg.severity_color(result.severity_index)

    # ── Severity badge (top-left) ─────────────────────────
    badge_text = result.severity_name.upper()
    (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
    cv2.rectangle(out, (0, 0), (tw + 2 * pad, th + 2 * pad), sev_color, -1)
    cv2.putText(out, badge_text, (pad, th + pad),
                cv2.FONT_HERSHEY_SIMPLEX, fs, cfg.text_color, thick)

    # ── Disease label (below badge) ───────────────────────
    label = f"{result.plant} -- {result.disease}"
    y_label = th + 3 * pad + th
    cv2.putText(out, label, (pad, y_label),
                cv2.FONT_HERSHEY_SIMPLEX, fs * 0.9, sev_color, thick)

    # ── Confidence bar (bottom) ───────────────────────────
    bar_h = max(14, int(h * 0.04))
    bar_y = h - bar_h - pad
    bar_w = int((w - 2 * pad) * result.confidence)
    cv2.rectangle(out, (pad, bar_y), (pad + bar_w, bar_y + bar_h), sev_color, -1)
    cv2.rectangle(out, (pad, bar_y), (w - pad, bar_y + bar_h), (180, 180, 180), 1)
    conf_text = f"{result.confidence:.0%}"
    cv2.putText(out, conf_text, (pad + 4, bar_y + bar_h - 3),
                cv2.FONT_HERSHEY_SIMPLEX, fs * 0.7, cfg.text_color, 1)

    # ── Lesion ratio (if available) ───────────────────────
    if result.lesion_ratio is not None:
        lr_text = f"Lesion: {result.lesion_ratio:.1%}"
        cv2.putText(out, lr_text, (pad, bar_y - pad),
                    cv2.FONT_HERSHEY_SIMPLEX, fs * 0.7, sev_color, 1)

    return out


# ── Batch grid ────────────────────────────────────────────

def make_grid(
    images: list[np.ndarray],
    results: list[PredictionResult],
    cfg: SeverityConfig | None = None,
) -> np.ndarray:
    """Build a labelled thumbnail grid from batch results."""
    cfg = cfg or SeverityConfig()
    sz = cfg.grid_thumb_size
    cols = cfg.grid_cols
    rows = max(1, (len(images) + cols - 1) // cols)

    grid = np.zeros((rows * sz, cols * sz, 3), dtype=np.uint8)

    for idx, (img, res) in enumerate(zip(images, results)):
        r, c = divmod(idx, cols)
        thumb = cv2.resize(annotate_image(img, res, cfg), (sz, sz))
        grid[r * sz:(r + 1) * sz, c * sz:(c + 1) * sz] = thumb

    return grid


def save_annotated(
    image_bgr: np.ndarray,
    result: PredictionResult,
    output_dir: str | Path,
    filename: str = "annotated.jpg",
    cfg: SeverityConfig | None = None,
) -> Path:
    """Annotate and save a single image."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated = annotate_image(image_bgr, result, cfg)
    out_path = out_dir / filename
    cv2.imwrite(str(out_path), annotated)
    return out_path


def save_grid(
    images: list[np.ndarray],
    results: list[PredictionResult],
    output_dir: str | Path,
    filename: str = "grid.jpg",
    cfg: SeverityConfig | None = None,
) -> Path:
    """Build grid and save to disk."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = make_grid(images, results, cfg)
    out_path = out_dir / filename
    cv2.imwrite(str(out_path), grid)
    return out_path
