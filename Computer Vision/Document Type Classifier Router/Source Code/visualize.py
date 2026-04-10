"""Document Type Classifier Router — visualisation utilities.

Annotates document images with type badge, route label, and confidence
bar.  Also builds thumbnail grids for batch results.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from classifier import ClassificationResult
from config import RouterConfig
from router import RoutingDecision


def annotate_image(
    image_bgr: np.ndarray,
    cls_result: ClassificationResult,
    route: RoutingDecision,
    cfg: RouterConfig | None = None,
) -> np.ndarray:
    """Draw type badge, route label, and confidence bar on image."""
    cfg = cfg or RouterConfig()
    out = image_bgr.copy()
    h, w = out.shape[:2]
    fs = cfg.font_scale
    thick = max(1, int(fs * 2))
    pad = 6

    # ── Type badge (top-left) ─────────────────────────────
    badge = cls_result.display_label.upper()
    (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
    cv2.rectangle(out, (0, 0), (tw + 2 * pad, th + 2 * pad), cfg.badge_color, -1)
    cv2.putText(out, badge, (pad, th + pad),
                cv2.FONT_HERSHEY_SIMPLEX, fs, cfg.text_color, thick)

    # ── Route label (below badge) ─────────────────────────
    route_text = f"→ {route.pipeline}"
    y_route = th + 3 * pad + th
    cv2.putText(out, route_text, (pad, y_route),
                cv2.FONT_HERSHEY_SIMPLEX, fs * 0.85, cfg.route_color, thick)

    # ── Confidence bar (bottom) ───────────────────────────
    bar_h = max(14, int(h * 0.04))
    bar_y = h - bar_h - pad
    bar_w = int((w - 2 * pad) * cls_result.confidence)
    bar_color = cfg.badge_color if route.routed else (0, 0, 180)
    cv2.rectangle(out, (pad, bar_y), (pad + bar_w, bar_y + bar_h), bar_color, -1)
    cv2.rectangle(out, (pad, bar_y), (w - pad, bar_y + bar_h), (180, 180, 180), 1)
    conf_text = f"{cls_result.confidence:.0%}"
    cv2.putText(out, conf_text, (pad + 4, bar_y + bar_h - 3),
                cv2.FONT_HERSHEY_SIMPLEX, fs * 0.7, cfg.text_color, 1)

    return out


def make_grid(
    images: list[np.ndarray],
    cls_results: list[ClassificationResult],
    routes: list[RoutingDecision],
    cfg: RouterConfig | None = None,
) -> np.ndarray:
    """Build a labelled thumbnail grid from batch results."""
    cfg = cfg or RouterConfig()
    sz = cfg.grid_thumb_size
    cols = cfg.grid_cols
    rows = max(1, (len(images) + cols - 1) // cols)

    grid = np.zeros((rows * sz, cols * sz, 3), dtype=np.uint8)

    for idx, (img, cr, rt) in enumerate(zip(images, cls_results, routes)):
        r, c = divmod(idx, cols)
        thumb = cv2.resize(annotate_image(img, cr, rt, cfg), (sz, sz))
        grid[r * sz:(r + 1) * sz, c * sz:(c + 1) * sz] = thumb

    return grid


def save_annotated(
    image_bgr: np.ndarray,
    cls_result: ClassificationResult,
    route: RoutingDecision,
    output_dir: str | Path,
    filename: str = "annotated.jpg",
    cfg: RouterConfig | None = None,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated = annotate_image(image_bgr, cls_result, route, cfg)
    out_path = out_dir / filename
    cv2.imwrite(str(out_path), annotated)
    return out_path


def save_grid(
    images: list[np.ndarray],
    cls_results: list[ClassificationResult],
    routes: list[RoutingDecision],
    output_dir: str | Path,
    filename: str = "grid.jpg",
    cfg: RouterConfig | None = None,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = make_grid(images, cls_results, routes, cfg)
    out_path = out_dir / filename
    cv2.imwrite(str(out_path), grid)
    return out_path
