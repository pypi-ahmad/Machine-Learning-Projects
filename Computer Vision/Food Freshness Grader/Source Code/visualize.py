"""Food Freshness Grader — visualisation helpers.

Annotate images with freshness grade, confidence, and produce type.
Build batch result grids.
"""

from __future__ import annotations

import cv2
import numpy as np

from config import FreshnessConfig
from grader import GradeResult

_GRADE_COLORS = {
    "fresh": (80, 200, 80),       # green
    "stale": (0, 0, 220),         # red
    "unknown": (0, 180, 255),     # orange
}


def annotate_image(
    image_bgr: np.ndarray,
    result: GradeResult,
    cfg: FreshnessConfig | None = None,
) -> np.ndarray:
    """Draw freshness grade, produce type, and confidence on the image."""
    cfg = cfg or FreshnessConfig()
    vis = image_bgr.copy()
    h, w = vis.shape[:2]

    grade = result.freshness.upper()
    produce = result.produce.replace("_", " ").title()
    conf = result.confidence

    # Choose colour
    if conf < cfg.confidence_threshold:
        color = cfg.uncertain_color
        grade = "UNCERTAIN"
    elif result.freshness == "fresh":
        color = cfg.fresh_color
    else:
        color = cfg.stale_color

    # Main label
    label = f"{grade}: {produce}"
    conf_text = f"{conf:.1%}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = cfg.font_scale
    thickness = 2

    (tw1, th1), _ = cv2.getTextSize(label, font, scale, thickness)
    (tw2, th2), _ = cv2.getTextSize(conf_text, font, scale * 0.8, thickness)

    # Background box
    box_h = th1 + th2 + 20
    box_w = max(tw1, tw2) + 20
    cv2.rectangle(vis, (5, 5), (5 + box_w, 5 + box_h), (0, 0, 0), -1)
    cv2.rectangle(vis, (5, 5), (5 + box_w, 5 + box_h), color, 2)

    # Text
    cv2.putText(vis, label, (12, 8 + th1),
                font, scale, color, thickness, cv2.LINE_AA)
    cv2.putText(vis, conf_text, (12, 18 + th1 + th2),
                font, scale * 0.8, cfg.text_color, thickness, cv2.LINE_AA)

    # Freshness bar
    bar_y = h - 20
    bar_w = int((w - 20) * conf)
    cv2.rectangle(vis, (10, bar_y), (w - 10, bar_y + 12), (60, 60, 60), -1)
    cv2.rectangle(vis, (10, bar_y), (10 + bar_w, bar_y + 12), color, -1)

    return vis


def make_batch_grid(
    images: list[np.ndarray],
    results: list[GradeResult],
    *,
    thumb_size: int = 180,
    cols: int = 4,
    cfg: FreshnessConfig | None = None,
) -> np.ndarray:
    """Build a grid of annotated thumbnails for batch results."""
    cfg = cfg or FreshnessConfig()
    sz = thumb_size
    pad = 4
    cell = sz + pad * 2
    row_h = cell + 36

    n = len(images)
    rows = (n + cols - 1) // cols

    canvas_w = cols * cell
    canvas_h = rows * row_h
    canvas = np.full((canvas_h, canvas_w, 3), 35, dtype=np.uint8)

    for i, (img, res) in enumerate(zip(images, results)):
        col = i % cols
        row = i // cols
        x = col * cell + pad
        y = row * row_h + pad

        # Annotated thumbnail
        annotated = annotate_image(img, res, cfg)
        thumb = _resize_pad(annotated, sz)
        _paste(canvas, thumb, x, y)

        # Grade colour for border
        if res.confidence < cfg.confidence_threshold:
            color = cfg.uncertain_color
        elif res.freshness == "fresh":
            color = cfg.fresh_color
        else:
            color = cfg.stale_color

        cv2.rectangle(canvas, (x - 1, y - 1), (x + sz, y + sz), color, 2)

        # Label below
        label = f"{res.freshness}: {res.produce} ({res.confidence:.0%})"
        ty = y + sz + pad + 14
        cv2.putText(canvas, label, (x, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, cfg.text_color, 1, cv2.LINE_AA)

    return canvas


def make_summary_bar(
    results: list[GradeResult],
    *,
    width: int = 600,
    height: int = 80,
) -> np.ndarray:
    """Draw a horizontal bar showing fresh vs stale counts."""
    n_fresh = sum(1 for r in results if r.freshness == "fresh")
    n_stale = sum(1 for r in results if r.freshness == "stale")
    total = len(results)

    canvas = np.full((height, width, 3), 35, dtype=np.uint8)

    if total == 0:
        return canvas

    bar_w = width - 40
    fresh_w = int(bar_w * n_fresh / total)

    # Fresh portion (green)
    cv2.rectangle(canvas, (20, 20), (20 + fresh_w, 55), (80, 200, 80), -1)
    # Stale portion (red)
    cv2.rectangle(canvas, (20 + fresh_w, 20), (20 + bar_w, 55), (0, 0, 220), -1)

    # Labels
    cv2.putText(canvas, f"Fresh: {n_fresh}", (20, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 200, 80), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Stale: {n_stale}", (width // 2, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"({n_fresh / total:.0%})", (20 + fresh_w // 2 - 15, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


# ── helpers ───────────────────────────────────────────────

def _resize_pad(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 35, dtype=np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _paste(canvas: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
    h, w = patch.shape[:2]
    ch, cw = canvas.shape[:2]
    if y + h > ch or x + w > cw or y < 0 or x < 0:
        return
    canvas[y:y + h, x:x + w] = patch
