"""Ecommerce Item Attribute Tagger — visualisation overlays.

Draws attribute labels, confidence bars, and grid summaries on
product images.

Usage::

    from visualize import draw_attributes, draw_batch_grid

    annotated = draw_attributes(image, prediction, cfg)
    grid = draw_batch_grid(results, cfg)
"""

from __future__ import annotations

import cv2
import numpy as np

from config import TaggerConfig


def draw_attributes(
    image: np.ndarray,
    prediction: dict[str, dict],
    cfg: TaggerConfig | None = None,
    *,
    source_name: str = "",
) -> np.ndarray:
    """Draw structured attribute annotations on a product image.

    The annotation panel is drawn as a sidebar to the right.
    """
    if cfg is None:
        cfg = TaggerConfig()

    h, w = image.shape[:2]
    panel_w = max(280, w)
    canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
    canvas[:, :w] = image
    canvas[:, w:] = cfg.bg_color

    y = 25
    # Title
    if source_name:
        cv2.putText(canvas, source_name[:30], (w + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cfg.text_color, 1)
        y += 25

    for attr_name, info in prediction.items():
        label = info.get("label", "?")
        conf = info.get("confidence", 0.0)
        uncertain = conf < (cfg.confidence_threshold if cfg else 0.3)

        # Attribute name
        cv2.putText(canvas, f"{attr_name}:", (w + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
        y += 18

        # Value + confidence
        val_color = (0, 140, 255) if uncertain else cfg.accent_color
        cv2.putText(canvas, f"  {label} ({conf:.0%})", (w + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, val_color, 1)

        # Confidence bar
        bar_x = w + 200
        bar_w = panel_w - 210
        cv2.rectangle(canvas, (bar_x, y - 10), (bar_x + bar_w, y), (80, 80, 80), -1)
        fill = int(bar_w * min(conf, 1.0))
        cv2.rectangle(canvas, (bar_x, y - 10), (bar_x + fill, y), val_color, -1)

        y += 22

        if y > h - 10:
            break

    return canvas


def draw_label_overlay(
    image: np.ndarray,
    prediction: dict[str, dict],
    cfg: TaggerConfig | None = None,
) -> np.ndarray:
    """Draw a compact label overlay directly on the image."""
    if cfg is None:
        cfg = TaggerConfig()

    vis = image.copy()
    y = 25

    for attr_name, info in prediction.items():
        label = info.get("label", "?")
        conf = info.get("confidence", 0.0)
        text = f"{attr_name}: {label} ({conf:.0%})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(vis, (5, y - th - 4), (tw + 10, y + 4), cfg.bg_color, -1)
        cv2.putText(vis, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, cfg.accent_color, 1)
        y += th + 10

    return vis


def draw_batch_grid(
    results: list[dict],
    cfg: TaggerConfig | None = None,
) -> np.ndarray:
    """Draw a grid overview of batch results.

    Each ``results`` item should have ``source``, ``image`` (thumbnail),
    and ``prediction`` (attribute dict).
    """
    if cfg is None:
        cfg = TaggerConfig()

    thumb = cfg.grid_thumb_size
    cols = cfg.grid_cols
    n = len(results)
    rows = (n + cols - 1) // cols

    cell_h = thumb + 60
    canvas = np.zeros((rows * cell_h, cols * thumb, 3), dtype=np.uint8)
    canvas[:] = cfg.bg_color

    for idx, r in enumerate(results):
        row, col = divmod(idx, cols)
        x0 = col * thumb
        y0 = row * cell_h

        img = r.get("image")
        if img is not None:
            img_resized = cv2.resize(img, (thumb, thumb))
            canvas[y0 : y0 + thumb, x0 : x0 + thumb] = img_resized

        pred = r.get("prediction", {})
        source = r.get("source", "")

        label_parts = []
        for attr_name in ("masterCategory", "baseColour", "usage"):
            info = pred.get(attr_name, {})
            if info:
                label_parts.append(info.get("label", "?"))

        text = " | ".join(label_parts) if label_parts else source[:20]
        cv2.putText(canvas, text, (x0 + 4, y0 + thumb + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, cfg.accent_color, 1)
        cv2.putText(canvas, source[:20] if source else "", (x0 + 4, y0 + thumb + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (140, 140, 140), 1)

    return canvas
