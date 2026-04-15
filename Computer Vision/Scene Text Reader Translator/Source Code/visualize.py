"""Overlay renderer for Scene Text Reader Translator.

Draws OCR bounding boxes, text labels, confidence scores,
optional translated text, and a summary panel on scene images.

Usage::

    from visualize import draw_overlay

    vis = draw_overlay(image, scene_text_result, cfg, ocr_blocks=blocks)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import SceneTextConfig
from ocr_engine import OCRBlock
from parser import SceneTextResult

FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_BG = (30, 30, 30)
BOX_COLOUR = (0, 255, 100)
TRANSLATED_COLOUR = (255, 180, 50)
LOW_CONF_COLOUR = (0, 0, 220)
TEXT_COLOUR = (255, 255, 255)


def draw_overlay(
    image: np.ndarray,
    result: SceneTextResult,
    cfg: SceneTextConfig,
    *,
    ocr_blocks: list[OCRBlock] | None = None,
) -> np.ndarray:
    """Render annotated overlay on *image* (copy returned)."""
    vis = image.copy()

    if cfg.show_ocr_boxes:
        _draw_text_regions(vis, result, cfg)

    _draw_panel(vis, result, cfg)

    return vis


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _draw_text_regions(
    vis: np.ndarray,
    result: SceneTextResult,
    cfg: SceneTextConfig,
) -> None:
    """Draw bounding polygons and text labels for each detected region."""
    for read in result.reads:
        pts = np.array(read.bbox, dtype=np.int32)

        # Colour: red if low confidence, green otherwise
        is_low = read.confidence < cfg.confidence_threshold
        colour = LOW_CONF_COLOUR if is_low else BOX_COLOUR

        # Draw polygon
        cv2.polylines(vis, [pts], True, colour, cfg.line_width)

        # Semi-transparent fill
        overlay = vis.copy()
        cv2.fillPoly(overlay, [pts], colour)
        cv2.addWeighted(overlay, 0.10, vis, 0.90, 0, vis)

        # Text labels
        anchor_x = pts[0][0]
        anchor_y = pts[0][1] - 6

        if cfg.show_text_labels:
            label = read.text[:40]
            if cfg.show_confidence:
                label += f" ({read.confidence:.2f})"

            (tw, th), baseline = cv2.getTextSize(label, FONT, 0.38, 1)
            cv2.rectangle(
                vis,
                (anchor_x - 1, anchor_y - th - 2),
                (anchor_x + tw + 2, anchor_y + baseline + 2),
                PANEL_BG, -1,
            )
            cv2.putText(
                vis, label, (anchor_x, anchor_y),
                FONT, 0.38, colour, 1,
            )

            # Translated text (below original)
            if result.translation_enabled and read.translated_text != read.text:
                t_label = f"-> {read.translated_text[:40]}"
                t_y = anchor_y + th + baseline + 14
                (tw2, th2), _ = cv2.getTextSize(t_label, FONT, 0.34, 1)
                cv2.rectangle(
                    vis,
                    (anchor_x - 1, t_y - th2 - 2),
                    (anchor_x + tw2 + 2, t_y + 4),
                    PANEL_BG, -1,
                )
                cv2.putText(
                    vis, t_label, (anchor_x, t_y),
                    FONT, 0.34, TRANSLATED_COLOUR, 1,
                )


def _draw_panel(
    vis: np.ndarray,
    result: SceneTextResult,
    cfg: SceneTextConfig,
) -> None:
    """Draw a summary panel in the top-right corner."""
    h, w = vis.shape[:2]
    margin = 8
    line_h = 20

    lines: list[tuple[str, tuple[int, int, int]]] = [
        ("Scene Text Reader", TEXT_COLOUR),
        (f"Frame: {result.frame_index}", (180, 180, 180)),
        (f"Blocks: {result.num_blocks}", (180, 180, 180)),
        (f"Avg conf: {result.mean_confidence:.2f}", (180, 180, 180)),
    ]

    if result.translation_enabled:
        lines.append((f"Translate: {result.translation_provider}", TRANSLATED_COLOUR))

    lines.append(("-" * 28, (100, 100, 100)))

    # Show first few text reads
    for read in result.reads[:8]:
        txt = read.text[:30]
        lines.append((f"  {txt}", BOX_COLOUR))
        if result.translation_enabled and read.translated_text != read.text:
            lines.append((f"  -> {read.translated_text[:30]}", TRANSLATED_COLOUR))

    if result.num_blocks > 8:
        lines.append((f"  ... +{result.num_blocks - 8} more", (140, 140, 140)))

    panel_h = margin * 2 + line_h * len(lines)
    panel_w = 320
    x0 = w - panel_w - margin
    y0 = margin

    # Semi-transparent background
    overlay = vis.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)

    ty = y0 + margin + line_h
    for text, colour in lines:
        cv2.putText(vis, text, (x0 + 8, ty), FONT, 0.38, colour, 1)
        ty += line_h
