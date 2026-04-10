"""Overlay renderer for Receipt Digitizer.

Draws OCR bounding boxes, extracted field highlights, line-item
markers, and a summary panel on the receipt image.

Usage::

    from visualize import draw_overlay
    from parser import ParseResult

    vis = draw_overlay(image, parse_result, cfg, ocr_blocks=blocks)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ReceiptConfig
from parser import ParseResult

FONT = cv2.FONT_HERSHEY_SIMPLEX
OCR_BOX_COLOUR = (0, 200, 0)
FIELD_BOX_COLOUR = (255, 200, 0)
ITEM_BOX_COLOUR = (200, 160, 0)
FIELD_TEXT_COLOUR = (255, 255, 255)
PANEL_BG = (30, 30, 30)


def draw_overlay(
    image: np.ndarray,
    result: ParseResult,
    cfg: ReceiptConfig,
    *,
    ocr_blocks: list | None = None,
) -> np.ndarray:
    """Render annotated overlay on *image* (copy returned)."""
    vis = image.copy()

    if cfg.show_ocr_boxes and ocr_blocks:
        _draw_ocr_boxes(vis, ocr_blocks, cfg)

    if cfg.highlight_fields and ocr_blocks:
        _draw_field_highlights(vis, result, ocr_blocks, cfg)

    _draw_panel(vis, result)

    return vis


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _draw_ocr_boxes(vis: np.ndarray, ocr_blocks: list, cfg: ReceiptConfig) -> None:
    for blk in ocr_blocks:
        pts = np.array(blk.bbox, dtype=np.int32)
        cv2.polylines(vis, [pts], True, OCR_BOX_COLOUR, cfg.line_width)
        label = f"{blk.text[:25]} ({blk.confidence:.2f})"
        cv2.putText(
            vis, label,
            (pts[0][0], pts[0][1] - 4),
            FONT, 0.32, OCR_BOX_COLOUR, 1,
        )


def _draw_field_highlights(
    vis: np.ndarray,
    result: ParseResult,
    ocr_blocks: list,
    cfg: ReceiptConfig,
) -> None:
    for name, ef in result.fields.items():
        idx = ef.source_block_idx
        if idx < 0 or idx >= len(ocr_blocks):
            continue
        blk = ocr_blocks[idx]
        pts = np.array(blk.bbox, dtype=np.int32)

        overlay = vis.copy()
        cv2.fillPoly(overlay, [pts], FIELD_BOX_COLOUR)
        cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
        cv2.polylines(vis, [pts], True, FIELD_BOX_COLOUR, cfg.line_width)

        tag = f"{name}: {ef.value}"
        (tw, th), _ = cv2.getTextSize(tag, FONT, 0.38, 1)
        tl = (pts[0][0], pts[0][1] - 6)
        cv2.rectangle(
            vis,
            (tl[0] - 2, tl[1] - th - 2),
            (tl[0] + tw + 2, tl[1] + 4),
            PANEL_BG, -1,
        )
        cv2.putText(vis, tag, tl, FONT, 0.38, FIELD_BOX_COLOUR, 1)


def _draw_panel(vis: np.ndarray, result: ParseResult) -> None:
    """Draw extracted-fields summary panel on the right side."""
    h, w = vis.shape[:2]
    margin = 10
    line_h = 20

    lines: list[tuple[str, tuple[int, int, int]]] = [
        ("Receipt Fields", FIELD_TEXT_COLOUR),
        ("-" * 24, (120, 120, 120)),
    ]
    for name, ef in result.fields.items():
        colour = (0, 200, 255) if ef.confidence >= 0.7 else (0, 140, 255)
        lines.append((f"{name}: {ef.value}", colour))
        lines.append((f"  conf: {ef.confidence:.2f}", (140, 140, 140)))

    if result.line_items:
        lines.append(("", FIELD_TEXT_COLOUR))
        lines.append((f"Items ({len(result.line_items)})", FIELD_TEXT_COLOUR))
        for li in result.line_items[:10]:
            qty_s = f"{li.quantity}x " if li.quantity else ""
            amt_s = li.amount or ""
            lines.append((f"  {qty_s}{li.description[:28]} {amt_s}", (180, 200, 180)))

    panel_h = line_h * len(lines) + margin * 2
    panel_w = 320
    x0 = w - panel_w - margin
    y0 = margin

    overlay = vis.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)

    ty = y0 + margin + line_h
    for text, colour in lines:
        cv2.putText(vis, text, (x0 + 8, ty), FONT, 0.38, colour, 1)
        ty += line_h
