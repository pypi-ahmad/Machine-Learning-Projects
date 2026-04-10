"""Overlay renderer for Form OCR Checkbox Extractor.

Draws checkbox outlines (green = checked, red = unchecked),
OCR bounding boxes, text-field highlights, and a summary panel
on the source image.

Usage::

    from visualize import draw_overlay

    vis = draw_overlay(image, parse_result, cfg,
                       ocr_blocks=blocks, controls=controls)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkbox_detector import ControlType, FormControl
from config import FormCheckboxConfig
from ocr_engine import OCRBlock
from parser import FormParseResult

FONT = cv2.FONT_HERSHEY_SIMPLEX

COLOUR_CHECKED = (0, 200, 0)       # green
COLOUR_UNCHECKED = (0, 0, 220)     # red
COLOUR_RADIO_CHECKED = (0, 220, 180)
COLOUR_RADIO_UNCHECKED = (180, 100, 0)
OCR_BOX_COLOUR = (200, 200, 0)
PANEL_BG = (30, 30, 30)
FIELD_COLOURS: dict[str, tuple[int, int, int]] = {
    "name":       (255, 200, 50),
    "date":       (200, 160, 255),
    "address":    (50, 200, 255),
    "phone":      (100, 255, 200),
    "email":      (180, 180, 255),
    "signature":  (255, 100, 100),
    "id_number":  (100, 255, 100),
}
DEFAULT_COLOUR = (200, 200, 200)


def draw_overlay(
    image: np.ndarray,
    result: FormParseResult,
    cfg: FormCheckboxConfig,
    *,
    ocr_blocks: list[OCRBlock] | None = None,
    controls: list[FormControl] | None = None,
) -> np.ndarray:
    """Render annotated overlay on *image* (copy returned)."""
    vis = image.copy()

    # Checkbox / radio outlines
    if cfg.show_checkboxes and controls:
        _draw_controls(vis, controls, cfg)

    # OCR boxes
    if cfg.show_ocr_boxes and ocr_blocks:
        _draw_ocr_boxes(vis, ocr_blocks, cfg)

    # Text-field highlights
    if cfg.highlight_fields and ocr_blocks:
        _draw_text_fields(vis, result, ocr_blocks, cfg)

    # Summary panel
    _draw_panel(vis, result)

    return vis


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _draw_controls(
    vis: np.ndarray,
    controls: list[FormControl],
    cfg: FormCheckboxConfig,
) -> None:
    for ctrl in controls:
        x, y, w, h = ctrl.bbox

        if ctrl.control_type == ControlType.RADIO:
            colour = COLOUR_RADIO_CHECKED if ctrl.state.value == "checked" else COLOUR_RADIO_UNCHECKED
            cx, cy = ctrl.centre
            r = max(w, h) // 2
            cv2.circle(vis, (cx, cy), r, colour, cfg.line_width)
            if ctrl.state.value == "checked":
                cv2.circle(vis, (cx, cy), r // 3, colour, -1)
        else:
            colour = COLOUR_CHECKED if ctrl.state.value == "checked" else COLOUR_UNCHECKED
            cv2.rectangle(vis, (x, y), (x + w, y + h), colour, cfg.line_width)
            if ctrl.state.value == "checked":
                # Draw tick mark
                p1 = (x + w // 4, y + h // 2)
                p2 = (x + w // 2, y + 3 * h // 4)
                p3 = (x + 3 * w // 4, y + h // 4)
                cv2.line(vis, p1, p2, colour, cfg.line_width)
                cv2.line(vis, p2, p3, colour, cfg.line_width)

        # State label
        state_label = ctrl.state.value
        (tw, th), _ = cv2.getTextSize(state_label, FONT, 0.35, 1)
        label_y = y - 6 if y > 20 else y + h + th + 4
        cv2.rectangle(
            vis, (x - 1, label_y - th - 2), (x + tw + 3, label_y + 3),
            PANEL_BG, -1,
        )
        cv2.putText(vis, state_label, (x, label_y), FONT, 0.35, colour, 1)


def _draw_ocr_boxes(
    vis: np.ndarray, ocr_blocks: list[OCRBlock], cfg: FormCheckboxConfig,
) -> None:
    for blk in ocr_blocks:
        pts = np.array(blk.bbox, dtype=np.int32)
        cv2.polylines(vis, [pts], True, OCR_BOX_COLOUR, cfg.line_width)
        label = f"{blk.text[:25]} ({blk.confidence:.2f})"
        cv2.putText(
            vis, label, (pts[0][0], pts[0][1] - 4),
            FONT, 0.30, OCR_BOX_COLOUR, 1,
        )


def _draw_text_fields(
    vis: np.ndarray,
    result: FormParseResult,
    ocr_blocks: list[OCRBlock],
    cfg: FormCheckboxConfig,
) -> None:
    for name, tf in result.text_fields.items():
        idx = tf.source_block_idx
        if idx < 0 or idx >= len(ocr_blocks):
            continue
        blk = ocr_blocks[idx]
        pts = np.array(blk.bbox, dtype=np.int32)
        colour = FIELD_COLOURS.get(name, DEFAULT_COLOUR)

        overlay = vis.copy()
        cv2.fillPoly(overlay, [pts], colour)
        cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
        cv2.polylines(vis, [pts], True, colour, cfg.line_width)

        tag = f"{name}: {tf.value[:30]}"
        (tw, th), _ = cv2.getTextSize(tag, FONT, 0.36, 1)
        tl = (pts[0][0], pts[0][1] - 6)
        cv2.rectangle(
            vis, (tl[0] - 2, tl[1] - th - 2),
            (tl[0] + tw + 2, tl[1] + 4), PANEL_BG, -1,
        )
        cv2.putText(vis, tag, tl, FONT, 0.36, colour, 1)


def _draw_panel(vis: np.ndarray, result: FormParseResult) -> None:
    h, w = vis.shape[:2]
    margin = 10
    line_h = 22

    lines: list[tuple[str, tuple[int, int, int]]] = [
        ("Form Fields", (255, 255, 255)),
        ("-" * 28, (120, 120, 120)),
        (f"OCR blocks: {result.num_ocr_blocks}", (180, 180, 180)),
        (
            f"Checkboxes: {result.num_checkboxes} "
            f"({result.num_checked} checked)",
            COLOUR_CHECKED,
        ),
    ]

    # Text fields
    for name, tf in result.text_fields.items():
        colour = FIELD_COLOURS.get(name, DEFAULT_COLOUR)
        val = tf.value if len(tf.value) <= 30 else tf.value[:27] + "..."
        lines.append((f"{name}: {val}", colour))

    # Checkbox fields (up to 12 to avoid overflow)
    lines.append(("-" * 28, (120, 120, 120)))
    for cb in result.checkbox_fields[:12]:
        icon = "[X]" if cb.state == "checked" else "[ ]"
        if cb.control_type == "radio":
            icon = "(o)" if cb.state == "checked" else "( )"
        lbl = cb.label[:25] if cb.label else "(unlabelled)"
        colour = COLOUR_CHECKED if cb.state == "checked" else COLOUR_UNCHECKED
        lines.append((f"{icon} {lbl}", colour))

    if len(result.checkbox_fields) > 12:
        lines.append(
            (f"  ... +{len(result.checkbox_fields) - 12} more", (140, 140, 140))
        )

    panel_h = line_h * len(lines) + margin * 2
    panel_w = 340
    x0 = w - panel_w - margin
    y0 = margin

    overlay = vis.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)

    ty = y0 + margin + line_h
    for text, colour in lines:
        cv2.putText(vis, text, (x0 + 8, ty), FONT, 0.38, colour, 1)
        ty += line_h
