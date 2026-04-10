"""Overlay renderer for ID Card KYC Parser.

Draws card boundary, OCR bounding boxes, field highlights, and a
KYC summary panel on the source image.

Usage::

    from visualize import draw_overlay
    from parser import ParseResult

    vis = draw_overlay(image, parse_result, cfg,
                       ocr_blocks=blocks, detection=det)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from card_detector import DetectionResult
from config import IDCardConfig
from parser import ParseResult

FONT = cv2.FONT_HERSHEY_SIMPLEX
OCR_BOX_COLOUR = (0, 200, 0)
CARD_COLOUR = (0, 255, 255)
FIELD_COLOURS: dict[str, tuple[int, int, int]] = {
    "full_name":      (255, 200, 50),
    "date_of_birth":  (200, 160, 255),
    "id_number":      (50, 200, 255),
    "nationality":    (100, 255, 200),
    "gender":         (180, 180, 255),
    "expiry_date":    (255, 100, 100),
    "issue_date":     (100, 255, 100),
    "address":        (200, 255, 100),
    "document_type":  (255, 200, 200),
}
DEFAULT_COLOUR = (200, 200, 200)
PANEL_BG = (30, 30, 30)


def draw_overlay(
    image: np.ndarray,
    result: ParseResult,
    cfg: IDCardConfig,
    *,
    ocr_blocks: list | None = None,
    detection: DetectionResult | None = None,
) -> np.ndarray:
    """Render annotated overlay on *image* (copy returned)."""
    vis = image.copy()

    # Card boundary
    if cfg.show_card_boundary and detection and detection.found and detection.corners is not None:
        _draw_card_boundary(vis, detection.corners, cfg)

    # OCR boxes
    if cfg.show_ocr_boxes and ocr_blocks:
        _draw_ocr_boxes(vis, ocr_blocks, cfg)

    # Field highlights
    if cfg.highlight_fields and ocr_blocks:
        _draw_field_highlights(vis, result, ocr_blocks, cfg)

    # Summary panel
    _draw_panel(vis, result, detection)

    return vis


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _draw_card_boundary(
    vis: np.ndarray,
    corners: np.ndarray,
    cfg: IDCardConfig,
) -> None:
    pts = corners.astype(np.int32)
    cv2.polylines(vis, [pts], True, CARD_COLOUR, cfg.line_width + 1)
    for pt in pts:
        cv2.circle(vis, tuple(pt), 5, CARD_COLOUR, -1)


def _draw_ocr_boxes(vis: np.ndarray, ocr_blocks: list, cfg: IDCardConfig) -> None:
    for blk in ocr_blocks:
        pts = np.array(blk.bbox, dtype=np.int32)
        cv2.polylines(vis, [pts], True, OCR_BOX_COLOUR, cfg.line_width)
        label = f"{blk.text[:25]} ({blk.confidence:.2f})"
        cv2.putText(vis, label, (pts[0][0], pts[0][1] - 4),
                    FONT, 0.30, OCR_BOX_COLOUR, 1)


def _draw_field_highlights(
    vis: np.ndarray,
    result: ParseResult,
    ocr_blocks: list,
    cfg: IDCardConfig,
) -> None:
    for name, ef in result.fields.items():
        idx = ef.source_block_idx
        if idx < 0 or idx >= len(ocr_blocks):
            continue
        blk = ocr_blocks[idx]
        pts = np.array(blk.bbox, dtype=np.int32)
        colour = FIELD_COLOURS.get(name, DEFAULT_COLOUR)

        overlay = vis.copy()
        cv2.fillPoly(overlay, [pts], colour)
        cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
        cv2.polylines(vis, [pts], True, colour, cfg.line_width)

        tag = f"{name}: {ef.value[:30]}"
        (tw, th), _ = cv2.getTextSize(tag, FONT, 0.36, 1)
        tl = (pts[0][0], pts[0][1] - 6)
        cv2.rectangle(vis, (tl[0] - 2, tl[1] - th - 2),
                      (tl[0] + tw + 2, tl[1] + 4), PANEL_BG, -1)
        cv2.putText(vis, tag, tl, FONT, 0.36, colour, 1)


def _draw_panel(
    vis: np.ndarray,
    result: ParseResult,
    detection: DetectionResult | None,
) -> None:
    h, w = vis.shape[:2]
    margin = 10
    line_h = 22

    lines: list[tuple[str, tuple[int, int, int]]] = [
        ("KYC Fields", (255, 255, 255)),
        ("-" * 28, (120, 120, 120)),
    ]

    if detection:
        status = "Detected" if detection.found else "Not detected"
        lines.append((f"Card: {status}", CARD_COLOUR if detection.found else (150, 150, 150)))

    lines.append((f"Template: {result.template_used}", (180, 180, 180)))

    field_order = [
        "document_type", "full_name", "id_number", "date_of_birth",
        "nationality", "gender", "expiry_date", "issue_date", "address",
    ]
    for fname in field_order:
        ef = result.fields.get(fname)
        if ef is None:
            continue
        colour = FIELD_COLOURS.get(fname, DEFAULT_COLOUR)
        val = ef.value if len(ef.value) <= 35 else ef.value[:32] + "..."
        lines.append((f"{fname}: {val}", colour))
        lines.append((f"  conf: {ef.confidence:.2f}", (140, 140, 140)))

    panel_h = line_h * len(lines) + margin * 2
    panel_w = 360
    x0 = w - panel_w - margin
    y0 = margin

    overlay = vis.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)

    ty = y0 + margin + line_h
    for text, colour in lines:
        cv2.putText(vis, text, (x0 + 8, ty), FONT, 0.38, colour, 1)
        ty += line_h
