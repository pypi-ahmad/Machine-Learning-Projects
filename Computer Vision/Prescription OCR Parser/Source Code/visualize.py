"""Overlay renderer for Prescription OCR Parser.

Draws OCR bounding boxes, colour-coded medicine/field highlights,
and a prescription summary panel on the source image.

Usage::

    from visualize import draw_overlay

    vis = draw_overlay(image, parse_result, cfg, ocr_blocks=blocks)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PrescriptionConfig
from ocr_engine import OCRBlock
from parser import PrescriptionResult

FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_BG = (30, 30, 30)
OCR_BOX_COLOUR = (200, 200, 0)

MEDICINE_COLOUR = (255, 160, 50)
DOSAGE_COLOUR = (50, 200, 255)
FREQUENCY_COLOUR = (200, 160, 255)
HEADER_COLOUR = (100, 255, 200)
INSTRUCTION_COLOUR = (180, 255, 100)
LOW_CONF_COLOUR = (0, 0, 220)
DEFAULT_COLOUR = (200, 200, 200)


def draw_overlay(
    image: np.ndarray,
    result: PrescriptionResult,
    cfg: PrescriptionConfig,
    *,
    ocr_blocks: list[OCRBlock] | None = None,
) -> np.ndarray:
    """Render annotated overlay on *image* (copy returned)."""
    vis = image.copy()

    if cfg.show_ocr_boxes and ocr_blocks:
        _draw_ocr_boxes(vis, ocr_blocks, cfg)

    if cfg.highlight_fields and ocr_blocks:
        _draw_medicine_highlights(vis, result, ocr_blocks, cfg)

    _draw_panel(vis, result)

    return vis


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _draw_ocr_boxes(
    vis: np.ndarray, blocks: list[OCRBlock], cfg: PrescriptionConfig,
) -> None:
    for blk in blocks:
        pts = np.array(blk.bbox, dtype=np.int32)
        cv2.polylines(vis, [pts], True, OCR_BOX_COLOUR, cfg.line_width)
        label = f"{blk.text[:25]} ({blk.confidence:.2f})"
        cv2.putText(
            vis, label, (pts[0][0], pts[0][1] - 4),
            FONT, 0.30, OCR_BOX_COLOUR, 1,
        )


def _draw_medicine_highlights(
    vis: np.ndarray,
    result: PrescriptionResult,
    blocks: list[OCRBlock],
    cfg: PrescriptionConfig,
) -> None:
    # Collect block indices used by medicines
    med_blocks: dict[int, tuple[str, tuple[int, int, int]]] = {}
    for med in result.medicines:
        for bi in med.source_blocks:
            if bi < len(blocks):
                colour = MEDICINE_COLOUR if bi == med.source_blocks[0] else DOSAGE_COLOUR
                tag = med.medicine_name[:20] if bi == med.source_blocks[0] else "detail"
                med_blocks[bi] = (tag, colour)

    # Header fields
    for name, ef in result.header_fields.items():
        if 0 <= ef.source_block_idx < len(blocks):
            med_blocks[ef.source_block_idx] = (name, HEADER_COLOUR)

    for bi, (tag, colour) in med_blocks.items():
        blk = blocks[bi]
        pts = np.array(blk.bbox, dtype=np.int32)

        overlay = vis.copy()
        cv2.fillPoly(overlay, [pts], colour)
        cv2.addWeighted(overlay, 0.20, vis, 0.80, 0, vis)
        cv2.polylines(vis, [pts], True, colour, cfg.line_width)

        (tw, th), _ = cv2.getTextSize(tag, FONT, 0.34, 1)
        tl = (pts[0][0], pts[0][1] - 6)
        cv2.rectangle(
            vis, (tl[0] - 2, tl[1] - th - 2),
            (tl[0] + tw + 2, tl[1] + 4), PANEL_BG, -1,
        )
        cv2.putText(vis, tag, tl, FONT, 0.34, colour, 1)


def _draw_panel(vis: np.ndarray, result: PrescriptionResult) -> None:
    h, w = vis.shape[:2]
    margin = 10
    line_h = 22

    lines: list[tuple[str, tuple[int, int, int]]] = [
        ("Prescription OCR", (255, 255, 255)),
        ("(Informational Only)", (100, 100, 255)),
        ("-" * 30, (120, 120, 120)),
        (f"OCR blocks: {result.num_blocks}", (180, 180, 180)),
        (f"Medicines: {result.num_medicines}", MEDICINE_COLOUR),
    ]

    # Header fields
    for name, ef in result.header_fields.items():
        val = ef.value if len(ef.value) <= 25 else ef.value[:22] + "..."
        lines.append((f"{name}: {val}", HEADER_COLOUR))

    lines.append(("-" * 30, (120, 120, 120)))

    # Medicine entries (up to 8)
    for med in result.medicines[:8]:
        name_str = med.medicine_name[:25]
        lines.append((f"Rx: {name_str}", MEDICINE_COLOUR))
        if med.dosage:
            lines.append((f"  dose: {med.dosage[:20]}", DOSAGE_COLOUR))
        if med.frequency:
            lines.append((f"  freq: {med.frequency[:20]}", FREQUENCY_COLOUR))
        if med.instructions:
            lines.append((f"  note: {med.instructions[:20]}", INSTRUCTION_COLOUR))

    if result.num_medicines > 8:
        lines.append(
            (f"  ... +{result.num_medicines - 8} more", (140, 140, 140)),
        )

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
