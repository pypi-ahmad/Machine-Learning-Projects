"""Overlay renderer for Exam Sheet Parser.

Draws colour-coded bounding boxes by structural role (heading,
question, MCQ option, marks, body), text labels, and a summary
panel on exam sheet images.

Usage::

    from visualize import draw_overlay

    vis = draw_overlay(image, result, cfg, elements=elements)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ExamSheetConfig
from layout_parser import (
    LayoutElement,
    ROLE_BODY,
    ROLE_HEADING,
    ROLE_MARKS,
    ROLE_MCQ_OPTION,
    ROLE_QUESTION,
    ROLE_SECTION,
)
from parser import ExamSheetResult

FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_BG = (30, 30, 30)
TEXT_COLOUR = (255, 255, 255)

# Role → colour mapping (BGR)
ROLE_COLOURS: dict[str, tuple[int, int, int]] = {
    ROLE_HEADING:    (255, 160, 50),    # orange
    ROLE_SECTION:    (100, 255, 200),   # teal
    ROLE_QUESTION:   (0, 255, 100),     # green
    ROLE_MCQ_OPTION: (200, 160, 255),   # lavender
    ROLE_MARKS:      (50, 200, 255),    # yellow
    ROLE_BODY:       (180, 180, 180),   # grey
}

LOW_CONF_COLOUR = (0, 0, 220)


def draw_overlay(
    image: np.ndarray,
    result: ExamSheetResult,
    cfg: ExamSheetConfig,
    *,
    elements: list[LayoutElement] | None = None,
) -> np.ndarray:
    """Render annotated overlay on *image* (copy returned)."""
    vis = image.copy()

    if cfg.show_ocr_boxes and elements:
        _draw_elements(vis, elements, cfg)

    _draw_panel(vis, result)

    return vis


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _draw_elements(
    vis: np.ndarray,
    elements: list[LayoutElement],
    cfg: ExamSheetConfig,
) -> None:
    """Draw colour-coded polygons and role labels for each block."""
    for elem in elements:
        pts = np.array(elem.bbox, dtype=np.int32)

        is_low = elem.confidence < cfg.confidence_threshold
        colour = LOW_CONF_COLOUR if is_low else ROLE_COLOURS.get(
            elem.role, (180, 180, 180),
        )

        # Draw polygon
        cv2.polylines(vis, [pts], True, colour, cfg.line_width)

        # Semi-transparent fill
        overlay = vis.copy()
        cv2.fillPoly(overlay, [pts], colour)
        cv2.addWeighted(overlay, 0.08, vis, 0.92, 0, vis)

        if cfg.show_labels:
            # Build label
            parts: list[str] = [elem.role.upper()]
            if elem.question_number is not None:
                parts[0] = f"Q{elem.question_number}"
            elif elem.option_letter:
                parts[0] = f"OPT {elem.option_letter}"
            elif elem.marks_value is not None:
                parts[0] = f"{elem.marks_value}m"

            if cfg.show_confidence:
                parts.append(f"{elem.confidence:.2f}")

            label = " | ".join(parts)

            anchor_x = pts[0][0]
            anchor_y = pts[0][1] - 5

            (tw, th), baseline = cv2.getTextSize(label, FONT, 0.34, 1)
            cv2.rectangle(
                vis,
                (anchor_x - 1, anchor_y - th - 2),
                (anchor_x + tw + 2, anchor_y + baseline + 2),
                PANEL_BG, -1,
            )
            cv2.putText(
                vis, label, (anchor_x, anchor_y),
                FONT, 0.34, colour, 1,
            )


def _draw_panel(vis: np.ndarray, result: ExamSheetResult) -> None:
    """Draw a summary panel in the top-right corner."""
    h, w = vis.shape[:2]
    margin = 8
    line_h = 20

    lines: list[tuple[str, tuple[int, int, int]]] = [
        ("Exam Sheet Parser", TEXT_COLOUR),
        (f"OCR blocks: {result.num_blocks}", (180, 180, 180)),
        (f"Questions: {result.num_questions}", ROLE_COLOURS[ROLE_QUESTION]),
        (f"Avg conf: {result.mean_confidence:.2f}", (180, 180, 180)),
    ]

    if result.total_marks is not None:
        lines.append(
            (f"Total marks: {result.total_marks}", ROLE_COLOURS[ROLE_MARKS]),
        )

    if result.headings:
        lines.append(("-" * 28, (100, 100, 100)))
        for hd in result.headings[:3]:
            lines.append((f"  H: {hd[:30]}", ROLE_COLOURS[ROLE_HEADING]))

    if result.questions:
        lines.append(("-" * 28, (100, 100, 100)))
        for q in result.questions[:8]:
            marks_str = f" [{q.marks}m]" if q.marks is not None else ""
            opts = f" ({len(q.options)} opts)" if q.options else ""
            lines.append(
                (f"  Q{q.number}{marks_str}{opts}", ROLE_COLOURS[ROLE_QUESTION]),
            )
        if result.num_questions > 8:
            lines.append(
                (f"  ... +{result.num_questions - 8} more", (140, 140, 140)),
            )

    panel_h = margin * 2 + line_h * len(lines)
    panel_w = 340
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
