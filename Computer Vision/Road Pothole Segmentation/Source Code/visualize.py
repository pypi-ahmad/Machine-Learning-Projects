"""Road Pothole Segmentation — overlay and visual report rendering."""

from __future__ import annotations

import cv2
import numpy as np

from segmentation import SegmentationResult
from severity import (
    SEVERITY_MINOR,
    SEVERITY_MODERATE,
    SEVERITY_SEVERE,
    SeverityReport,
)


_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)

# Default severity colours (BGR)
_COLOURS = {
    SEVERITY_MINOR: (0, 200, 255),    # yellow
    SEVERITY_MODERATE: (0, 140, 255),  # orange
    SEVERITY_SEVERE: (0, 0, 220),      # red
}


def draw_overlay(
    image: np.ndarray,
    seg: SegmentationResult,
    report: SeverityReport,
    *,
    alpha: float = 0.45,
    colours: dict[str, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Render a pothole overlay with severity-coloured masks and labels."""
    colours = colours or _COLOURS
    canvas = image.copy()

    # Draw masks
    for assessment in report.assessments:
        inst = seg.instances[assessment.instance_id]
        color = colours.get(assessment.severity, _COLOURS[SEVERITY_MINOR])
        overlay = canvas.copy()
        overlay[inst.mask > 127] = color
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

    # Draw bounding boxes and labels
    for assessment in report.assessments:
        inst = seg.instances[assessment.instance_id]
        color = colours.get(assessment.severity, _COLOURS[SEVERITY_MINOR])
        x1, y1, x2, y2 = inst.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        label = f"{assessment.severity} ({assessment.area_px}px)"
        if assessment.area_m2 > 0:
            label += f" {assessment.area_m2:.2f}m²"
        label += f" {assessment.confidence:.0%}"

        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.45, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(canvas, label, (x1 + 2, y1 - 4), _FONT, 0.45, _WHITE, 1, cv2.LINE_AA)

    # Draw summary panel (top-left)
    _draw_summary(canvas, report)
    # Draw legend (top-right)
    _draw_legend(canvas, colours)

    return canvas


def _draw_summary(canvas: np.ndarray, report: SeverityReport) -> None:
    """Render a summary panel in the top-left corner."""
    lines = [
        f"Potholes: {report.total_count}",
        f"Condition: {report.road_condition}",
        f"Minor/Mod/Sev: {report.minor_count}/{report.moderate_count}/{report.severe_count}",
    ]
    y = 24
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, _FONT, 0.55, 1)
        cv2.rectangle(canvas, (8, y - th - 4), (16 + tw, y + 4), _BLACK, -1)
        cv2.putText(canvas, line, (12, y), _FONT, 0.55, _WHITE, 1, cv2.LINE_AA)
        y += th + 12


def _draw_legend(
    canvas: np.ndarray,
    colours: dict[str, tuple[int, int, int]],
) -> None:
    """Draw a colour legend in the top-right corner."""
    h, w = canvas.shape[:2]
    items = [
        (colours.get(SEVERITY_MINOR, _COLOURS[SEVERITY_MINOR]), "Minor"),
        (colours.get(SEVERITY_MODERATE, _COLOURS[SEVERITY_MODERATE]), "Moderate"),
        (colours.get(SEVERITY_SEVERE, _COLOURS[SEVERITY_SEVERE]), "Severe"),
    ]
    bx = w - 140
    by = 10
    for i, (color, label) in enumerate(items):
        y = by + i * 24
        cv2.rectangle(canvas, (bx, y), (bx + 16, y + 16), color, -1)
        cv2.putText(canvas, label, (bx + 22, y + 13), _FONT, 0.45, _WHITE, 1, cv2.LINE_AA)
