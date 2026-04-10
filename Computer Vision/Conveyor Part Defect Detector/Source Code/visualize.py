"""Conveyor Part Defect Detector — overlay renderer.

Draws defect bounding boxes, pass/fail banners, and a summary dashboard
onto each frame.

Usage::

    from visualize import OverlayRenderer

    renderer = OverlayRenderer()
    annotated = renderer.draw(frame, frame_result)
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from inspector import Detection, FrameResult


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLOR_PASS      = (0, 200, 0)       # green
COLOR_FAIL      = (0, 0, 220)       # red
COLOR_DEFECT    = (0, 0, 255)       # red box
COLOR_OK_BOX    = (0, 200, 0)       # green box (non-defect)
COLOR_TEXT_BG   = (40, 40, 40)
COLOR_WHITE     = (255, 255, 255)
COLOR_BANNER_OK = (0, 160, 0)
COLOR_BANNER_NG = (0, 0, 200)


class OverlayRenderer:
    """Compose inspection overlays onto a BGR frame."""

    # ---- public API --------------------------------------------------------

    def draw(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        """Return a copy of *frame* with all overlays composited."""
        canvas = frame.copy()
        self._draw_detections(canvas, result)
        self._draw_banner(canvas, result)
        self._draw_dashboard(canvas, result)
        return canvas

    # ---- defect boxes ------------------------------------------------------

    def _draw_detections(self, canvas: np.ndarray, result: FrameResult) -> None:
        defect_set = {id(d) for d in result.defects}
        for det in result.all_detections:
            is_defect = id(det) in defect_set
            colour = COLOR_DEFECT if is_defect else COLOR_OK_BOX
            x1, y1, x2, y2 = det.box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
            label = f"{det.class_name} {det.confidence:.0%}"
            self._put_label(canvas, label, (x1, y1 - 6), colour)

    # ---- pass / fail banner ------------------------------------------------

    def _draw_banner(self, canvas: np.ndarray, result: FrameResult) -> None:
        h, w = canvas.shape[:2]
        colour = COLOR_BANNER_OK if result.passed else COLOR_BANNER_NG
        text = f"  {result.verdict}  —  {result.defect_count} defect(s)"
        cv2.rectangle(canvas, (0, 0), (w, 36), colour, -1)
        cv2.putText(canvas, text, (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, COLOR_WHITE, 2)

    # ---- dashboard (bottom-right) ------------------------------------------

    def _draw_dashboard(self, canvas: np.ndarray, result: FrameResult) -> None:
        h, w = canvas.shape[:2]
        lines = [
            f"Detections: {len(result.all_detections)}",
            f"Defects: {result.defect_count}",
            f"Verdict: {result.verdict}",
        ]
        box_w, box_h = 180, 20 + 22 * len(lines)
        x0 = w - box_w - 10
        y0 = h - box_h - 10
        cv2.rectangle(canvas, (x0, y0), (x0 + box_w, y0 + box_h), COLOR_TEXT_BG, -1)
        for i, line in enumerate(lines):
            cv2.putText(canvas, line, (x0 + 8, y0 + 20 + 22 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)

    # ---- util --------------------------------------------------------------

    @staticmethod
    def _put_label(canvas: np.ndarray, text: str, org: tuple[int, int],
                   colour: tuple[int, int, int]) -> None:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        x, y = org
        cv2.rectangle(canvas, (x, y - th - 4), (x + tw + 4, y + 2), colour, -1)
        cv2.putText(canvas, text, (x + 2, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)
