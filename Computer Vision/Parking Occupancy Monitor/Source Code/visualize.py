"""Parking Occupancy Monitor — overlay renderer.

Draws slot polygons (coloured by occupancy), vehicle bounding boxes,
and a mini dashboard with total free/occupied counts.

Usage::

    from visualize import OverlayRenderer

    renderer = OverlayRenderer()
    annotated = renderer.draw(frame, frame_result)
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from slots import SlotStatus, FrameResult, Detection


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLOR_FREE     = (0, 200, 0)       # green
COLOR_OCCUPIED = (0, 0, 220)       # red
COLOR_VEHICLE  = (255, 180, 0)     # cyan-ish
COLOR_TEXT_BG  = (40, 40, 40)
COLOR_WHITE    = (255, 255, 255)


class OverlayRenderer:
    """Compose parking-lot overlays onto a BGR frame."""

    def __init__(self, alpha: float = 0.30) -> None:
        self.alpha = alpha

    # ---- public API --------------------------------------------------------

    def draw(
        self,
        frame: np.ndarray,
        result: FrameResult,
    ) -> np.ndarray:
        """Return a copy of *frame* with all overlays composited."""
        canvas = frame.copy()
        self._draw_slots(canvas, result.slot_statuses)
        self._draw_vehicles(canvas, result.vehicle_detections)
        self._draw_dashboard(canvas, result)
        return canvas

    # ---- slot polygons -----------------------------------------------------

    def _draw_slots(self, canvas: np.ndarray, statuses: Sequence[SlotStatus]) -> None:
        overlay = canvas.copy()
        for ss in statuses:
            if not ss.polygon:
                continue
            pts = np.array(ss.polygon, dtype=np.int32)
            colour = COLOR_OCCUPIED if ss.occupied else COLOR_FREE
            cv2.fillPoly(overlay, [pts], colour)
            cv2.polylines(canvas, [pts], True, colour, 2)

            # Slot label
            cx = int(np.mean([p[0] for p in ss.polygon]))
            cy = int(np.mean([p[1] for p in ss.polygon]))
            label = ss.name
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.putText(canvas, label, (cx - tw // 2, cy + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_WHITE, 1)

        cv2.addWeighted(overlay, self.alpha, canvas, 1 - self.alpha, 0, canvas)

    # ---- vehicle boxes -----------------------------------------------------

    def _draw_vehicles(self, canvas: np.ndarray, vehicles: Sequence[Detection]) -> None:
        for v in vehicles:
            x1, y1, x2, y2 = v.box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_VEHICLE, 2)
            label = f"{v.class_name} {v.confidence:.0%}"
            self._put_label(canvas, label, (x1, y1 - 6), COLOR_VEHICLE)

    # ---- dashboard (bottom-right summary) ----------------------------------

    def _draw_dashboard(self, canvas: np.ndarray, result: FrameResult) -> None:
        h, w = canvas.shape[:2]
        lines = [
            f"Slots: {result.total_slots}",
            f"Free: {result.free_count}",
            f"Occupied: {result.occupied_count}",
            f"Vehicles: {len(result.vehicle_detections)}",
        ]
        box_w, box_h = 170, 20 + 22 * len(lines)
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
