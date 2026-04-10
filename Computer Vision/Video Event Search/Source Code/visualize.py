"""Video Event Search — visualisation overlays.

Draws zones, virtual lines, bounding boxes with track IDs, movement
trails, and an event log sidebar on annotated frames.

Usage::

    from visualize import OverlayRenderer

    renderer = OverlayRenderer()
    annotated = renderer.draw(frame, detections, recent_events, trails, cfg)
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from config import Event, EventSearchConfig
from detector import Detection

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLOR_LINE = (0, 255, 255)
COLOR_ZONE = (200, 180, 0)
COLOR_ZONE_FILL = (120, 100, 0)
COLOR_BOX = (255, 180, 0)
COLOR_TRAIL = (180, 180, 180)
COLOR_TEXT_BG = (40, 40, 40)
COLOR_WHITE = (255, 255, 255)
COLOR_EVENT = (0, 200, 0)
COLOR_ALERT = (0, 80, 255)


class OverlayRenderer:
    """Compose analytics overlays onto a BGR frame."""

    def __init__(self, alpha: float = 0.25) -> None:
        self.alpha = alpha

    def draw(
        self,
        frame: np.ndarray,
        detections: Sequence[Detection],
        recent_events: Sequence[Event],
        trails: dict[int, list[tuple[int, int]]],
        cfg: EventSearchConfig,
    ) -> np.ndarray:
        canvas = frame.copy()
        self._draw_zones(canvas, cfg)
        self._draw_lines(canvas, cfg)
        self._draw_trails(canvas, trails)
        self._draw_detections(canvas, detections)
        self._draw_event_log(canvas, recent_events)
        return canvas

    def _draw_zones(self, canvas: np.ndarray, cfg: EventSearchConfig) -> None:
        overlay = canvas.copy()
        for z in cfg.zones:
            pts = np.array(z.polygon, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], COLOR_ZONE_FILL)
            cv2.polylines(canvas, [pts], True, COLOR_ZONE, 2)
            cx = int(np.mean([p[0] for p in z.polygon]))
            cy = int(np.mean([p[1] for p in z.polygon])) - 10
            cv2.putText(canvas, z.name, (cx - 40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 2)
        cv2.addWeighted(overlay, self.alpha, canvas, 1 - self.alpha, 0, canvas)

    def _draw_lines(self, canvas: np.ndarray, cfg: EventSearchConfig) -> None:
        for ln in cfg.lines:
            cv2.line(canvas, ln.pt1, ln.pt2, COLOR_LINE, 2)
            mid_x = (ln.pt1[0] + ln.pt2[0]) // 2
            mid_y = (ln.pt1[1] + ln.pt2[1]) // 2 - 10
            cv2.putText(canvas, ln.name, (mid_x - 40, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_LINE, 1)

    def _draw_trails(self, canvas: np.ndarray,
                     trails: dict[int, list[tuple[int, int]]]) -> None:
        for trail in trails.values():
            if len(trail) < 2:
                continue
            for i in range(1, len(trail)):
                cv2.line(canvas, trail[i - 1], trail[i], COLOR_TRAIL, 1)

    def _draw_detections(self, canvas: np.ndarray,
                         detections: Sequence[Detection]) -> None:
        for det in detections:
            x1, y1, x2, y2 = det.box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_BOX, 2)
            label = f"{det.class_name}"
            if det.track_id is not None:
                label = f"ID:{det.track_id} {label}"
            label += f" {det.confidence:.2f}"
            self._put_label(canvas, label, (x1, y1 - 5))

    def _draw_event_log(self, canvas: np.ndarray,
                        events: Sequence[Event]) -> None:
        """Draw recent events as text lines at top-right."""
        if not events:
            return
        x_start = canvas.shape[1] - 350
        y = 25
        for evt in events[-8:]:
            text = f"{evt.event_type} T{evt.track_id} {evt.zone_or_line}"
            if evt.dwell_seconds:
                text += f" {evt.dwell_seconds:.1f}s"
            cv2.rectangle(canvas, (x_start - 5, y - 15),
                          (canvas.shape[1] - 5, y + 5), COLOR_TEXT_BG, -1)
            cv2.putText(canvas, text, (x_start, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_EVENT, 1)
            y += 22

    @staticmethod
    def _put_label(canvas: np.ndarray, text: str, org: tuple[int, int]) -> None:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(canvas, (org[0], org[1] - th - 4),
                      (org[0] + tw + 4, org[1] + 2), COLOR_TEXT_BG, -1)
        cv2.putText(canvas, text, (org[0] + 2, org[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)
