"""Traffic Violation Analyzer — overlay renderer.

Draws virtual lines, zone polygons, vehicle bounding boxes with track
IDs, movement trails, violation alerts, and a mini dashboard.

Usage::

    from visualize import OverlayRenderer

    renderer = OverlayRenderer()
    annotated = renderer.draw(frame, detections, frame_events, trails, cfg)
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from config import TrafficConfig
from detector import Detection
from rules import FrameEvents, ViolationEvent


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLOR_LINE       = (0, 255, 255)     # yellow
COLOR_LINE_WARN  = (0, 0, 255)       # red (wrong-way line)
COLOR_ZONE       = (200, 180, 0)     # teal
COLOR_BOX        = (255, 180, 0)     # cyan-ish
COLOR_TRAIL      = (180, 180, 180)   # grey
COLOR_WRONG_WAY  = (0, 0, 255)       # red
COLOR_TEXT_BG    = (40, 40, 40)
COLOR_WHITE      = (255, 255, 255)
COLOR_ALERT      = (0, 80, 255)      # orange-red


class OverlayRenderer:
    """Compose traffic-analytics overlays onto a BGR frame."""

    def __init__(self, alpha: float = 0.25) -> None:
        self.alpha = alpha

    # ---- public API --------------------------------------------------------

    def draw(
        self,
        frame: np.ndarray,
        detections: Sequence[Detection],
        frame_events: FrameEvents,
        trails: dict[int, list[tuple[int, int]]],
        cfg: TrafficConfig,
    ) -> np.ndarray:
        """Return a copy of *frame* with all overlays composited."""
        canvas = frame.copy()
        self._draw_zones(canvas, cfg)
        self._draw_lines(canvas, cfg, frame_events)
        self._draw_trails(canvas, trails)
        self._draw_detections(canvas, detections)
        self._draw_alerts(canvas, frame_events)
        self._draw_dashboard(canvas, frame_events, len(detections))
        return canvas

    # ---- virtual lines -----------------------------------------------------

    def _draw_lines(self, canvas: np.ndarray, cfg: TrafficConfig,
                    fe: FrameEvents) -> None:
        wrong_lines = {e.line_or_zone for e in fe.events if e.event_type == "wrong_way"}
        for ln in cfg.lines:
            colour = COLOR_LINE_WARN if ln.name in wrong_lines else COLOR_LINE
            cv2.line(canvas, ln.pt1, ln.pt2, colour, 2)
            mid_x = (ln.pt1[0] + ln.pt2[0]) // 2
            mid_y = (ln.pt1[1] + ln.pt2[1]) // 2 - 10
            count = fe.line_counts.get(ln.name, 0)
            label = f"{ln.name}: {count}"
            self._put_label(canvas, label, (mid_x - 40, mid_y), colour)

    # ---- zone polygons -----------------------------------------------------

    def _draw_zones(self, canvas: np.ndarray, cfg: TrafficConfig) -> None:
        overlay = canvas.copy()
        for z in cfg.zones:
            pts = np.array(z.polygon, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], COLOR_ZONE)
            cv2.polylines(canvas, [pts], True, COLOR_ZONE, 2)
            cx = int(np.mean([p[0] for p in z.polygon]))
            cy = int(np.mean([p[1] for p in z.polygon])) - 10
            cv2.putText(canvas, z.name, (cx - 40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 2)
        cv2.addWeighted(overlay, self.alpha, canvas, 1 - self.alpha, 0, canvas)

    # ---- trails ------------------------------------------------------------

    def _draw_trails(self, canvas: np.ndarray,
                     trails: dict[int, list[tuple[int, int]]]) -> None:
        for trail in trails.values():
            if len(trail) < 2:
                continue
            for i in range(1, len(trail)):
                cv2.line(canvas, trail[i - 1], trail[i], COLOR_TRAIL, 1)

    # ---- vehicle boxes + track IDs ----------------------------------------

    def _draw_detections(self, canvas: np.ndarray,
                         detections: Sequence[Detection]) -> None:
        for det in detections:
            x1, y1, x2, y2 = det.box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_BOX, 2)
            tid = f"#{det.track_id}" if det.track_id is not None else ""
            label = f"{det.class_name} {tid} {det.confidence:.0%}"
            self._put_label(canvas, label, (x1, y1 - 6), COLOR_BOX)

    # ---- alert banners -----------------------------------------------------

    def _draw_alerts(self, canvas: np.ndarray, fe: FrameEvents) -> None:
        alerts = [e for e in fe.events if e.event_type == "wrong_way"]
        if not alerts:
            return
        h, w = canvas.shape[:2]
        y_offset = 30
        for a in alerts[:5]:
            text = f"WRONG WAY [{a.line_or_zone}] #{a.track_id} ({a.direction})"
            cv2.rectangle(canvas, (10, y_offset - 18), (w - 10, y_offset + 6),
                          COLOR_ALERT, -1)
            cv2.putText(canvas, text, (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 2)
            y_offset += 30

    # ---- dashboard (bottom-right summary) ----------------------------------

    def _draw_dashboard(self, canvas: np.ndarray, fe: FrameEvents,
                        n_detections: int) -> None:
        h, w = canvas.shape[:2]
        lines_text = [f"Vehicles: {n_detections}"]
        for name, count in fe.line_counts.items():
            lines_text.append(f"{name}: {count}")
        if fe.wrong_way_count:
            lines_text.append(f"Wrong-way: {fe.wrong_way_count}")

        box_w = 200
        box_h = 20 + 22 * len(lines_text)
        x0 = w - box_w - 10
        y0 = h - box_h - 10
        cv2.rectangle(canvas, (x0, y0), (x0 + box_w, y0 + box_h), COLOR_TEXT_BG, -1)
        for i, line in enumerate(lines_text):
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
