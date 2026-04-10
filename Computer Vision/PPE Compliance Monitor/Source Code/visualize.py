"""PPE Compliance Monitor — overlay renderer.

Draws zone polygons, person bounding boxes (coloured by compliance
status), PPE item boxes, alert banners, and a mini compliance dashboard
onto each video frame.

Usage::

    from visualize import OverlayRenderer

    renderer = OverlayRenderer(cfg)
    annotated = renderer.draw(frame, zone_statuses, alerts)
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from compliance import PersonCompliance
from zones import ZoneStatus, AlertEvent


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLOR_COMPLIANT = (0, 200, 0)       # green
COLOR_VIOLATION = (0, 0, 220)       # red
COLOR_PPE_BOX   = (255, 200, 0)     # cyan-ish
COLOR_ZONE_OK   = (0, 180, 0)       # green (zone fill)
COLOR_ZONE_WARN = (0, 0, 200)       # red (zone fill)
COLOR_TEXT_BG   = (40, 40, 40)
COLOR_WHITE     = (255, 255, 255)
COLOR_ALERT     = (0, 80, 255)      # orange-red


class OverlayRenderer:
    """Compose compliance overlays onto a BGR frame."""

    def __init__(self, alpha: float = 0.30) -> None:
        self.alpha = alpha

    # ---- public API --------------------------------------------------------

    def draw(
        self,
        frame: np.ndarray,
        zone_statuses: Sequence[ZoneStatus],
        alerts: Sequence[AlertEvent],
    ) -> np.ndarray:
        """Return a copy of *frame* with all overlays composited."""
        canvas = frame.copy()
        self._draw_zones(canvas, zone_statuses)
        for zs in zone_statuses:
            self._draw_persons(canvas, zs.persons)
        self._draw_alerts(canvas, alerts)
        self._draw_dashboard(canvas, zone_statuses)
        return canvas

    # ---- zone polygons -----------------------------------------------------

    def _draw_zones(self, canvas: np.ndarray, zone_statuses: Sequence[ZoneStatus]) -> None:
        overlay = canvas.copy()
        for zs in zone_statuses:
            if not zs.polygon:
                continue
            pts = np.array(zs.polygon, dtype=np.int32)
            colour = COLOR_ZONE_WARN if zs.violation_count > 0 else COLOR_ZONE_OK
            cv2.fillPoly(overlay, [pts], colour)
            cv2.polylines(canvas, [pts], True, colour, 2)
            # Zone label
            cx = int(np.mean([p[0] for p in zs.polygon]))
            cy = int(np.mean([p[1] for p in zs.polygon])) - 10
            cv2.putText(canvas, zs.name, (cx - 40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
        cv2.addWeighted(overlay, self.alpha, canvas, 1 - self.alpha, 0, canvas)

    # ---- person + PPE boxes ------------------------------------------------

    def _draw_persons(self, canvas: np.ndarray, persons: Sequence[PersonCompliance]) -> None:
        for pc in persons:
            colour = COLOR_COMPLIANT if pc.is_compliant else COLOR_VIOLATION
            x1, y1, x2, y2 = pc.person.box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)

            # Label: compliance status
            label = "OK" if pc.is_compliant else f"MISSING: {', '.join(pc.missing_items)}"
            self._put_label(canvas, label, (x1, y1 - 6), colour)

            # Draw associated PPE items
            for cls_name, det in pc.ppe_items.items():
                bx1, by1, bx2, by2 = det.box
                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), COLOR_PPE_BOX, 1)
                cv2.putText(canvas, cls_name, (bx1, by1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_PPE_BOX, 1)

    # ---- alert banners -----------------------------------------------------

    def _draw_alerts(self, canvas: np.ndarray, alerts: Sequence[AlertEvent]) -> None:
        if not alerts:
            return
        h, w = canvas.shape[:2]
        y_offset = 30
        for alert in alerts[:5]:  # cap visual alerts
            text = f"ALERT [{alert.zone_name}]: missing {', '.join(alert.missing_items)}"
            cv2.rectangle(canvas, (10, y_offset - 18), (w - 10, y_offset + 6), COLOR_ALERT, -1)
            cv2.putText(canvas, text, (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 2)
            y_offset += 30

    # ---- dashboard (bottom-right summary) ----------------------------------

    def _draw_dashboard(self, canvas: np.ndarray, zone_statuses: Sequence[ZoneStatus]) -> None:
        h, w = canvas.shape[:2]
        total = sum(zs.person_count for zs in zone_statuses)
        ok = sum(zs.compliant_count for zs in zone_statuses)
        viol = sum(zs.violation_count for zs in zone_statuses)

        lines = [
            f"Persons: {total}",
            f"Compliant: {ok}",
            f"Violations: {viol}",
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
