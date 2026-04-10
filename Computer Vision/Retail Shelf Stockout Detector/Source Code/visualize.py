"""On-frame visualization for inference and monitoring.

Renders detection boxes, zone polygons with fill, per-zone counts,
low-stock warnings, and an info dashboard onto frames.

Usage::

    from visualize import OverlayRenderer

    renderer = OverlayRenderer(zone_counter)
    annotated = renderer.draw(frame, frame_result)
"""

from __future__ import annotations

import cv2
import numpy as np

from config import ZoneConfig
from zones import FrameResult, ZoneCounter


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
_GREEN = (0, 200, 0)
_RED = (0, 0, 255)
_YELLOW = (0, 220, 255)
_WHITE = (255, 255, 255)
_DARK = (30, 30, 30)


class OverlayRenderer:
    """Draws detection + zone overlays on frames.

    Parameters
    ----------
    counter : ZoneCounter
        Provides zone polygon definitions for rendering.
    box_thickness : int
        Bounding-box line width.
    font_scale : float
        Global font scale multiplier.
    zone_alpha : float
        Transparency for zone polygon fills (0–1).
    """

    def __init__(
        self,
        counter: ZoneCounter,
        *,
        box_thickness: int = 2,
        font_scale: float = 0.55,
        zone_alpha: float = 0.20,
    ) -> None:
        self._counter = counter
        self._box_thick = box_thickness
        self._font_scale = font_scale
        self._zone_alpha = zone_alpha
        self._font = cv2.FONT_HERSHEY_SIMPLEX

    def draw(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        """Render all overlays onto *frame* (in-place) and return it."""
        vis = frame.copy()
        self._draw_zones(vis, result)
        self._draw_detections(vis, result)
        self._draw_alerts(vis, result)
        self._draw_dashboard(vis, result)
        return vis

    # ── private ────────────────────────────────────────────

    def _draw_zones(self, vis: np.ndarray, result: FrameResult) -> None:
        """Draw zone polygons with semi-transparent fill."""
        overlay = vis.copy()
        for zs in result.zone_statuses:
            zone_cfg = self._zone_by_name(zs.name)
            if zone_cfg is None:
                continue
            pts = np.array(zone_cfg.polygon, dtype=np.int32)
            color = _RED if zs.is_low_stock else _GREEN

            # Fill
            cv2.fillPoly(overlay, [pts], color)

            # Border
            cv2.polylines(vis, [pts], True, color, self._box_thick)

            # Label
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1])) - 10
            label = f"{zs.name}: {zs.count}/{zs.threshold}"
            self._put_label(vis, label, (cx - 40, cy), color)

        # Blend fill
        cv2.addWeighted(overlay, self._zone_alpha, vis, 1 - self._zone_alpha, 0, vis)

    def _draw_detections(self, vis: np.ndarray, result: FrameResult) -> None:
        """Draw bounding boxes with class labels."""
        for det in result.detections:
            x1, y1, x2, y2 = det.box
            color = _GREEN
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, self._box_thick)
            label = f"{det.class_name} {det.confidence:.2f}"
            self._put_label(vis, label, (x1, y1 - 6), color, scale=0.4)

    def _draw_alerts(self, vis: np.ndarray, result: FrameResult) -> None:
        """Draw alert banners in the top-left corner."""
        for i, alert in enumerate(result.alerts):
            y = 28 + i * 28
            cv2.putText(vis, alert, (10, y), self._font,
                        self._font_scale + 0.1, _RED, 2, cv2.LINE_AA)

    def _draw_dashboard(self, vis: np.ndarray, result: FrameResult) -> None:
        """Draw a small translucent info panel in the bottom-left."""
        h, w = vis.shape[:2]
        panel_h = 30 + len(result.zone_statuses) * 22
        panel_w = 260
        y0 = h - panel_h - 10
        x0 = 10

        # Semi-transparent background
        overlay = vis.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, h - 10), _DARK, -1)
        cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)

        # Title
        cv2.putText(vis, f"Total: {result.total_count} objects",
                    (x0 + 8, y0 + 20), self._font, 0.45, _WHITE, 1, cv2.LINE_AA)

        # Per-zone
        for i, zs in enumerate(result.zone_statuses):
            c = _RED if zs.is_low_stock else _GREEN
            line = f"  {zs.name}: {zs.count}/{zs.threshold}"
            cv2.putText(vis, line, (x0 + 8, y0 + 42 + i * 22),
                        self._font, 0.40, c, 1, cv2.LINE_AA)

    def _put_label(
        self, vis: np.ndarray, text: str,
        org: tuple[int, int], color: tuple[int, int, int],
        scale: float | None = None,
    ) -> None:
        """Draw text with a dark background for legibility."""
        sc = scale or self._font_scale
        (tw, th), _ = cv2.getTextSize(text, self._font, sc, 1)
        x, y = org
        cv2.rectangle(vis, (x, y - th - 4), (x + tw + 4, y + 4), _DARK, -1)
        cv2.putText(vis, text, (x + 2, y), self._font, sc, color, 1, cv2.LINE_AA)

    def _zone_by_name(self, name: str) -> ZoneConfig | None:
        for z in self._counter.zone_configs:
            if z.name == name:
                return z
        return None
