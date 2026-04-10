"""Overlay renderer for Crowd Zone Counter.

Draws:
- Zone polygons (transparent fill + border)
- Person bounding boxes
- Per-zone count labels
- Overcrowding alert banners
- Summary dashboard
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CrowdConfig
from detector import FrameDetections
from zone_counter import FrameResult, ZoneState

FONT = cv2.FONT_HERSHEY_SIMPLEX
PERSON_COLOUR = (0, 200, 0)      # green
ALERT_COLOUR = (0, 0, 255)       # red
TEXT_BG = (0, 0, 0)


def draw_overlay(frame: np.ndarray, dets: FrameDetections,
                 result: FrameResult, cfg: CrowdConfig) -> np.ndarray:
    """Render all visual elements on *frame* (copy returned)."""
    vis = frame.copy()

    # 1. Zone polygons
    if cfg.zones:
        _draw_zones(vis, result, cfg)

    # 2. Person boxes
    for p in dets.persons:
        _draw_person(vis, p, cfg)

    # 3. Zone count labels
    if cfg.show_counts:
        _draw_zone_labels(vis, result, cfg)

    # 4. Alerts
    if cfg.show_alerts and result.alerts:
        _draw_alerts(vis, result)

    # 5. Dashboard
    _draw_dashboard(vis, result)

    return vis


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _zone_colour(idx: int, zone: ZoneState, cfg: CrowdConfig) -> tuple[int, int, int]:
    """Get BGR colour for a zone, auto-generating if not specified."""
    zone_cfg = cfg.zones[idx] if idx < len(cfg.zones) else None
    if zone_cfg and zone_cfg.colour:
        return tuple(zone_cfg.colour)  # type: ignore[return-value]
    # Auto-generate via HSV
    hue = int(idx * 60) % 180
    hsv = np.uint8([[[hue, 180, 200]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _draw_zones(vis: np.ndarray, result: FrameResult, cfg: CrowdConfig) -> None:
    overlay = vis.copy()
    for i, zs in enumerate(result.zone_states):
        colour = ALERT_COLOUR if zs.overcrowded else _zone_colour(i, zs, cfg)
        pts = np.array(cfg.zones[i].polygon, dtype=np.int32)

        if cfg.show_zone_fill:
            cv2.fillPoly(overlay, [pts], colour)

        border_thick = cfg.line_width + 1 if zs.overcrowded else cfg.line_width
        cv2.polylines(vis, [pts], isClosed=True, color=colour, thickness=border_thick)

    cv2.addWeighted(overlay, cfg.zone_alpha, vis, 1 - cfg.zone_alpha, 0, vis)


def _draw_person(vis: np.ndarray, p, cfg: CrowdConfig) -> None:
    x1, y1, x2, y2 = p.bbox
    cv2.rectangle(vis, (x1, y1), (x2, y2), PERSON_COLOUR, 1)
    cv2.circle(vis, p.foot_point, 3, PERSON_COLOUR, -1)


def _draw_zone_labels(vis: np.ndarray, result: FrameResult, cfg: CrowdConfig) -> None:
    for i, zs in enumerate(result.zone_states):
        pts = cfg.zones[i].polygon
        cx = int(np.mean([p[0] for p in pts]))
        cy = int(np.mean([p[1] for p in pts]))

        cap_text = f"/{zs.max_capacity}" if zs.max_capacity > 0 else ""
        label = f"{zs.name}: {zs.count}{cap_text}"

        colour = ALERT_COLOUR if zs.overcrowded else _zone_colour(i, zs, cfg)
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 2)
        cv2.rectangle(vis, (cx - tw // 2 - 4, cy - th - 6),
                       (cx + tw // 2 + 4, cy + 4), TEXT_BG, -1)
        cv2.putText(vis, label, (cx - tw // 2, cy), FONT, 0.6, colour, 2)


def _draw_alerts(vis: np.ndarray, result: FrameResult) -> None:
    h, w = vis.shape[:2]
    y = 60
    for alert in result.alerts:
        msg = f"OVERCROWDED: {alert.zone_name} ({alert.count}/{alert.max_capacity})"
        (tw, th), _ = cv2.getTextSize(msg, FONT, 0.7, 2)
        cx = w // 2
        cv2.rectangle(vis, (cx - tw // 2 - 8, y - th - 8),
                       (cx + tw // 2 + 8, y + 8), (0, 0, 180), -1)
        cv2.putText(vis, msg, (cx - tw // 2, y), FONT, 0.7, (255, 255, 255), 2)
        y += th + 24


def _draw_dashboard(vis: np.ndarray, result: FrameResult) -> None:
    h, w = vis.shape[:2]
    margin = 10
    line_h = 22
    num_lines = len(result.zone_states) + 3  # header + total + unzoned
    panel_h = line_h * num_lines + margin * 2
    panel_w = 220

    x0, y0 = w - panel_w - margin, margin
    overlay = vis.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

    ty = y0 + margin + line_h
    cv2.putText(vis, "Zone Counts", (x0 + 8, ty), FONT, 0.55, (255, 255, 255), 1)
    ty += line_h

    for zs in result.zone_states:
        colour = ALERT_COLOUR if zs.overcrowded else (200, 200, 200)
        cap = f"/{zs.max_capacity}" if zs.max_capacity > 0 else ""
        cv2.putText(vis, f"{zs.name}: {zs.count}{cap}", (x0 + 12, ty),
                    FONT, 0.45, colour, 1)
        ty += line_h

    cv2.putText(vis, f"Unzoned: {result.unzoned_count}", (x0 + 8, ty),
                FONT, 0.45, (160, 160, 160), 1)
    ty += line_h
    cv2.putText(vis, f"Total: {result.total_persons}", (x0 + 8, ty),
                FONT, 0.5, (0, 255, 255), 1)
