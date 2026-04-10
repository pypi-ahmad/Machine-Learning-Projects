"""Overlay renderer for Sports Ball Possession Tracker.

Draws:
- Player bounding boxes with track IDs
- Ball detection with highlight
- Possession indicator line (ball → holder)
- Player trails
- Possession timeline bar
"""

from __future__ import annotations

import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PossessionConfig
from possession import PossessionState
from tracker import Detection, FrameDetections

FONT = cv2.FONT_HERSHEY_SIMPLEX
PLAYER_COLOUR = (0, 220, 0)       # green
BALL_COLOUR = (0, 200, 255)       # yellow
HOLDER_COLOUR = (0, 255, 255)     # cyan
POSSESSION_LINE = (0, 140, 255)   # orange
TRAIL_ALPHA = 0.5


class Visualizer:
    """Stateful renderer that maintains trail history."""

    def __init__(self, cfg: PossessionConfig) -> None:
        self.cfg = cfg
        self._trails: defaultdict[int, deque[tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=cfg.trail_length)
        )

    def draw(self, frame: np.ndarray, dets: FrameDetections,
             state: PossessionState) -> np.ndarray:
        """Render all overlays on *frame* (copy returned)."""
        vis = frame.copy()

        # Update trails
        for p in dets.players:
            if p.track_id >= 0:
                self._trails[p.track_id].append(p.centre)

        # 1. Trails
        if self.cfg.show_trails:
            self._draw_trails(vis)

        # 2. Player boxes
        for p in dets.players:
            is_holder = p.track_id == state.current_holder_id
            self._draw_player(vis, p, is_holder)

        # 3. Ball
        for b in dets.balls:
            self._draw_ball(vis, b)

        # 4. Possession line
        if state.current_holder_id is not None and state.ball_centre:
            self._draw_possession_link(vis, dets, state)

        # 5. Possession bar
        if self.cfg.show_possession_bar:
            self._draw_possession_bar(vis, state)

        # 6. Info panel
        self._draw_info(vis, dets, state)

        return vis

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_player(self, vis: np.ndarray, det: Detection, is_holder: bool) -> None:
        x1, y1, x2, y2 = det.bbox
        colour = HOLDER_COLOUR if is_holder else PLAYER_COLOUR
        thickness = self.cfg.line_width + 1 if is_holder else self.cfg.line_width
        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, thickness)

        label = f"#{det.track_id}" if det.track_id >= 0 else det.class_name
        if is_holder:
            label += " [POSS]"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.45, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4), FONT, 0.45, (0, 0, 0), 1)

    def _draw_ball(self, vis: np.ndarray, det: Detection) -> None:
        cx, cy = det.centre
        radius = max(8, (det.bbox[2] - det.bbox[0]) // 2)
        cv2.circle(vis, (cx, cy), radius + 4, BALL_COLOUR, 2)
        cv2.circle(vis, (cx, cy), 3, BALL_COLOUR, -1)
        label = f"ball {det.confidence:.0%}"
        cv2.putText(vis, label, (cx + radius + 4, cy + 4), FONT, 0.4, BALL_COLOUR, 1)

    def _draw_possession_link(self, vis: np.ndarray, dets: FrameDetections,
                               state: PossessionState) -> None:
        holder = None
        for p in dets.players:
            if p.track_id == state.current_holder_id:
                holder = p
                break
        if holder is None or state.ball_centre is None:
            return
        cv2.line(vis, state.ball_centre, holder.centre, POSSESSION_LINE, 2, cv2.LINE_AA)

    def _draw_trails(self, vis: np.ndarray) -> None:
        overlay = vis.copy()
        for tid, trail in self._trails.items():
            pts = list(trail)
            if len(pts) < 2:
                continue
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                colour = (
                    int(PLAYER_COLOUR[0] * alpha),
                    int(PLAYER_COLOUR[1] * alpha),
                    int(PLAYER_COLOUR[2] * alpha),
                )
                cv2.line(overlay, pts[i - 1], pts[i], colour, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, TRAIL_ALPHA, vis, 1 - TRAIL_ALPHA, 0, vis)

    def _draw_possession_bar(self, vis: np.ndarray, state: PossessionState) -> None:
        """Horizontal bar at the bottom showing cumulative possession split."""
        h, w = vis.shape[:2]
        bar_h = 24
        y0 = h - bar_h - 10

        total = sum(state.cumulative_frames.values()) or 1
        x_cursor = 10
        bar_w = w - 20

        overlay = vis.copy()
        cv2.rectangle(overlay, (10, y0), (10 + bar_w, y0 + bar_h), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

        colours = _generate_colours(len(state.cumulative_frames))
        for i, (pid, frames) in enumerate(
            sorted(state.cumulative_frames.items(), key=lambda x: -x[1])
        ):
            seg_w = max(1, int(frames / total * bar_w))
            colour = colours[i % len(colours)]
            cv2.rectangle(vis, (x_cursor, y0), (x_cursor + seg_w, y0 + bar_h), colour, -1)
            if seg_w > 30:
                pct = frames / total * 100
                cv2.putText(vis, f"#{pid} {pct:.0f}%", (x_cursor + 3, y0 + 17),
                            FONT, 0.35, (255, 255, 255), 1)
            x_cursor += seg_w

    def _draw_info(self, vis: np.ndarray, dets: FrameDetections,
                    state: PossessionState) -> None:
        """Top-left info panel."""
        lines = [
            f"Players: {len(dets.players)}  Ball: {'YES' if state.ball_detected else 'NO'}",
            f"Holder: {state.current_holder_name or 'None'}",
        ]
        if state.distance_to_holder is not None:
            lines.append(f"Dist: {state.distance_to_holder:.0f}px")

        y = 25
        for line in lines:
            (tw, th), _ = cv2.getTextSize(line, FONT, 0.5, 1)
            cv2.rectangle(vis, (8, y - th - 4), (12 + tw + 4, y + 4), (0, 0, 0), -1)
            cv2.putText(vis, line, (12, y), FONT, 0.5, (255, 255, 255), 1)
            y += th + 14


def _generate_colours(n: int) -> list[tuple[int, int, int]]:
    """Generate *n* distinct BGR colours via HSV spacing."""
    colours = []
    for i in range(max(n, 1)):
        hue = int(i * 180 / max(n, 1)) % 180
        hsv = np.uint8([[[hue, 200, 220]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colours.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colours
