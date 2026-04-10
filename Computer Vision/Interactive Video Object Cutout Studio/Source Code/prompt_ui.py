"""Interactive Video Object Cutout Studio — interactive prompt collector.

Pure UI module: uses OpenCV highgui to let the user click foreground /
background points and drag bounding boxes on an image.  Knows nothing
about SAM 2 — just collects geometric prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class PromptSet:
    """Collection of user-supplied prompts for SAM 2."""

    points: list[tuple[int, int]] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)           # 1=fg, 0=bg
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.points and not self.boxes

    # ── numpy helpers for the engine ──────────────────────

    def points_array(self) -> np.ndarray | None:
        if not self.points:
            return None
        return np.array(self.points, dtype=np.float32)

    def labels_array(self) -> np.ndarray | None:
        if not self.labels:
            return None
        return np.array(self.labels, dtype=np.int32)

    def box_array(self) -> np.ndarray | None:
        """Return the last box as (4,) or None."""
        if not self.boxes:
            return None
        return np.array(self.boxes[-1], dtype=np.float32)


# ── instructions overlay ──────────────────────────────────

_INSTRUCTIONS = [
    "Left-click  : foreground point (green)",
    "Right-click : background point  (red)",
    "Shift+drag  : bounding box      (cyan)",
    "'u'         : undo last action",
    "'c'         : clear all prompts",
    "Enter       : confirm & segment",
    "Esc         : cancel",
]


class PromptCollector:
    """Interactive OpenCV window for collecting point / box prompts."""

    def __init__(
        self,
        window_name: str = "SAM 2 Prompt",
        show_instructions: bool = True,
    ) -> None:
        self._win = window_name
        self._show_instr = show_instructions
        self._prompts = PromptSet()
        self._actions: list[str] = []  # "point" | "box" for undo
        self._dragging = False
        self._drag_start: tuple[int, int] | None = None
        self._drag_end: tuple[int, int] | None = None
        self._base_image: np.ndarray | None = None

    # ── public API ────────────────────────────────────────

    def collect(self, image_bgr: np.ndarray) -> PromptSet | None:
        """Show *image_bgr* and let the user annotate prompts.

        Returns a :class:`PromptSet` on Enter, or ``None`` on Esc.
        """
        self._base_image = image_bgr.copy()
        self._prompts = PromptSet()
        self._actions.clear()
        self._dragging = False

        cv2.namedWindow(self._win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self._win, self._mouse_cb)
        self._redraw()

        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # Esc
                cv2.destroyWindow(self._win)
                return None
            if key in (13, 10):  # Enter
                cv2.destroyWindow(self._win)
                return self._prompts
            if key == ord("u"):
                self._undo()
                self._redraw()
            if key == ord("c"):
                self._prompts = PromptSet()
                self._actions.clear()
                self._redraw()

    # ── mouse callback ────────────────────────────────────

    def _mouse_cb(self, event: int, x: int, y: int, flags: int, _: object) -> None:
        shift = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)

        # ── box drag ──
        if shift and event == cv2.EVENT_LBUTTONDOWN:
            self._dragging = True
            self._drag_start = (x, y)
            self._drag_end = (x, y)
            return

        if self._dragging and event == cv2.EVENT_MOUSEMOVE:
            self._drag_end = (x, y)
            self._redraw()
            return

        if self._dragging and event == cv2.EVENT_LBUTTONUP:
            self._dragging = False
            if self._drag_start is not None:
                x1, y1 = self._drag_start
                x2, y2 = x, y
                self._prompts.boxes.append((
                    min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2),
                ))
                self._actions.append("box")
            self._drag_start = self._drag_end = None
            self._redraw()
            return

        # ── point clicks (no shift) ──
        if event == cv2.EVENT_LBUTTONDOWN and not shift:
            self._prompts.points.append((x, y))
            self._prompts.labels.append(1)
            self._actions.append("point")
            self._redraw()

        elif event == cv2.EVENT_RBUTTONDOWN:
            self._prompts.points.append((x, y))
            self._prompts.labels.append(0)
            self._actions.append("point")
            self._redraw()

    # ── drawing helpers ───────────────────────────────────

    def _redraw(self) -> None:
        vis = self._base_image.copy()

        # draw boxes
        for x1, y1, x2, y2 in self._prompts.boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 200, 0), 2)

        # draw in-progress drag
        if self._dragging and self._drag_start and self._drag_end:
            cv2.rectangle(vis, self._drag_start, self._drag_end, (255, 200, 0), 1)

        # draw points
        for (px, py), lbl in zip(self._prompts.points, self._prompts.labels):
            color = (0, 255, 0) if lbl == 1 else (0, 0, 255)
            cv2.circle(vis, (px, py), 6, color, -1)
            cv2.circle(vis, (px, py), 6, (255, 255, 255), 1)

        # instructions
        if self._show_instr:
            for i, line in enumerate(_INSTRUCTIONS):
                cv2.putText(
                    vis, line, (10, 22 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA,
                )

        cv2.imshow(self._win, vis)

    def _undo(self) -> None:
        if not self._actions:
            return
        last = self._actions.pop()
        if last == "point" and self._prompts.points:
            self._prompts.points.pop()
            self._prompts.labels.pop()
        elif last == "box" and self._prompts.boxes:
            self._prompts.boxes.pop()
