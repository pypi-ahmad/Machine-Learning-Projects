"""
Modern Warp / Click-Detect — CVProject wrapper v2
===================================================
Auto-detect largest quadrilateral + perspective warp, with interactive
click-point annotation mode.

Merged: Absorbs "Project 45 - Detecting Clicks on Images" (interactive
click annotation). Both features are now available:
  - Auto-warp: detects quadrilaterals and applies perspective transform
  - Click annotation: interactive point marking via add_click()

Original: Warp.py / Detecting_Clicks_On_Images.py
Modern:   Unified perspective tools, CVProject interface

Usage:
    python -m core.runner --import-all warp_perspective_v2 --source image.jpg
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


def _order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


@register("warp_perspective_v2")
@register("click_detection_v2")
class WarpPerspectiveV2(CVProject):
    display_name = "Warp / Click Detect (v2)"
    category = "opencv_utility"

    def load(self):
        self.click_points = []  # Interactive click annotation

    # -- Interactive click API (from P45) --
    def add_click(self, x, y):
        """Register a mouse-click coordinate for annotation overlay."""
        self.click_points.append((x, y))

    def predict(self, frame: np.ndarray):
        # Auto-detect quadrilateral
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        quad = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                quad = approx
                break
        warped = None
        if quad is not None:
            pts = _order_points(quad.reshape(4, 2).astype("float32"))
            (tl, tr, br, bl) = pts
            maxW = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
            maxH = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
            dst = np.array([[0, 0], [maxW - 1, 0],
                            [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(pts, dst)
            warped = cv2.warpPerspective(frame, M, (maxW, maxH))
        return {"quad": quad, "warped": warped, "clicks": list(self.click_points)}

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        annotated = frame.copy()
        # Warp overlay
        if output["quad"] is not None:
            cv2.drawContours(annotated, [output["quad"]], -1, (0, 255, 0), 3)
            cv2.putText(annotated, "Quadrilateral detected - warping", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(annotated, "Looking for quadrilateral...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        # Click annotation overlay
        for i, (x, y) in enumerate(output.get("clicks", [])):
            cv2.circle(annotated, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(annotated, f"P{i + 1}({x},{y})", (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        n_clicks = len(output.get("clicks", []))
        if n_clicks:
            cv2.putText(annotated, f"Clicks: {n_clicks}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        return annotated
