"""
Modern Coin Line Drawing — CVProject wrapper v2
=================================================
Wraps coin detection + vertical line drawing in the unified CVProject framework.

Original: Drawing vertical lines on coin.ipynb (HoughCircles + lines)
Modern:   Same core logic, unified interface

Usage:
    python -m core.runner --import-all coin_lines_v2 --source coins.jpg
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


@register("coin_lines_v2")
class CoinLinesV2(CVProject):
    display_name = "Coin Lines (v2)"
    category = "opencv_utility"

    def load(self):
        pass

    def predict(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
            param1=100, param2=40, minRadius=15, maxRadius=100
        )
        return {"circles": circles}

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        annotated = frame.copy()
        h = annotated.shape[0]
        circles = output["circles"]
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for (x, y, r) in circles[0, :]:
                cv2.circle(annotated, (x, y), r, (0, 255, 0), 2)
                cv2.circle(annotated, (x, y), 2, (0, 0, 255), 3)
                cv2.line(annotated, (x, 0), (x, h), (255, 0, 0), 1)
            cv2.putText(annotated, f"Coins: {len(circles[0])}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(annotated, "No coins detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        return annotated
