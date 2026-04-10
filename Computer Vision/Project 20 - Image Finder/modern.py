"""
Modern Image Finder — CVProject wrapper v2
============================================
Wraps template matching logic in the unified CVProject framework.

Upgrade note: For scale/rotation-invariant matching, replace
cv2.matchTemplate with ORB or SIFT feature matching
(cv2.BFMatcher / cv2.FlannBasedMatcher). For semantic similarity,
use CLIP embeddings.

Original: finder.py (cv2.matchTemplate)
Modern:   Same core logic, unified interface

Usage:
    python -m core.runner --import-all image_finder_v2 --source scene.jpg
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


@register("image_finder_v2")
class ImageFinderV2(CVProject):
    display_name = "Image Finder (v2)"
    category = "opencv_utility"

    THRESHOLD = 0.8

    def load(self):
        self.template = None  # Set via set_template() before use

    def set_template(self, template: np.ndarray):
        self.template = template

    def predict(self, frame: np.ndarray):
        if self.template is None:
            return {"matches": [], "msg": "No template set. Call set_template() first."}
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_tpl = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY) \
            if len(self.template.shape) == 3 else self.template
        res = cv2.matchTemplate(gray_frame, gray_tpl, cv2.TM_CCOEFF_NORMED)
        locs = np.where(res >= self.THRESHOLD)
        h, w = gray_tpl.shape[:2]
        matches = [(int(x), int(y), w, h) for (y, x) in zip(*locs)]
        return {"matches": matches}

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        annotated = frame.copy()
        for (x, y, w, h) in output.get("matches", []):
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        n = len(output.get("matches", []))
        msg = output.get("msg", f"Matches: {n}")
        cv2.putText(annotated, msg, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return annotated
