"""
Modern OMR Evaluator — CVProject wrapper v2
=============================================
Wraps existing OMR evaluation logic in the unified CVProject framework.

Upgrade note: Add per-bubble confidence (fill_ratio threshold, e.g. >0.5
for filled) and a score summary overlay to make grading actionable.
Consider perspective-warp alignment before bubble detection.

Original: OMRevaluator.py (threshold + contour-based MCQ grading)
Modern:   Same core logic, unified interface

Usage:
    python -m core.runner --import-all omr_evaluator_v2 --source omr_sheet.jpg
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


@register("omr_evaluator_v2")
class OMREvaluatorV2(CVProject):
    display_name = "OMR Evaluator (v2)"
    category = "opencv_utility"

    def load(self):
        pass

    def predict(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        doc_contour = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                doc_contour = approx
                break
        # Count filled bubbles via thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        bubble_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bubbles = []
        for c in bubble_contours:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h) if h > 0 else 0
            if 0.7 <= ar <= 1.3 and 15 <= w <= 50:
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)
                total = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
                bubbles.append({"bbox": (x, y, w, h), "fill_ratio": total / (w * h)})
        return {"doc_contour": doc_contour, "bubbles": bubbles}

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        annotated = frame.copy()
        if output["doc_contour"] is not None:
            cv2.drawContours(annotated, [output["doc_contour"]], -1, (0, 255, 0), 3)
        filled = [b for b in output["bubbles"] if b["fill_ratio"] > 0.5]
        for b in filled:
            x, y, w, h = b["bbox"]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(annotated, f"Filled bubbles: {len(filled)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return annotated
