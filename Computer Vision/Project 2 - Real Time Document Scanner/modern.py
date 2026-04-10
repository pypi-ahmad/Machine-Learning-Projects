"""
Modern Document Scanner — CVProject wrapper v2
================================================
Wraps existing OpenCV document scanning logic in the unified CVProject framework.
No DL replacement needed — contour-based perspective warp.

Upgrade note: After perspective warp, pipe the warped crop into PaddleOCR or
Tesseract for full scan-to-text. See Project 13 (Sudoku) for an OCR pipeline
example.

Original: docScanner.py (contour detection + perspective transform)
Modern:   Same core logic, unified interface

Usage:
    python -m core.runner --import-all doc_scanner_v2 --source 0
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


def _four_point_transform(image, pts):
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0], [maxW - 1, 0],
        [maxW - 1, maxH - 1], [0, maxH - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH))


@register("doc_scanner_v2")
class DocScannerV2(CVProject):
    display_name = "Document Scanner (v2)"
    category = "opencv_utility"

    def load(self):
        pass  # No model to load

    def predict(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        doc_contour = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                doc_contour = approx
                break
        return {"contour": doc_contour, "edged": edged}

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        annotated = frame.copy()
        doc_contour = output.get("contour")
        if doc_contour is not None:
            cv2.drawContours(annotated, [doc_contour], -1, (0, 255, 0), 3)
            pts = doc_contour.reshape(4, 2).astype("float32")
            warped = _four_point_transform(frame, pts)
            h, w = annotated.shape[:2]
            wh, ww = warped.shape[:2]
            scale = min(h // 3 / max(wh, 1), w // 3 / max(ww, 1))
            if scale > 0:
                thumb = cv2.resize(warped, (int(ww * scale), int(wh * scale)))
                th, tw = thumb.shape[:2]
                annotated[10:10 + th, w - tw - 10:w - 10] = thumb
            cv2.putText(annotated, "Document detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(annotated, "Scanning for document...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        return annotated
