"""Image preprocessing for Receipt Digitizer.

Cleans noisy receipt images before OCR: denoising, deskewing,
sharpening, binarisation, and resizing.

Usage::

    from preprocess import preprocess_receipt
    from config import ReceiptConfig

    clean = preprocess_receipt(image, ReceiptConfig())
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger("receipt_digitizer.preprocess")


def preprocess_receipt(image: np.ndarray, cfg) -> np.ndarray:
    """Apply configured preprocessing steps to *image* (BGR).

    Returns a cleaned BGR image suitable for OCR.
    """
    out = image.copy()

    if cfg.resize_max > 0:
        out = _resize(out, cfg.resize_max)

    if cfg.deskew:
        out = _deskew(out)

    if cfg.denoise:
        out = _denoise(out)

    if cfg.sharpen:
        out = _sharpen(out)

    if cfg.binarize:
        out = _binarize(out)

    return out


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _resize(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image
    scale = max_side / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _denoise(image: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(image, None, 8, 8, 7, 21)


def _sharpen(image: np.ndarray) -> np.ndarray:
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)


def _binarize(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 8,
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def _deskew(image: np.ndarray) -> np.ndarray:
    """Correct small rotational skew using Hough line detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 200, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100,
        minLineLength=gray.shape[1] // 4, maxLineGap=10,
    )
    if lines is None:
        return image

    angles: list[float] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2 - x1, y2 - y1
        if abs(dx) < 1:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        if abs(angle) < 15:
            angles.append(angle)

    if not angles:
        return image

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.3:
        return image

    h, w = image.shape[:2]
    centre = (w // 2, h // 2)
    mat = cv2.getRotationMatrix2D(centre, median_angle, 1.0)
    rotated = cv2.warpAffine(
        image, mat, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    log.debug("Deskewed by %.2f°", median_angle)
    return rotated
