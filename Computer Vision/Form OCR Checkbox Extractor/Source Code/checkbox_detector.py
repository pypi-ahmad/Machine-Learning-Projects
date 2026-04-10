"""OpenCV-based checkbox and radio-button detector.

Uses adaptive thresholding + morphological operations + contour
analysis to locate small rectangular (checkbox) and circular
(radio) form elements, then classifies each as checked or
unchecked based on interior pixel density.

Usage::

    from checkbox_detector import CheckboxDetector
    from config import FormCheckboxConfig

    det = CheckboxDetector(FormCheckboxConfig())
    results = det.detect(image)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

log = logging.getLogger("form_checkbox.detector")


class ControlType(Enum):
    CHECKBOX = "checkbox"
    RADIO = "radio"


class ControlState(Enum):
    CHECKED = "checked"
    UNCHECKED = "unchecked"
    UNKNOWN = "unknown"


@dataclass
class FormControl:
    """A single detected checkbox or radio button."""

    control_type: ControlType
    state: ControlState
    bbox: tuple[int, int, int, int]   # (x, y, w, h)
    centre: tuple[int, int]
    fill_ratio: float
    contour: np.ndarray


class CheckboxDetector:
    """Detect checkboxes and radio buttons using OpenCV morphology."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def detect(self, image: np.ndarray) -> list[FormControl]:
        """Find all checkbox / radio controls in *image* (BGR)."""
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            grey, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.cfg.adaptive_block_size,
            self.cfg.adaptive_c,
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.cfg.morph_kernel_size, self.cfg.morph_kernel_size),
        )
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
        )

        controls: list[FormControl] = []
        for cnt in contours:
            ctrl = self._classify_contour(cnt, grey)
            if ctrl is not None:
                controls.append(ctrl)

        controls = self._deduplicate(controls)
        controls.sort(key=lambda c: (c.centre[1], c.centre[0]))
        log.info("Detected %d form controls", len(controls))
        return controls

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _classify_contour(
        self, cnt: np.ndarray, grey: np.ndarray,
    ) -> FormControl | None:
        x, y, w, h = cv2.boundingRect(cnt)

        # Size filter
        if w < self.cfg.checkbox_min_size or h < self.cfg.checkbox_min_size:
            return None
        if w > self.cfg.checkbox_max_size or h > self.cfg.checkbox_max_size:
            return None

        # Aspect ratio filter
        aspect = w / max(h, 1)
        if aspect < self.cfg.checkbox_aspect_lo or aspect > self.cfg.checkbox_aspect_hi:
            return None

        # Determine control type via circularity
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            return None
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity >= self.cfg.radio_circularity:
            ctype = ControlType.RADIO
        else:
            # Check squareness via contour approximation
            approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
            if len(approx) < 4:
                return None
            ctype = ControlType.CHECKBOX

        # Fill ratio → checked / unchecked
        roi = grey[y : y + h, x : x + w]
        if roi.size == 0:
            return None
        _, roi_bin = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        fill_ratio = float(np.count_nonzero(roi_bin)) / roi.size

        if fill_ratio >= self.cfg.fill_threshold:
            state = ControlState.CHECKED
        else:
            state = ControlState.UNCHECKED

        cx = x + w // 2
        cy = y + h // 2

        return FormControl(
            control_type=ctype,
            state=state,
            bbox=(x, y, w, h),
            centre=(cx, cy),
            fill_ratio=fill_ratio,
            contour=cnt,
        )

    def _deduplicate(self, controls: list[FormControl]) -> list[FormControl]:
        """Remove overlapping detections, keeping the one with larger area."""
        if not controls:
            return controls

        keep: list[FormControl] = []
        used = [False] * len(controls)

        for i, a in enumerate(controls):
            if used[i]:
                continue
            best = a
            for j in range(i + 1, len(controls)):
                if used[j]:
                    continue
                b = controls[j]
                if self._iou(a.bbox, b.bbox) > 0.3:
                    used[j] = True
                    if b.bbox[2] * b.bbox[3] > best.bbox[2] * best.bbox[3]:
                        best = b
            keep.append(best)

        return keep

    @staticmethod
    def _iou(
        a: tuple[int, int, int, int],
        b: tuple[int, int, int, int],
    ) -> float:
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        if ix1 >= ix2 or iy1 >= iy2:
            return 0.0

        inter = (ix2 - ix1) * (iy2 - iy1)
        union = aw * ah + bw * bh - inter
        return inter / max(union, 1)
