"""Card boundary detection and perspective rectification.

Finds the ID card quadrilateral in an image via contour detection
and applies a four-point perspective transform to produce a
front-facing, axis-aligned card crop.

Usage::

    from card_detector import CardDetector
    from config import IDCardConfig

    detector = CardDetector(IDCardConfig())
    rectified, corners = detector.detect_and_rectify(image)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger("id_card_kyc.card_detector")


@dataclass
class DetectionResult:
    """Result of card boundary detection."""

    found: bool
    corners: np.ndarray | None = None   # (4, 2) ordered quad or None
    rectified: np.ndarray | None = None  # perspective-corrected crop


class CardDetector:
    """Detect and rectify an ID card in an image."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def detect_and_rectify(self, image: np.ndarray) -> DetectionResult:
        """Find the card and return a rectified crop.

        Falls back to the original image if no card boundary is found.
        """
        if not self.cfg.detect_card:
            return DetectionResult(found=False, rectified=image)

        corners = self._find_card_quad(image)

        if corners is None:
            log.info("No card boundary detected — using full image")
            return DetectionResult(found=False, rectified=image)

        if self.cfg.rectify:
            rectified = self._four_point_transform(image, corners)
        else:
            rectified = image

        return DetectionResult(found=True, corners=corners, rectified=rectified)

    # ------------------------------------------------------------------
    # Boundary detection
    # ------------------------------------------------------------------

    def _find_card_quad(self, image: np.ndarray) -> np.ndarray | None:
        """Find the largest quadrilateral contour in *image*.

        Returns ordered corners (4, 2) or None.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(
            blurred, self.cfg.canny_low, self.cfg.canny_high,
        )

        # Dilate to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return None

        img_area = image.shape[0] * image.shape[1]
        min_area = img_area * self.cfg.min_card_area_ratio

        # Sort by area descending
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours[:5]:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, self.cfg.approx_eps * peri, True)

            if len(approx) == 4:
                corners = approx.reshape(4, 2).astype(np.float32)
                return self._order_corners(corners)

        return None

    # ------------------------------------------------------------------
    # Perspective transform
    # ------------------------------------------------------------------

    def _four_point_transform(
        self,
        image: np.ndarray,
        corners: np.ndarray,
    ) -> np.ndarray:
        """Apply perspective warp to produce a front-facing card."""
        dst = np.array([
            [0, 0],
            [self.cfg.target_width - 1, 0],
            [self.cfg.target_width - 1, self.cfg.target_height - 1],
            [0, self.cfg.target_height - 1],
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(corners, dst)
        warped = cv2.warpPerspective(
            image, M, (self.cfg.target_width, self.cfg.target_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return warped

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _order_corners(pts: np.ndarray) -> np.ndarray:
        """Order corners: top-left, top-right, bottom-right, bottom-left."""
        ordered = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)
        ordered[0] = pts[np.argmin(s)]   # top-left
        ordered[2] = pts[np.argmax(s)]   # bottom-right

        d = np.diff(pts, axis=1)
        ordered[1] = pts[np.argmin(d)]   # top-right
        ordered[3] = pts[np.argmax(d)]   # bottom-left

        return ordered
