"""Line segmentation for handwritten page images.

Splits a full-page handwritten image into individual text-line
crops using horizontal projection profiles.

Usage::

    from line_segmenter import LineSegmenter
    from config import NoteConfig

    seg = LineSegmenter(NoteConfig())
    lines = seg.segment(image)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger("handwritten_note.line_segmenter")


@dataclass
class LineRegion:
    """A single segmented text line."""

    y_start: int
    y_end: int
    x_offset: int               # left-most ink pixel (for indent detection)
    height: int
    crop: np.ndarray             # BGR line crop


class LineSegmenter:
    """Segment a page image into horizontal text lines."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def segment(self, image: np.ndarray) -> list[LineRegion]:
        """Split *image* (BGR) into text-line crops.

        If segmentation is disabled or the image is small enough to be
        a single line, returns the full image as one region.
        """
        if not self.cfg.enable_segmentation:
            return [self._full_image_region(image)]

        h, w = image.shape[:2]
        if h < self.cfg.min_line_height * 3:
            return [self._full_image_region(image)]

        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(
            grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        rows = self._find_line_rows(binary, w)
        rows = self._merge_close_rows(rows)

        if not rows:
            return [self._full_image_region(image)]

        regions: list[LineRegion] = []
        for y0, y1 in rows:
            # Add padding
            py0 = max(0, y0 - self.cfg.padding)
            py1 = min(h, y1 + self.cfg.padding)
            crop = image[py0:py1, :]

            # Detect x-offset (left-most ink) for indent detection
            line_bin = binary[y0:y1, :]
            col_sum = np.sum(line_bin, axis=0)
            ink_cols = np.where(col_sum > 0)[0]
            x_offset = int(ink_cols[0]) if len(ink_cols) > 0 else 0

            regions.append(LineRegion(
                y_start=y0,
                y_end=y1,
                x_offset=x_offset,
                height=y1 - y0,
                crop=crop,
            ))

        log.info("Segmented %d text lines", len(regions))
        return regions

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _find_line_rows(
        self, binary: np.ndarray, width: int,
    ) -> list[tuple[int, int]]:
        """Use horizontal projection profile to find text rows."""
        projection = np.sum(binary, axis=1) / 255.0
        threshold = width * self.cfg.projection_threshold

        rows: list[tuple[int, int]] = []
        in_line = False
        y_start = 0

        for y, val in enumerate(projection):
            if val >= threshold and not in_line:
                in_line = True
                y_start = y
            elif val < threshold and in_line:
                in_line = False
                if y - y_start >= self.cfg.min_line_height:
                    rows.append((y_start, y))

        # Handle line extending to bottom
        if in_line and len(binary) - y_start >= self.cfg.min_line_height:
            rows.append((y_start, len(binary)))

        return rows

    def _merge_close_rows(
        self, rows: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Merge rows that are closer than *merge_gap*."""
        if len(rows) <= 1:
            return rows

        merged: list[tuple[int, int]] = [rows[0]]
        for y0, y1 in rows[1:]:
            prev_y0, prev_y1 = merged[-1]
            if y0 - prev_y1 <= self.cfg.merge_gap:
                merged[-1] = (prev_y0, y1)
            else:
                merged.append((y0, y1))

        return merged

    @staticmethod
    def _full_image_region(image: np.ndarray) -> LineRegion:
        h, w = image.shape[:2]
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(
            grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        col_sum = np.sum(binary, axis=0)
        ink_cols = np.where(col_sum > 0)[0]
        x_offset = int(ink_cols[0]) if len(ink_cols) > 0 else 0
        return LineRegion(
            y_start=0, y_end=h, x_offset=x_offset, height=h, crop=image,
        )
