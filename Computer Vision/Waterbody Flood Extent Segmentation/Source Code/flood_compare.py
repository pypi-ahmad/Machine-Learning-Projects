"""Waterbody & Flood Extent Segmentation — before/after flood comparison.

Compares water masks from two time-points to classify every pixel as:
  * **flooded_new** — water in *after* but not *before*
  * **receded**     — water in *before* but not *after*
  * **permanent**   — water in both
  * **dry**         — no water in either

Applies morphological cleanup and filters out tiny change regions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class FloodRegion:
    """One connected component of flood change."""

    label: str                       # "flooded_new" | "receded"
    area_px: int
    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    centroid: tuple[int, int]


@dataclass
class ComparisonResult:
    """Full before→after flood comparison output."""

    flooded_new_mask: np.ndarray     # uint8 (H, W) — 255 = new flooding
    receded_mask: np.ndarray         # uint8 (H, W) — 255 = receded
    permanent_mask: np.ndarray       # uint8 (H, W) — 255 = stable water
    change_map: np.ndarray           # uint8 (H, W) — 0=dry, 1=new, 2=receded, 3=permanent
    regions: list[FloodRegion] = field(default_factory=list)


def compare_flood_extent(
    before_mask: np.ndarray,
    after_mask: np.ndarray,
    *,
    morph_kernel_size: int = 5,
    min_change_area: int = 200,
) -> ComparisonResult:
    """Pixel-level before→after flood comparison.

    Parameters
    ----------
    before_mask, after_mask
        Binary uint8 masks (H, W) — 255 = water, 0 = dry.
    morph_kernel_size
        Kernel for morphological open/close.
    min_change_area
        Discard blobs smaller than this (pixels).
    """
    b = (before_mask > 127).astype(np.uint8)
    a = (after_mask > 127).astype(np.uint8)

    raw_new = a & (~b & 1)
    raw_receded = b & (~a & 1)
    permanent = b & a

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size),
    )

    new_clean = _morph_clean(raw_new, kernel, min_change_area)
    receded_clean = _morph_clean(raw_receded, kernel, min_change_area)

    change_map = np.zeros_like(b, dtype=np.uint8)
    change_map[new_clean > 0] = 1
    change_map[receded_clean > 0] = 2
    change_map[permanent > 0] = 3

    regions = _extract_regions(new_clean, "flooded_new", min_change_area)
    regions += _extract_regions(receded_clean, "receded", min_change_area)

    return ComparisonResult(
        flooded_new_mask=new_clean * 255,
        receded_mask=receded_clean * 255,
        permanent_mask=permanent * 255,
        change_map=change_map,
        regions=regions,
    )


def _morph_clean(binary: np.ndarray, kernel: np.ndarray, min_area: int) -> np.ndarray:
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            cleaned[labels == i] = 0
    return cleaned


def _extract_regions(binary: np.ndarray, label: str, min_area: int) -> list[FloodRegion]:
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    regions: list[FloodRegion] = []
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        regions.append(FloodRegion(label=label, area_px=area, bbox=(x, y, w, h), centroid=(cx, cy)))
    return regions
