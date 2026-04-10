"""Building Footprint Change Detector — mask differencing engine.

Compares before/after binary building masks to classify every pixel as:
  * **new**         — building present in *after* but not *before*
  * **demolished**  — building present in *before* but not *after*
  * **unchanged**   — building in both (stable footprint)
  * **background**  — no building in either

Applies morphological cleanup and filters out tiny change regions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class ChangeRegion:
    """One connected component of change."""

    label: str                     # "new" | "demolished"
    area_px: int                   # pixel count
    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    centroid: tuple[int, int]      # (cx, cy)


@dataclass
class DiffResult:
    """Full diff output for a single image pair."""

    new_mask: np.ndarray           # uint8 (H, W) — 255 where new
    demolished_mask: np.ndarray    # uint8 (H, W) — 255 where demolished
    unchanged_mask: np.ndarray     # uint8 (H, W) — 255 where stable building
    change_map: np.ndarray         # uint8 (H, W) — labelled: 0=bg, 1=new, 2=demolished, 3=unchanged
    regions: list[ChangeRegion] = field(default_factory=list)


def compute_diff(
    before_mask: np.ndarray,
    after_mask: np.ndarray,
    *,
    morph_kernel_size: int = 5,
    min_change_area: int = 100,
) -> DiffResult:
    """Compute pixel-level building change between two binary masks.

    Parameters
    ----------
    before_mask, after_mask
        Binary uint8 masks (H, W) — 255 = building, 0 = background.
    morph_kernel_size
        Kernel size for morphological open/close cleanup.
    min_change_area
        Change blobs smaller than this (in pixels) are discarded.

    Returns
    -------
    DiffResult
    """
    b = (before_mask > 127).astype(np.uint8)
    a = (after_mask > 127).astype(np.uint8)

    raw_new = a & (~b & 1)           # 1 where after=1 and before=0
    raw_demo = b & (~a & 1)          # 1 where before=1 and after=0
    unchanged = b & a                # 1 where both=1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size),
    )

    new_clean = _morph_clean(raw_new, kernel, min_change_area)
    demo_clean = _morph_clean(raw_demo, kernel, min_change_area)

    # Build labelled change map
    change_map = np.zeros_like(b, dtype=np.uint8)
    change_map[new_clean > 0] = 1
    change_map[demo_clean > 0] = 2
    change_map[unchanged > 0] = 3

    # Extract connected-component regions
    regions = _extract_regions(new_clean, "new", min_change_area)
    regions += _extract_regions(demo_clean, "demolished", min_change_area)

    return DiffResult(
        new_mask=new_clean * 255,
        demolished_mask=demo_clean * 255,
        unchanged_mask=unchanged * 255,
        change_map=change_map,
        regions=regions,
    )


# ── helpers ────────────────────────────────────────────────


def _morph_clean(
    binary: np.ndarray,
    kernel: np.ndarray,
    min_area: int,
) -> np.ndarray:
    """Morphological open → close, then remove small blobs."""
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    # Remove blobs below min_area
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8,
    )
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            cleaned[labels == i] = 0
    return cleaned


def _extract_regions(
    binary: np.ndarray,
    label: str,
    min_area: int,
) -> list[ChangeRegion]:
    """Find connected components and return as ChangeRegion list."""
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8,
    )
    regions: list[ChangeRegion] = []
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        regions.append(ChangeRegion(
            label=label, area_px=area,
            bbox=(x, y, w, h), centroid=(cx, cy),
        ))
    return regions
