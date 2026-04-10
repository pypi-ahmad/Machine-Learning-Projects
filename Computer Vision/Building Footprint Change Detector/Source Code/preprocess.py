"""Building Footprint Change Detector — image pair preprocessing.

Responsibilities:
  * Load before / after images from disk.
  * Resize both to a common resolution so masks are pixel-aligned.
  * Optional histogram-matching to reduce illumination differences.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class ImagePair:
    """A preprocessed before/after image pair."""

    before: np.ndarray  # BGR, uint8
    after: np.ndarray   # BGR, uint8
    original_before: np.ndarray  # untouched load
    original_after: np.ndarray
    target_size: tuple[int, int]  # (width, height) after resize


def load_image(path: str | Path) -> np.ndarray:
    """Read an image from *path* and return as BGR numpy array."""
    p = str(Path(path).resolve())
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {p}")
    return img


def resize_to_common(
    before: np.ndarray,
    after: np.ndarray,
    target_size: int | tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Resize *before* and *after* to identical dimensions.

    Parameters
    ----------
    target_size
        ``int`` → square, ``(w, h)`` → explicit, or ``None`` → use the
        larger of the two input sizes.

    Returns
    -------
    tuple
        ``(resized_before, resized_after, (width, height))``
    """
    if target_size is None:
        h = max(before.shape[0], after.shape[0])
        w = max(before.shape[1], after.shape[1])
    elif isinstance(target_size, int):
        w = h = target_size
    else:
        w, h = target_size

    bfr = cv2.resize(before, (w, h), interpolation=cv2.INTER_LINEAR)
    afr = cv2.resize(after, (w, h), interpolation=cv2.INTER_LINEAR)
    return bfr, afr, (w, h)


def histogram_match(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match the histogram of *source* to *reference* (per channel).

    Reduces false positives caused by illumination / seasonal differences
    between the two capture dates.
    """
    matched = np.empty_like(source)
    for ch in range(3):
        s = source[:, :, ch].ravel()
        r = reference[:, :, ch].ravel()

        s_counts, _ = np.histogram(s, bins=256, range=(0, 256))
        r_counts, _ = np.histogram(r, bins=256, range=(0, 256))

        s_cdf = np.cumsum(s_counts).astype(np.float64)
        s_cdf /= s_cdf[-1]
        r_cdf = np.cumsum(r_counts).astype(np.float64)
        r_cdf /= r_cdf[-1]

        lookup = np.interp(s_cdf, r_cdf, np.arange(256)).astype(np.uint8)
        matched[:, :, ch] = lookup[source[:, :, ch]]
    return matched


def prepare_pair(
    before_path: str | Path,
    after_path: str | Path,
    *,
    target_size: int | tuple[int, int] | None = None,
    match_histograms: bool = False,
) -> ImagePair:
    """Full preprocessing pipeline for one image pair."""
    orig_b = load_image(before_path)
    orig_a = load_image(after_path)

    b, a, size = resize_to_common(orig_b, orig_a, target_size)

    if match_histograms:
        a = histogram_match(a, b)

    return ImagePair(
        before=b,
        after=a,
        original_before=orig_b,
        original_after=orig_a,
        target_size=size,
    )
