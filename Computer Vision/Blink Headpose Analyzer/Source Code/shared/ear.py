"""Eye Aspect Ratio (EAR) computation — pure, reusable functions.

EAR measures eye openness from 6 contour landmarks::

    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 · ||p1 - p4||)

Typical values:
    - 0.25–0.35  eyes wide open
    - 0.15–0.21  eyes closing / drowsy
    - < 0.15     eyes closed

Usage::

    from shared.ear import compute_ear_from_points

    pts = [(x1,y1), (x2,y2), ..., (x6,y6)]
    ear = compute_ear_from_points(pts)
"""

from __future__ import annotations

import numpy as np


def compute_ear_from_points(
    pts: list[tuple[float, float]] | np.ndarray,
) -> float:
    """Compute EAR from 6 (x, y) eye-contour points.

    Parameters
    ----------
    pts : list or ndarray
        Exactly 6 points in MediaPipe order:
        ``[p1_lateral, p2_upper_a, p3_upper_b, p4_medial,
          p5_lower_b, p6_lower_a]``

    Returns
    -------
    float
        Eye Aspect Ratio (≥ 0).
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (6, 2):
        raise ValueError(f"Expected 6 points, got shape {pts.shape}")

    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h = np.linalg.norm(pts[0] - pts[3])

    if h < 1e-6:
        return 0.3  # fallback: assume open
    return float((v1 + v2) / (2.0 * h))


def compute_ear_bilateral(
    left_pts: list[tuple[float, float]] | np.ndarray,
    right_pts: list[tuple[float, float]] | np.ndarray,
) -> tuple[float, float, float]:
    """Compute EAR for both eyes and return the average.

    Returns
    -------
    tuple[float, float, float]
        ``(left_ear, right_ear, avg_ear)``
    """
    left = compute_ear_from_points(left_pts)
    right = compute_ear_from_points(right_pts)
    return left, right, (left + right) / 2.0
