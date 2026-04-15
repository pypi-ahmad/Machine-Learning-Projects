"""Iris position locator for Gaze Direction Estimator.

Extracts iris center positions relative to the eye bounding box
to produce horizontal and vertical gaze ratios.

MediaPipe iris indices:
    - Subject-left iris:  468 (center), 469–472 (contour)
    - Subject-right iris: 473 (center), 474–477 (contour)

Eye contour indices used for bounding box:
    - Left eye:  [362, 385, 387, 263, 373, 380]
    - Right eye: [33, 160, 158, 133, 153, 144]

Gaze ratios:
    - ``h_ratio``: 0.0 = looking left, 0.5 = center, 1.0 = right
    - ``v_ratio``: 0.0 = looking up,   0.5 = center, 1.0 = down
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from landmark_engine import LandmarkResult

log = logging.getLogger("gaze.iris_locator")

# Iris center landmarks paired to image-space eye contours.
# MediaPipe's subject-centric iris ordering is the reverse of the
# viewer-space eye contour arrays used below.
LEFT_IRIS_CENTER = 473
RIGHT_IRIS_CENTER = 468

# Eye contour landmarks for bounding-box estimation
LEFT_EYE_CONTOUR = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_CONTOUR = [33, 160, 158, 133, 153, 144]


@dataclass
class IrisPosition:
    """Iris position data for one frame."""

    detected: bool = False

    # Per-eye iris center (pixel coords)
    left_iris_x: float = 0.0
    left_iris_y: float = 0.0
    right_iris_x: float = 0.0
    right_iris_y: float = 0.0

    # Gaze ratios (0–1)
    left_h_ratio: float = 0.5
    left_v_ratio: float = 0.5
    right_h_ratio: float = 0.5
    right_v_ratio: float = 0.5

    # Averaged ratios
    h_ratio: float = 0.5
    v_ratio: float = 0.5


def locate_iris(lm: LandmarkResult) -> IrisPosition:
    """Compute iris gaze ratios from landmarks.

    Parameters
    ----------
    lm : LandmarkResult
        Detected face landmarks (must include iris refinement).

    Returns
    -------
    IrisPosition
    """
    pos = IrisPosition()
    if not lm.detected:
        return pos

    try:
        # Left eye
        l_iris = np.array(lm.pixel_coords(LEFT_IRIS_CENTER))
        l_pts = np.array([lm.pixel_coords(i) for i in LEFT_EYE_CONTOUR])
        l_h, l_v = _compute_ratios(l_iris, l_pts)

        # Right eye
        r_iris = np.array(lm.pixel_coords(RIGHT_IRIS_CENTER))
        r_pts = np.array([lm.pixel_coords(i) for i in RIGHT_EYE_CONTOUR])
        r_h, r_v = _compute_ratios(r_iris, r_pts)

        pos.detected = True
        pos.left_iris_x, pos.left_iris_y = float(l_iris[0]), float(l_iris[1])
        pos.right_iris_x, pos.right_iris_y = float(r_iris[0]), float(r_iris[1])
        pos.left_h_ratio = l_h
        pos.left_v_ratio = l_v
        pos.right_h_ratio = r_h
        pos.right_v_ratio = r_v
        pos.h_ratio = (l_h + r_h) / 2.0
        pos.v_ratio = (l_v + r_v) / 2.0
    except (IndexError, ValueError) as exc:
        log.debug("Iris location failed: %s", exc)

    return pos


def _compute_ratios(
    iris_center: np.ndarray,
    eye_contour: np.ndarray,
) -> tuple[float, float]:
    """Compute horizontal and vertical iris position ratios.

    The iris center is measured relative to the eye contour
    bounding box.

    Returns
    -------
    tuple[float, float]
        ``(h_ratio, v_ratio)`` each in [0, 1].
    """
    x_min, y_min = eye_contour.min(axis=0)
    x_max, y_max = eye_contour.max(axis=0)

    w = x_max - x_min
    h = y_max - y_min

    if w < 1e-4 or h < 1e-4:
        return 0.5, 0.5

    h_ratio = float(np.clip((iris_center[0] - x_min) / w, 0.0, 1.0))
    v_ratio = float(np.clip((iris_center[1] - y_min) / h, 0.0, 1.0))

    return h_ratio, v_ratio
