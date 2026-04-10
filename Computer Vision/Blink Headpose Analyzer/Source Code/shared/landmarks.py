"""MediaPipe Face Mesh landmark index constants and helpers.

Centralises the landmark indices and 3D model points used across
face-analysis projects.  Import from here instead of hard-coding::

    from shared.landmarks import LEFT_EYE, RIGHT_EYE, MODEL_POINTS_3D
"""

from __future__ import annotations

import numpy as np

# ── 6-point eye contour (for EAR) ────────────────────────
LEFT_EYE: list[int] = [362, 385, 387, 263, 373, 380]
RIGHT_EYE: list[int] = [33, 160, 158, 133, 153, 144]

# ── Mouth contour (for MAR, optional) ────────────────────
UPPER_LIP: list[int] = [13]
LOWER_LIP: list[int] = [14]
LEFT_MOUTH: list[int] = [78]
RIGHT_MOUTH: list[int] = [308]

# ── Head-pose landmark indices ────────────────────────────
NOSE_TIP: int = 1
CHIN: int = 152
LEFT_EYE_CORNER: int = 263
RIGHT_EYE_CORNER: int = 33
LEFT_MOUTH_CORNER: int = 61
RIGHT_MOUTH_CORNER: int = 291

POSE_LANDMARKS: list[int] = [
    NOSE_TIP,
    CHIN,
    LEFT_EYE_CORNER,
    RIGHT_EYE_CORNER,
    LEFT_MOUTH_CORNER,
    RIGHT_MOUTH_CORNER,
]

# ── 3D model points for solvePnP (generic face, mm) ──────
MODEL_POINTS_3D: np.ndarray = np.array([
    (0.0, 0.0, 0.0),           # Nose tip
    (0.0, -330.0, -65.0),      # Chin
    (-225.0, 170.0, -135.0),   # Left eye corner
    (225.0, 170.0, -135.0),    # Right eye corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0, -150.0, -125.0),   # Right mouth corner
], dtype=np.float64)


def pixel_coords(
    landmarks,
    index: int,
    frame_w: int,
    frame_h: int,
) -> tuple[float, float]:
    """Extract (x, y) pixel coordinates from a MediaPipe landmark list.

    Parameters
    ----------
    landmarks
        ``face_landmarks.landmark`` list from MediaPipe.
    index : int
        Landmark index (0–467).
    frame_w, frame_h : int
        Frame dimensions.

    Returns
    -------
    tuple[float, float]
    """
    lm = landmarks[index]
    return lm.x * frame_w, lm.y * frame_h


def extract_eye_points(
    landmarks,
    eye_indices: list[int],
    frame_w: int,
    frame_h: int,
) -> list[tuple[float, float]]:
    """Extract eye contour points as pixel coordinates.

    Parameters
    ----------
    landmarks
        MediaPipe landmark list.
    eye_indices : list[int]
        6 landmark indices for one eye.
    frame_w, frame_h : int
        Frame dimensions.

    Returns
    -------
    list[tuple[float, float]]
    """
    return [pixel_coords(landmarks, i, frame_w, frame_h) for i in eye_indices]


def extract_pose_points(
    landmarks,
    frame_w: int,
    frame_h: int,
    *,
    indices: list[int] | None = None,
) -> np.ndarray:
    """Extract head-pose landmark points as pixel coordinates.

    Parameters
    ----------
    landmarks
        MediaPipe landmark list.
    frame_w, frame_h : int
        Frame dimensions.
    indices : list[int], optional
        Landmark indices.  Defaults to :pydata:`POSE_LANDMARKS`.

    Returns
    -------
    np.ndarray, shape (N, 2)
    """
    if indices is None:
        indices = POSE_LANDMARKS
    return np.array(
        [pixel_coords(landmarks, i, frame_w, frame_h) for i in indices],
        dtype=np.float64,
    )
