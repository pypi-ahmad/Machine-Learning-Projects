"""Shared reusable face-landmark utilities.

This package provides pure-function utilities for:

- **EAR** (Eye Aspect Ratio) computation
- **Head pose** estimation (solvePnP + Euler conversion)
- **Landmark index constants** for MediaPipe face landmarks

Other face-analysis projects can import these directly::

    from shared.ear import compute_ear_from_points, compute_ear_bilateral
    from shared.head_pose_math import solve_head_pose, rotation_to_euler
    from shared.landmarks import LEFT_EYE, RIGHT_EYE, MODEL_POINTS_3D
"""

from shared.ear import compute_ear_bilateral, compute_ear_from_points
from shared.head_pose_math import rotation_to_euler, solve_head_pose
from shared.landmarks import (
    CHIN,
    LEFT_EYE,
    LEFT_EYE_CORNER,
    LEFT_MOUTH_CORNER,
    MODEL_POINTS_3D,
    NOSE_TIP,
    POSE_LANDMARKS,
    RIGHT_EYE,
    RIGHT_EYE_CORNER,
    RIGHT_MOUTH_CORNER,
    extract_eye_points,
    extract_pose_points,
    pixel_coords,
)

__all__ = [
    "CHIN",
    "LEFT_EYE",
    "LEFT_EYE_CORNER",
    "LEFT_MOUTH_CORNER",
    "MODEL_POINTS_3D",
    "NOSE_TIP",
    "POSE_LANDMARKS",
    "RIGHT_EYE",
    "RIGHT_EYE_CORNER",
    "RIGHT_MOUTH_CORNER",
    "compute_ear_bilateral",
    "compute_ear_from_points",
    "extract_eye_points",
    "extract_pose_points",
    "pixel_coords",
    "rotation_to_euler",
    "solve_head_pose",
]
