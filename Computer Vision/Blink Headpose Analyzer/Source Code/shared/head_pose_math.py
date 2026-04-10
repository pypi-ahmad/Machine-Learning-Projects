"""Head pose estimation math — pure, reusable functions.

Provides ``solvePnP``-based head pose estimation and rotation
matrix ↔ Euler angle conversion.  No MediaPipe dependency;
works with any set of 2D–3D point correspondences.

Usage::

    from shared.head_pose_math import solve_head_pose, rotation_to_euler

    yaw, pitch, roll = solve_head_pose(image_points_6x2, frame_w, frame_h)
"""

from __future__ import annotations

import cv2
import numpy as np

from shared.landmarks import MODEL_POINTS_3D


def solve_head_pose(
    image_points: np.ndarray,
    frame_w: int,
    frame_h: int,
    *,
    model_points: np.ndarray | None = None,
    camera_matrix: np.ndarray | None = None,
) -> tuple[float, float, float] | None:
    """Estimate yaw / pitch / roll from 2D ↔ 3D correspondences.

    Parameters
    ----------
    image_points : ndarray, shape (N, 2)
        2D pixel locations of the face landmarks (at least 6).
    frame_w, frame_h : int
        Frame dimensions (used to build a default camera matrix).
    model_points : ndarray, optional
        3D reference model.  Defaults to the 6-point canonical
        face model in :pydata:`shared.landmarks.MODEL_POINTS_3D`.
    camera_matrix : ndarray, optional
        3×3 intrinsic matrix.  Defaults to pinhole with
        ``focal_length = frame_w``.

    Returns
    -------
    tuple[float, float, float] or None
        ``(yaw, pitch, roll)`` in degrees, or *None* on failure.
    """
    if model_points is None:
        model_points = MODEL_POINTS_3D

    image_points = np.asarray(image_points, dtype=np.float64)
    model_points = np.asarray(model_points, dtype=np.float64)

    if camera_matrix is None:
        focal = float(frame_w)
        cx, cy = frame_w / 2.0, frame_h / 2.0
        camera_matrix = np.array([
            [focal, 0, cx],
            [0, focal, cy],
            [0, 0, 1],
        ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    return rotation_to_euler(rmat)


def rotation_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """Convert a 3×3 rotation matrix to ``(yaw, pitch, roll)`` degrees.

    Uses the ZYX (Tait-Bryan) convention.

    Parameters
    ----------
    R : ndarray, shape (3, 3)
        Rotation matrix.

    Returns
    -------
    tuple[float, float, float]
        ``(yaw, pitch, roll)`` in degrees.
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        yaw = np.degrees(np.arctan2(-R[2, 0], sy))
        roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        yaw = np.degrees(np.arctan2(-R[2, 0], sy))
        roll = 0.0

    return float(yaw), float(pitch), float(roll)
