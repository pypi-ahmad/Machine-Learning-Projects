"""Yoga Pose Correction Coach — pure joint-angle computation.

All functions take plain coordinates and return plain numbers.
No MediaPipe dependency — fully testable with synthetic points.
"""

from __future__ import annotations

import math


def angle_3pt(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> float:
    """Compute the interior angle at vertex *b* (degrees, 0–180)."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)
    if mag_ba < 1e-9 or mag_bc < 1e-9:
        return 0.0
    cos_a = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_a))


def vertical_angle(
    top: tuple[float, float],
    bottom: tuple[float, float],
) -> float:
    """Angle (degrees) of the top→bottom vector from vertical.

    0° = perfectly vertical, 90° = horizontal.
    """
    dx = bottom[0] - top[0]
    dy = bottom[1] - top[1]
    if abs(dy) < 1e-9 and abs(dx) < 1e-9:
        return 0.0
    return abs(math.degrees(math.atan2(dx, dy)))


def horizontal_deviation(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    """How far a left–right pair deviates from horizontal (degrees).

    0° = perfectly horizontal.
    """
    dy = right[1] - left[1]
    dx = right[0] - left[0]
    if abs(dx) < 1e-9:
        return 90.0
    return abs(math.degrees(math.atan2(dy, dx)))
