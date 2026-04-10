"""Exercise Rep Counter — pure joint-angle computation.

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
    """Compute the interior angle at point *b* (in degrees).

    Parameters
    ----------
    a, b, c : (x, y) tuples
        Three points defining the angle.  *b* is the vertex.

    Returns
    -------
    float
        Angle in degrees in the range [0, 180].
    """
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)
    if mag_ba < 1e-9 or mag_bc < 1e-9:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


def midpoint(
    a: tuple[float, float],
    b: tuple[float, float],
) -> tuple[float, float]:
    """Return the midpoint between *a* and *b*."""
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
