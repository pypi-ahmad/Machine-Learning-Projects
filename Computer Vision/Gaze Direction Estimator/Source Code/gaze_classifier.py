"""Coarse gaze direction classifier.

Maps horizontal / vertical iris ratios to one of five gaze
directions: LEFT, RIGHT, UP, DOWN, CENTER.

This is a **heuristic approximation** — not a precise gaze-point
estimator.  Accuracy depends on camera angle, lighting, head pose,
and individual eye geometry.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import GazeConfig
from iris_locator import IrisPosition

# Gaze direction labels
LEFT = "LEFT"
RIGHT = "RIGHT"
UP = "UP"
DOWN = "DOWN"
CENTER = "CENTER"

ALL_DIRECTIONS = [LEFT, RIGHT, UP, DOWN, CENTER]


@dataclass
class GazeResult:
    """Classified gaze direction for one frame."""

    direction: str = CENTER
    h_ratio: float = 0.5
    v_ratio: float = 0.5
    confidence: str = "low"     # "low" | "medium" | "high"


def classify_gaze(
    iris: IrisPosition,
    cfg: GazeConfig,
    *,
    h_offset: float = 0.0,
    v_offset: float = 0.0,
) -> GazeResult:
    """Classify gaze direction from iris position ratios.

    Parameters
    ----------
    iris : IrisPosition
        Iris position with horizontal/vertical ratios.
    cfg : GazeConfig
        Threshold configuration.
    h_offset, v_offset : float
        Calibration offsets applied to ratios.

    Returns
    -------
    GazeResult
    """
    result = GazeResult(h_ratio=iris.h_ratio, v_ratio=iris.v_ratio)

    if not iris.detected:
        return result

    h = iris.h_ratio - h_offset
    v = iris.v_ratio - v_offset

    result.h_ratio = h
    result.v_ratio = v

    # Horizontal takes priority when both are off-center,
    # since horizontal gaze shifts are more reliable with
    # this iris-ratio method.
    if h < cfg.horiz_left_threshold:
        result.direction = LEFT
    elif h > cfg.horiz_right_threshold:
        result.direction = RIGHT
    elif v < cfg.vert_up_threshold:
        result.direction = UP
    elif v > cfg.vert_down_threshold:
        result.direction = DOWN
    else:
        result.direction = CENTER

    # Simple confidence: how far from center
    h_dist = abs(h - 0.5)
    v_dist = abs(v - 0.5)
    max_dist = max(h_dist, v_dist)

    if result.direction == CENTER:
        result.confidence = "high" if max_dist < 0.08 else "medium"
    else:
        if max_dist > 0.20:
            result.confidence = "high"
        elif max_dist > 0.12:
            result.confidence = "medium"
        else:
            result.confidence = "low"

    return result
