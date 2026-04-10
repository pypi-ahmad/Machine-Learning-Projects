"""Exercise Rep Counter — exercise definitions and stage detection.

Each exercise is defined by:
- Which landmarks form the measured joint angle
- Threshold angles for stage transitions ("up" / "down")
- A human-readable name

The rule logic is intentionally separated from the rep-counting
state machine so exercises can be added or modified independently.
"""

from __future__ import annotations

import dataclasses

from angle_calculator import angle_3pt
from pose_detector import LM, PoseResult


@dataclasses.dataclass
class ExerciseAnalysis:
    """Result of analysing one frame for a specific exercise."""

    exercise: str
    angle: float          # measured joint angle (degrees)
    stage: str            # "up", "down", or "unknown"
    landmarks_used: tuple[int, int, int]  # (a, b, c) landmark indices
    side: str             # "left" or "right"


# ---------------------------------------------------------------------------
# Exercise definitions
# ---------------------------------------------------------------------------

def _analyse_angle_exercise(
    pose: PoseResult,
    name: str,
    lm_a: int, lm_b: int, lm_c: int,
    down_threshold: float,
    up_threshold: float,
    side: str,
    *,
    invert: bool = False,
) -> ExerciseAnalysis:
    """Generic angle-based exercise analysis.

    *invert* flips the stage logic (used for bicep curls where
    a small angle means "up" rather than "down").
    """
    a = (pose.norm(lm_a)[0], pose.norm(lm_a)[1])
    b = (pose.norm(lm_b)[0], pose.norm(lm_b)[1])
    c = (pose.norm(lm_c)[0], pose.norm(lm_c)[1])
    angle = angle_3pt(a, b, c)

    if invert:
        if angle <= up_threshold:
            stage = "up"
        elif angle >= down_threshold:
            stage = "down"
        else:
            stage = "unknown"
    else:
        if angle <= down_threshold:
            stage = "down"
        elif angle >= up_threshold:
            stage = "up"
        else:
            stage = "unknown"

    return ExerciseAnalysis(
        exercise=name,
        angle=angle,
        stage=stage,
        landmarks_used=(lm_a, lm_b, lm_c),
        side=side,
    )


def analyse_squat(
    pose: PoseResult,
    down_angle: float = 90.0,
    up_angle: float = 160.0,
    side: str = "left",
) -> ExerciseAnalysis:
    """Squat: measure hip-knee-ankle angle."""
    if side == "right":
        return _analyse_angle_exercise(
            pose, "squat",
            LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE,
            down_angle, up_angle, "right",
        )
    return _analyse_angle_exercise(
        pose, "squat",
        LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE,
        down_angle, up_angle, "left",
    )


def analyse_pushup(
    pose: PoseResult,
    down_angle: float = 90.0,
    up_angle: float = 160.0,
    side: str = "left",
) -> ExerciseAnalysis:
    """Push-up: measure shoulder-elbow-wrist angle."""
    if side == "right":
        return _analyse_angle_exercise(
            pose, "pushup",
            LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST,
            down_angle, up_angle, "right",
        )
    return _analyse_angle_exercise(
        pose, "pushup",
        LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST,
        down_angle, up_angle, "left",
    )


def analyse_bicep_curl(
    pose: PoseResult,
    down_angle: float = 160.0,
    up_angle: float = 40.0,
    side: str = "left",
) -> ExerciseAnalysis:
    """Bicep curl: measure shoulder-elbow-wrist angle (inverted)."""
    if side == "right":
        return _analyse_angle_exercise(
            pose, "bicep_curl",
            LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST,
            down_angle, up_angle, "right",
            invert=True,
        )
    return _analyse_angle_exercise(
        pose, "bicep_curl",
        LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST,
        down_angle, up_angle, "left",
        invert=True,
    )


# Registry mapping exercise name → analysis function
EXERCISE_REGISTRY: dict[str, type] = {
    "squat": analyse_squat,
    "pushup": analyse_pushup,
    "bicep_curl": analyse_bicep_curl,
}


def get_exercise_analyser(exercise: str):
    """Return the analysis function for *exercise*."""
    fn = EXERCISE_REGISTRY.get(exercise)
    if fn is None:
        raise ValueError(
            f"Unknown exercise '{exercise}'. "
            f"Available: {', '.join(EXERCISE_REGISTRY)}"
        )
    return fn
