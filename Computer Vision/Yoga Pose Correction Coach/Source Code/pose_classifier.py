"""Yoga Pose Correction Coach — heuristic pose classifier.
"""Yoga Pose Correction Coach — heuristic pose classifier.

Classifies a detected body pose into one of the supported yoga poses
based on joint-angle signatures.  Each pose has a template of expected
angles; the classifier scores the current pose against all templates
and picks the best match.

This is a **heuristic** classifier — it uses hard-coded angle ranges
rather than a trained model.  Accuracy is limited to canonical poses
captured from roughly frontal or side views.
"""
"""

from __future__ import annotations

import dataclasses

from angle_calculator import angle_3pt, horizontal_deviation, vertical_angle
from pose_detector import LM, PoseResult


@dataclasses.dataclass
class ClassificationResult:
    """Pose classification output."""

    pose: str           # e.g. "warrior_ii" or "unknown"
    confidence: float   # 0.0–1.0
    scores: dict[str, float]  # per-pose match scores
    angles: dict[str, float]  # measured angles used for classification


# ---------------------------------------------------------------------------
# Angle extraction helpers
# ---------------------------------------------------------------------------

def _xy(pose: PoseResult, idx: int) -> tuple[float, float]:
    """Return (x, y) normalised coords for landmark *idx*."""
    return pose.norm(idx)[0], pose.norm(idx)[1]


def _measure_angles(pose: PoseResult) -> dict[str, float]:
    """Compute all angles used by the classifier."""
    return {
        # Left side
        "l_shoulder": angle_3pt(_xy(pose, LM.LEFT_HIP), _xy(pose, LM.LEFT_SHOULDER), _xy(pose, LM.LEFT_ELBOW)),
        "l_elbow": angle_3pt(_xy(pose, LM.LEFT_SHOULDER), _xy(pose, LM.LEFT_ELBOW), _xy(pose, LM.LEFT_WRIST)),
        "l_hip": angle_3pt(_xy(pose, LM.LEFT_SHOULDER), _xy(pose, LM.LEFT_HIP), _xy(pose, LM.LEFT_KNEE)),
        "l_knee": angle_3pt(_xy(pose, LM.LEFT_HIP), _xy(pose, LM.LEFT_KNEE), _xy(pose, LM.LEFT_ANKLE)),
        # Right side
        "r_shoulder": angle_3pt(_xy(pose, LM.RIGHT_HIP), _xy(pose, LM.RIGHT_SHOULDER), _xy(pose, LM.RIGHT_ELBOW)),
        "r_elbow": angle_3pt(_xy(pose, LM.RIGHT_SHOULDER), _xy(pose, LM.RIGHT_ELBOW), _xy(pose, LM.RIGHT_WRIST)),
        "r_hip": angle_3pt(_xy(pose, LM.RIGHT_SHOULDER), _xy(pose, LM.RIGHT_HIP), _xy(pose, LM.RIGHT_KNEE)),
        "r_knee": angle_3pt(_xy(pose, LM.RIGHT_HIP), _xy(pose, LM.RIGHT_KNEE), _xy(pose, LM.RIGHT_ANKLE)),
        # Torso
        "torso_vertical": vertical_angle(_xy(pose, LM.NOSE), _xy(pose, LM.LEFT_HIP)),
        "shoulder_level": horizontal_deviation(_xy(pose, LM.LEFT_SHOULDER), _xy(pose, LM.RIGHT_SHOULDER)),
        "hip_level": horizontal_deviation(_xy(pose, LM.LEFT_HIP), _xy(pose, LM.RIGHT_HIP)),
    }


# ---------------------------------------------------------------------------
# Pose templates: (angle_key, expected_value, weight)
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, list[tuple[str, float, float]]] = {
    "mountain": [
        # Standing upright, legs straight, arms at sides
        ("l_knee", 170, 1.0),
        ("r_knee", 170, 1.0),
        ("l_hip", 170, 1.0),
        ("r_hip", 170, 1.0),
        ("torso_vertical", 5, 1.5),
        ("l_shoulder", 15, 0.8),
        ("r_shoulder", 15, 0.8),
    ],
    "warrior_ii": [
        # Front knee bent ~90°, back leg straight, arms horizontal
        ("l_knee", 100, 1.2),    # one knee bent
        ("r_knee", 170, 1.0),   # other straight (or vice versa)
        ("l_shoulder", 90, 1.0),
        ("r_shoulder", 90, 1.0),
        ("l_elbow", 170, 0.8),  # arms straight
        ("r_elbow", 170, 0.8),
        ("torso_vertical", 5, 1.0),
    ],
    "warrior_i": [
        # Front knee bent ~90°, back leg straight, arms raised
        ("l_knee", 100, 1.2),
        ("r_knee", 170, 1.0),
        ("l_shoulder", 170, 1.0),  # arms up
        ("r_shoulder", 170, 1.0),
        ("l_elbow", 170, 0.8),
        ("r_elbow", 170, 0.8),
        ("torso_vertical", 5, 0.8),
    ],
    "tree": [
        # One leg straight, torso upright, arms up
        ("l_knee", 170, 0.8),  # standing leg straight
        ("torso_vertical", 5, 1.5),
        ("l_shoulder", 170, 1.0),
        ("r_shoulder", 170, 1.0),
        ("l_elbow", 170, 0.7),
        ("r_elbow", 170, 0.7),
    ],
    "downward_dog": [
        # Inverted V: arms and legs straight, hips high
        ("l_elbow", 170, 1.0),
        ("r_elbow", 170, 1.0),
        ("l_knee", 170, 1.0),
        ("r_knee", 170, 1.0),
        ("l_hip", 70, 1.5),   # acute hip angle (inverted V)
        ("r_hip", 70, 1.5),
        ("torso_vertical", 50, 1.0),  # torso is angled
    ],
}


def classify_pose(
    pose: PoseResult,
    confidence_threshold: float = 0.4,
) -> ClassificationResult:
    """Classify the detected body into a yoga pose.
    """Classify the detected body into a yoga pose.

    Returns the best-matching pose and per-pose scores.
    """
    """
    angles = _measure_angles(pose)
    scores: dict[str, float] = {}

    for pose_name, template in _TEMPLATES.items():
        score = _score_template(angles, template)
        scores[pose_name] = round(score, 3)

    best = max(scores, key=scores.get)
    best_score = scores[best]

    if best_score < confidence_threshold:
        return ClassificationResult(
            pose="unknown", confidence=best_score,
            scores=scores, angles=angles,
        )

    return ClassificationResult(
        pose=best, confidence=best_score,
        scores=scores, angles=angles,
    )


def _score_template(
    angles: dict[str, float],
    template: list[tuple[str, float, float]],
) -> float:
    """Score how well measured angles match a template.
    """Score how well measured angles match a template.

    Returns a value in [0, 1] where 1 = perfect match.
    Uses a Gaussian-like decay: score = exp(-diff²/σ²) weighted.
    """
    """
    import math

    sigma = 30.0  # degrees -- controls tolerance width
    total_weight = 0.0
    weighted_sum = 0.0

    for key, expected, weight in template:
        measured = angles.get(key, 0.0)
        # Handle symmetric poses: try both sides
        diff = abs(measured - expected)
        # Also try the opposite-side angle
        if key.startswith("l_"):
            alt_key = "r_" + key[2:]
        elif key.startswith("r_"):
            alt_key = "l_" + key[2:]
        else:
            alt_key = None
        if alt_key and alt_key in angles:
            alt_diff = abs(angles[alt_key] - expected)
            diff = min(diff, alt_diff)

        s = math.exp(-(diff ** 2) / (sigma ** 2))
        weighted_sum += s * weight
        total_weight += weight

    if total_weight < 1e-9:
        return 0.0
    return weighted_sum / total_weight
