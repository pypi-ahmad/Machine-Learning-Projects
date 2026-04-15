"""Yoga Pose Correction Coach — rule-based correction hints.
"""Yoga Pose Correction Coach — rule-based correction hints.

Each pose has a set of checkpoints.  A checkpoint inspects a
specific measured angle and produces a human-readable hint if
the angle is outside the acceptable range.

Hints are intentionally phrased as gentle suggestions, not commands.
"""
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class CorrectionHint:
    """A single correction suggestion."""

    joint: str          # human-readable joint name (e.g. "left knee")
    hint: str           # correction text
    severity: str       # "minor" or "major"
    measured: float     # actual angle (degrees)
    expected: float     # target angle (degrees)


def generate_corrections(
    pose_name: str,
    angles: dict[str, float],
    tolerance: float = 15.0,
    max_hints: int = 3,
) -> list[CorrectionHint]:
    """Return correction hints for *pose_name* given measured *angles*.
    """Return correction hints for *pose_name* given measured *angles*.

    Hints are sorted by severity (major first), capped at *max_hints*.
    """
    """
    rules = _CORRECTION_RULES.get(pose_name)
    if rules is None:
        return []

    hints: list[CorrectionHint] = []
    for rule in rules:
        # Try both sides for symmetric joints
        angle_key = rule["key"]
        joint_label = rule["joint"]
        expected = rule["expected"]
        measured = angles.get(angle_key, None)

        # For symmetric poses check the side closer to expected
        if measured is None:
            continue

        diff = abs(measured - expected)
        if diff <= tolerance:
            continue

        severity = "major" if diff > tolerance * 2 else "minor"
        hint_text = rule["too_low"] if measured < expected else rule["too_high"]

        hints.append(CorrectionHint(
            joint=joint_label,
            hint=hint_text,
            severity=severity,
            measured=round(measured, 1),
            expected=expected,
        ))

    # Sort major first, then cap
    hints.sort(key=lambda h: (0 if h.severity == "major" else 1, -abs(h.measured - h.expected)))
    return hints[:max_hints]


# ---------------------------------------------------------------------------
# Per-pose correction rules
# ---------------------------------------------------------------------------

_CORRECTION_RULES: dict[str, list[dict]] = {
    "mountain": [
        {
            "key": "l_knee", "joint": "left knee", "expected": 170,
            "too_low": "Try to straighten your left leg fully.",
            "too_high": "Avoid locking your left knee -- keep a micro-bend.",
        },
        {
            "key": "r_knee", "joint": "right knee", "expected": 170,
            "too_low": "Try to straighten your right leg fully.",
            "too_high": "Avoid locking your right knee.",
        },
        {
            "key": "torso_vertical", "joint": "torso", "expected": 0,
            "too_low": "Good torso alignment.",
            "too_high": "Try to stand more upright -- tuck your tailbone.",
        },
        {
            "key": "shoulder_level", "joint": "shoulders", "expected": 0,
            "too_low": "Good shoulder alignment.",
            "too_high": "Level your shoulders -- one may be raised.",
        },
    ],
    "warrior_ii": [
        {
            "key": "l_knee", "joint": "front knee", "expected": 95,
            "too_low": "Bend your front knee a bit more toward 90°.",
            "too_high": "Your front knee may be too far forward -- try easing back.",
        },
        {
            "key": "l_shoulder", "joint": "left arm", "expected": 90,
            "too_low": "Raise your left arm to shoulder height.",
            "too_high": "Lower your left arm -- aim for shoulder height.",
        },
        {
            "key": "r_shoulder", "joint": "right arm", "expected": 90,
            "too_low": "Raise your right arm to shoulder height.",
            "too_high": "Lower your right arm -- aim for shoulder height.",
        },
        {
            "key": "l_elbow", "joint": "left elbow", "expected": 170,
            "too_low": "Try to straighten your left arm.",
            "too_high": "Good arm extension.",
        },
        {
            "key": "torso_vertical", "joint": "torso", "expected": 0,
            "too_low": "Good torso alignment.",
            "too_high": "Keep your torso upright -- avoid leaning forward.",
        },
    ],
    "warrior_i": [
        {
            "key": "l_knee", "joint": "front knee", "expected": 95,
            "too_low": "Bend your front knee a bit more toward 90°.",
            "too_high": "Your front knee may be past your toes -- ease back slightly.",
        },
        {
            "key": "l_shoulder", "joint": "left arm", "expected": 170,
            "too_low": "Raise your left arm higher overhead.",
            "too_high": "Good arm height.",
        },
        {
            "key": "r_shoulder", "joint": "right arm", "expected": 170,
            "too_low": "Raise your right arm higher overhead.",
            "too_high": "Good arm height.",
        },
        {
            "key": "torso_vertical", "joint": "torso", "expected": 0,
            "too_low": "Good torso alignment.",
            "too_high": "Keep your torso upright -- avoid leaning.",
        },
    ],
    "tree": [
        {
            "key": "l_knee", "joint": "standing leg", "expected": 170,
            "too_low": "Straighten your standing leg.",
            "too_high": "Avoid hyper-extending your standing knee.",
        },
        {
            "key": "torso_vertical", "joint": "torso", "expected": 0,
            "too_low": "Good balance.",
            "too_high": "Try to keep your torso more upright.",
        },
        {
            "key": "l_shoulder", "joint": "left arm", "expected": 170,
            "too_low": "Raise your arms overhead.",
            "too_high": "Good arm position.",
        },
        {
            "key": "hip_level", "joint": "hips", "expected": 0,
            "too_low": "Good hip alignment.",
            "too_high": "Try to level your hips -- one may be dropped.",
        },
    ],
    "downward_dog": [
        {
            "key": "l_elbow", "joint": "left elbow", "expected": 170,
            "too_low": "Try to straighten your left arm.",
            "too_high": "Avoid locking your left elbow.",
        },
        {
            "key": "r_elbow", "joint": "right elbow", "expected": 170,
            "too_low": "Try to straighten your right arm.",
            "too_high": "Avoid locking your right elbow.",
        },
        {
            "key": "l_knee", "joint": "left knee", "expected": 170,
            "too_low": "Work toward straightening your left leg.",
            "too_high": "Good leg extension.",
        },
        {
            "key": "r_knee", "joint": "right knee", "expected": 170,
            "too_low": "Work toward straightening your right leg.",
            "too_high": "Good leg extension.",
        },
        {
            "key": "l_hip", "joint": "hips", "expected": 70,
            "too_low": "Push your hips higher -- aim for an inverted V.",
            "too_high": "Your hips may be too high -- find the inverted V shape.",
        },
    ],
}
