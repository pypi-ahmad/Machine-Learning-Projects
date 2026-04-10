"""Exercise Rep Counter — rep-counting state machine."""

from __future__ import annotations

import dataclasses

from exercise_rules import ExerciseAnalysis


@dataclasses.dataclass
class RepState:
    """Current state of the rep counter."""

    reps: int
    stage: str          # current confirmed stage ("up" / "down" / "unknown")
    prev_stage: str     # previous confirmed stage
    angle: float        # current (possibly smoothed) angle
    exercise: str


class RepCounter:
    """Counts reps by detecting stage transitions.

    A rep is counted on a *down → up* transition (the completion
    of the concentric phase).

    The optional *stable_frames* parameter requires the same stage
    to persist for N consecutive frames before confirming a
    transition, filtering out single-frame noise.
    """

    def __init__(self, stable_frames: int = 2) -> None:
        self._stable_frames = stable_frames
        self._reps = 0
        self._stage = "unknown"
        self._prev_stage = "unknown"
        self._pending_stage = "unknown"
        self._pending_count = 0

    def update(self, analysis: ExerciseAnalysis) -> RepState:
        """Feed one frame's analysis and return the current state."""
        raw_stage = analysis.stage

        # Stability filter
        if raw_stage == self._pending_stage:
            self._pending_count += 1
        else:
            self._pending_stage = raw_stage
            self._pending_count = 1

        if (
            self._pending_count >= self._stable_frames
            and self._pending_stage != self._stage
            and self._pending_stage != "unknown"
        ):
            self._prev_stage = self._stage
            self._stage = self._pending_stage

            # Count rep on down → up transition
            if self._prev_stage == "down" and self._stage == "up":
                self._reps += 1

        return RepState(
            reps=self._reps,
            stage=self._stage,
            prev_stage=self._prev_stage,
            angle=analysis.angle,
            exercise=analysis.exercise,
        )

    def reset(self) -> None:
        self._reps = 0
        self._stage = "unknown"
        self._prev_stage = "unknown"
        self._pending_stage = "unknown"
        self._pending_count = 0

    @property
    def reps(self) -> int:
        return self._reps
