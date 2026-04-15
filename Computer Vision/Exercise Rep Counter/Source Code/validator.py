"""Exercise Rep Counter -- quality validator."""

from __future__ import annotations

import dataclasses

from pose_detector import PoseResult


@dataclasses.dataclass
class ValidationReport:
    ok: bool
    warnings: list[str]


class FrameValidator:
    """Lightweight quality checks on pose detection output."""

    def __init__(self, min_visibility: float = 0.5) -> None:
        self._min_vis = min_visibility

    def validate(
        self,
        pose: PoseResult,
        landmark_indices: tuple[int, ...] = (),
    ) -> ValidationReport:
        warnings: list[str] = []
        if not pose.detected:
            warnings.append("No pose detected")
        elif landmark_indices:
            for idx in landmark_indices:
                if pose.vis(idx) < self._min_vis:
                    warnings.append(
                        f"Landmark {idx} low visibility ({pose.vis(idx):.2f})"
                    )
        return ValidationReport(ok=len(warnings) == 0, warnings=warnings)
