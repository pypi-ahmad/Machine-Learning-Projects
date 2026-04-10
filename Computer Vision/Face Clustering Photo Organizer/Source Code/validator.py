"""Validation module for Face Clustering Photo Organizer.

Checks pipeline results for common issues:
no faces detected, single-face images, low confidence.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import FaceClusterConfig
from parser import ClusterResult


@dataclass
class Warning:
    """Single validation warning."""

    field_name: str
    message: str


@dataclass
class ValidationReport:
    """Pipeline validation report."""

    valid: bool = True
    warnings: list[Warning] = field(default_factory=list)

    def summary(self) -> str:
        if self.valid and not self.warnings:
            return "OK"
        parts = []
        if not self.valid:
            parts.append("INVALID")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warning(s)")
        return "; ".join(parts)


class ClusterValidator:
    """Validate clustering pipeline results."""

    def __init__(self, cfg: FaceClusterConfig) -> None:
        self.cfg = cfg

    def validate(self, result: ClusterResult) -> ValidationReport:
        """Validate a pipeline result.

        Parameters
        ----------
        result : ClusterResult
            Pipeline output.

        Returns
        -------
        ValidationReport
        """
        report = ValidationReport()

        if self.cfg.warn_no_faces and result.total_faces == 0:
            report.warnings.append(Warning(
                "faces", "No faces detected in any image",
            ))

        if result.images_without_faces > 0:
            report.warnings.append(Warning(
                "images",
                f"{result.images_without_faces} image(s) had no detectable faces",
            ))

        if result.num_clusters == 0 and result.total_faces > 0:
            report.warnings.append(Warning(
                "clusters",
                "No clusters formed — all faces may be singletons "
                "(try lowering distance_threshold)",
            ))

        if self.cfg.warn_single_face and result.num_clusters == 1:
            report.warnings.append(Warning(
                "clusters",
                "Only one cluster formed — all faces grouped as one identity",
            ))

        if report.warnings:
            report.valid = False

        return report
