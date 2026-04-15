"""Modern registry entry for Face Verification Attendance System.

Registers the project with the repo's ``core/registry.py`` decorator
so it can be discovered and launched via the unified CLI.

Pipeline: YOLO face detect -> InsightFace embedding -> cosine match -> attendance log.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


@register("face_verification_attendance")
class FaceVerificationAttendanceModern(CVProject):
    """Face verification attendance -- enrollment + verification pipeline."""

    project_type = "detection"
    description = (
        "Face detection + InsightFace ArcFace embeddings for "
        "enrollment-based attendance verification"
    )
    legacy_tech = "N/A (new project)"
    modern_tech = "YOLO face detection + InsightFace ArcFace embeddings"

    def __init__(self) -> None:
        super().__init__()
        self._pipeline = None
        self._validator = None
        self.cfg = None

    def load(self) -> None:
        from config import FaceAttendanceConfig
        from parser import FaceAttendancePipeline
        from validator import AttendanceValidator

        self.cfg = FaceAttendanceConfig()
        self._pipeline = FaceAttendancePipeline(self.cfg)
        self._pipeline.load()
        self._validator = AttendanceValidator(self.cfg)
        self._loaded = True

    def predict(self, input_data, **kwargs):
        if not self._loaded:
            self.load()

        if isinstance(input_data, np.ndarray):
            img = input_data
        else:
            img = cv2.imread(str(input_data))

        result = self._pipeline.process(img)
        report = self._validator.validate(
            result,
            gallery_size=self._pipeline.matcher.gallery_size,
        )

        return {
            "result": result,
            "report": report,
            # Legacy-compatible keys
            "faces": [
                {
                    "box": m.box,
                    "identity": m.identity,
                    "similarity": m.similarity,
                    "matched": m.matched,
                    "det_confidence": m.det_confidence,
                }
                for m in result.matches
            ],
            "count": result.num_faces,
        }

    def visualize(self, input_data, output, **kwargs):
        from visualize import draw_overlay

        if isinstance(input_data, np.ndarray):
            img = input_data
        else:
            img = cv2.imread(str(input_data))

        return draw_overlay(
            img,
            output["result"],
            self.cfg,
            recent_attendance=self._pipeline.logger.recent_identities(),
        )

    def enroll(self, name: str, image) -> bool:
        """Enroll a face identity (convenience wrapper)."""
        if not self._loaded:
            self.load()
        img = image if isinstance(image, np.ndarray) else str(image)
        return self._pipeline.enroll_single(name, img)

    def load_gallery(self, path: str | None = None) -> bool:
        """Load enrolled gallery from disk."""
        if not self._loaded:
            self.load()
        return self._pipeline.load_gallery(path)

    def save_gallery(self, path: str | None = None) -> None:
        """Save enrolled gallery to disk."""
        self._pipeline.enrollment.save(path)

    def export_attendance_csv(self, path: str) -> None:
        """Export attendance log to CSV."""
        self._pipeline.logger.save_csv(path)

    def setup(self, **kwargs) -> None:
        from config import FaceAttendanceConfig, load_config
        from parser import FaceAttendancePipeline
        from validator import AttendanceValidator

        config_path = kwargs.get("config")
        self.cfg = load_config(config_path) if config_path else FaceAttendanceConfig()
        if kwargs.get("gallery_dir"):
            self.cfg.gallery_dir = kwargs["gallery_dir"]
        if kwargs.get("threshold"):
            self.cfg.similarity_threshold = kwargs["threshold"]
        self._pipeline = FaceAttendancePipeline(self.cfg)
        self._pipeline.load()
        self._validator = AttendanceValidator(self.cfg)
        self._loaded = True

    def train(self, **kwargs) -> None:
        from train import main as train_main
        train_main()

    def evaluate(self, **kwargs) -> None:
        from train import main as eval_main
        eval_main()
