"""High-level pipeline for Face Verification Attendance System.

Orchestrates: detect → embed → match → log into a single
:class:`AttendanceResult`.

Usage::

    from parser import FaceAttendancePipeline

    pipeline = FaceAttendancePipeline(cfg)
    pipeline.load()
    result = pipeline.process(frame)
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attendance_log import AttendanceLogger
from config import FaceAttendanceConfig
from embedder import FaceEmbedder
from enrollment import EnrollmentManager
from face_detector import FaceDetector
from matcher import FaceMatcher, MatchResult

log = logging.getLogger("face_attendance.parser")


@dataclass
class AttendanceResult:
    """Complete pipeline result for a single frame."""

    matches: list[MatchResult] = field(default_factory=list)
    num_faces: int = 0
    num_matched: int = 0
    num_unknown: int = 0
    attendance_logged: list[str] = field(default_factory=list)
    backend: str = ""


class FaceAttendancePipeline:
    """Full face verification attendance pipeline.

    detect faces → extract embeddings → match against gallery → log attendance
    """

    def __init__(self, cfg: FaceAttendanceConfig) -> None:
        self.cfg = cfg
        self.embedder = FaceEmbedder(cfg)
        self.detector = FaceDetector(cfg)
        self.enrollment = EnrollmentManager(
            cfg,
            embedder=self.embedder,
            detector=self.detector,
        )
        self.matcher = FaceMatcher(cfg)
        self.logger = AttendanceLogger(cfg)
        self._loaded = False

    def load(self) -> None:
        """Initialize all pipeline components."""
        # Load embedder first (provides InsightFace app)
        insightface_app = self.embedder.load()

        # Load detector (can share InsightFace app)
        det_backend = self.detector.load(insightface_app=insightface_app)

        self._loaded = True
        log.info(
            "Pipeline ready: det=%s, emb=%s",
            det_backend,
            "insightface" if self.embedder.ready else "none",
        )

    def load_gallery(self, path: str | Path | None = None) -> bool:
        """Load gallery into matcher."""
        ok = self.enrollment.load(path)
        if ok:
            self.matcher.set_gallery(self.enrollment.gallery)
        return ok

    def enroll(self, name: str, images: list) -> bool:
        """Enroll an identity and update matcher gallery."""
        ok = self.enrollment.enroll(name, images)
        if ok:
            self.matcher.set_gallery(self.enrollment.gallery)
        return ok

    def enroll_single(self, name: str, image) -> bool:
        """Enroll from a single image."""
        ok = self.enrollment.enroll_single(name, image)
        if ok:
            self.matcher.set_gallery(self.enrollment.gallery)
        return ok

    def process(self, frame: np.ndarray) -> AttendanceResult:
        """Process a single frame through the full pipeline.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H, W, 3).

        Returns
        -------
        AttendanceResult
        """
        if not self._loaded:
            self.load()

        if not self.embedder.ready:
            return AttendanceResult(backend="none")

        # 1. Detect faces, then embed each detected crop.
        detected_faces = self.detector.detect(frame)
        face_data: list[dict] = []

        for face in detected_faces:
            embedding = self.embedder.extract_single(face.crop)
            if embedding is None:
                continue
            face_data.append(
                {
                    "box": face.box,
                    "confidence": face.confidence,
                    "embedding": embedding,
                }
            )

        if not face_data and self.detector.backend == "insightface":
            face_data = self.embedder.extract_from_frame(frame)

        # 2. Match each face
        matches = self.matcher.match_batch(face_data)

        # 3. Log attendance
        logged: list[str] = []
        for m in matches:
            if m.matched and self.logger.log(m.identity, m.similarity):
                logged.append(m.identity)

        num_matched = sum(1 for m in matches if m.matched)

        return AttendanceResult(
            matches=matches,
            num_faces=len(matches),
            num_matched=num_matched,
            num_unknown=len(matches) - num_matched,
            attendance_logged=logged,
            backend=self.detector.backend or "insightface",
        )
