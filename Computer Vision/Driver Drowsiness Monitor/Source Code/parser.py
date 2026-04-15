"""High-level pipeline for Driver Drowsiness Monitor.

Orchestrates: landmark detection → blink/EAR → yawn/MAR →
head pose → alert management.

Usage::

    from parser import DrowsinessPipeline

    pipeline = DrowsinessPipeline(cfg)
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

from alert_manager import AlertEvent, AlertManager
from blink_tracker import BlinkState, BlinkTracker
from config import DrowsinessConfig
from head_pose import HeadPoseEstimator, HeadPoseState
from landmark_detector import LandmarkDetector, LandmarkResult
from yawn_tracker import YawnState, YawnTracker

log = logging.getLogger("drowsiness.parser")


@dataclass
class DrowsinessResult:
    """Complete pipeline result for a single frame."""

    landmarks: LandmarkResult = field(default_factory=LandmarkResult)
    blink: BlinkState = field(default_factory=BlinkState)
    yawn: YawnState = field(default_factory=YawnState)
    head_pose: HeadPoseState = field(default_factory=HeadPoseState)
    alerts: list[AlertEvent] = field(default_factory=list)
    active_alerts: set[str] = field(default_factory=set)
    face_detected: bool = False


class DrowsinessPipeline:
    """Full drowsiness monitoring pipeline.

    landmarks → blink EAR → yawn MAR → head pose → alerts
    """

    def __init__(self, cfg: DrowsinessConfig) -> None:
        self.cfg = cfg
        self.detector = LandmarkDetector(cfg)
        self.blink_tracker = BlinkTracker(cfg)
        self.yawn_tracker = YawnTracker(cfg)
        self.head_pose = HeadPoseEstimator(cfg)
        self.alert_manager = AlertManager(cfg)
        self._loaded = False

    def load(self) -> None:
        """Initialize all pipeline components."""
        ok = self.detector.load()
        self._loaded = ok
        if ok:
            log.info("Drowsiness pipeline ready")
        else:
            log.error("Pipeline failed to load -- MediaPipe unavailable")

    def process(self, frame: np.ndarray) -> DrowsinessResult:
        """Process a single frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H, W, 3).

        Returns
        -------
        DrowsinessResult
        """
        if not self._loaded:
            self.load()

        result = DrowsinessResult()

        # 1. Detect landmarks
        lm = self.detector.detect(frame)
        result.landmarks = lm
        result.face_detected = lm.detected

        if not lm.detected:
            return result

        # 2. Blink / EAR tracking
        blink_state = self.blink_tracker.update(lm)
        result.blink = blink_state

        # 3. Yawn / MAR tracking
        yawn_state = self.yawn_tracker.update(lm)
        result.yawn = yawn_state

        # 4. Head pose estimation
        pose_state = self.head_pose.update(lm)
        result.head_pose = pose_state

        # 5. Alert evaluation
        alerts = self.alert_manager.check_and_alert(
            prolonged_closure=blink_state.prolonged_closure,
            drowsy_by_perclos=blink_state.drowsy_by_perclos,
            perclos=blink_state.perclos,
            yawn_detected=yawn_state.yawn_detected,
            distracted=pose_state.distracted,
            yaw=pose_state.yaw,
            ear=blink_state.ear,
        )
        result.alerts = alerts
        result.active_alerts = self.alert_manager.active_alerts

        return result

    def reset(self) -> None:
        """Reset all tracker state."""
        self.blink_tracker.reset()
        self.yawn_tracker.reset()
        self.head_pose.reset()
        self.alert_manager.reset()
