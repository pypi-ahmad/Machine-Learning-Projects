"""Modern registry entry for Blink Headpose Analyzer.

Registers the project with ``core/registry.py`` so it can be
discovered and launched via the unified CLI.
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


@register("blink_headpose_analyzer")
class BlinkHeadposeModern(CVProject):
    """Blink counting and head pose estimation from facial landmarks."""

    project_type = "detection"
    description = (
        "MediaPipe Face Landmarker analysis for blink counting "
        "and head-pose estimation with reusable utilities"
    )
    legacy_tech = "N/A (new project)"
    modern_tech = "MediaPipe Face Landmarker + shared EAR/headpose utilities"

    def __init__(self) -> None:
        super().__init__()
        self._pipeline = None
        self._validator = None
        self.cfg = None

    def load(self) -> None:
        from analyzer import AnalyzerPipeline
        from config import AnalyzerConfig
        from validator import AnalyzerValidator

        self.cfg = AnalyzerConfig()
        self._pipeline = AnalyzerPipeline(self.cfg)
        self._pipeline.load()
        self._validator = AnalyzerValidator(self.cfg)
        self._loaded = True

    def predict(self, input_data, **kwargs):
        if not self._loaded:
            self.load()

        if isinstance(input_data, np.ndarray):
            result = self._pipeline.process(input_data)
            report = self._validator.validate(result)
            return {
                "result": result,
                "report": report,
                "face_detected": result.face_detected,
                "ear": result.blink.ear,
                "total_blinks": result.blink.total_blinks,
                "yaw": result.head_pose.yaw,
                "pitch": result.head_pose.pitch,
                "roll": result.head_pose.roll,
            }

        # Video / webcam path
        source = str(input_data)
        cap = cv2.VideoCapture(
            int(source) if source.isdigit() else source,
        )
        if not cap.isOpened():
            return {"error": f"Cannot open source: {source}"}

        results = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = self._pipeline.process(frame)
            results.append({
                "face_detected": result.face_detected,
                "ear": result.blink.ear,
                "blink_detected": result.blink.blink_detected,
                "yaw": result.head_pose.yaw,
                "pitch": result.head_pose.pitch,
            })
        cap.release()
        return {"frames": len(results), "results": results}

    def visualize(self, input_data, output, **kwargs):
        from visualize import draw_overlay

        if isinstance(input_data, np.ndarray):
            result = output.get("result")
            if result is None:
                return input_data.copy()
            return draw_overlay(input_data, result, self.cfg)
        return np.zeros((100, 400, 3), dtype=np.uint8)

    def setup(self, **kwargs) -> None:
        from analyzer import AnalyzerPipeline
        from config import AnalyzerConfig, load_config
        from validator import AnalyzerValidator

        config_path = kwargs.get("config")
        self.cfg = (
            load_config(config_path) if config_path else AnalyzerConfig()
        )
        if kwargs.get("ear_threshold"):
            self.cfg.ear_threshold = kwargs["ear_threshold"]
        if kwargs.get("yaw_threshold"):
            self.cfg.yaw_threshold = kwargs["yaw_threshold"]

        self._pipeline = AnalyzerPipeline(self.cfg)
        self._pipeline.load()
        self._validator = AnalyzerValidator(self.cfg)
        self._loaded = True

    def train(self, **kwargs) -> None:
        from train import main as train_main
        train_main()

    def evaluate(self, **kwargs) -> None:
        from train import main as eval_main
        eval_main()
