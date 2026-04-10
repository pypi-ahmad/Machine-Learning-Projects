"""Modern registry entry for Gesture Controlled Slideshow.

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


@register("gesture_controlled_slideshow")
class GestureSlideshowModern(CVProject):
    """Hand-gesture controlled slideshow via MediaPipe Hands."""

    project_type = "detection"
    description = (
        "MediaPipe hand landmark gesture recognition for slideshow "
        "control with debouncing and keyboard fallback"
    )
    legacy_tech = "N/A (new project)"
    modern_tech = "MediaPipe Hands + finger-state gesture classifier"

    def __init__(self) -> None:
        super().__init__()
        self._controller = None
        self._validator = None
        self.cfg = None

    def load(self) -> None:
        from config import GestureConfig
        from controller import SlideshowController
        from validator import GestureValidator

        self.cfg = GestureConfig()
        self._controller = SlideshowController(self.cfg)
        self._controller.load()
        self._validator = GestureValidator(self.cfg)
        self._loaded = True

    def predict(self, input_data, **kwargs):
        if not self._loaded:
            self.load()

        if isinstance(input_data, np.ndarray):
            result = self._controller.process(input_data)
            report = self._validator.validate(result)
            return {
                "result": result,
                "report": report,
                "hand_detected": result.hand_detected,
                "gesture": result.gesture.gesture,
                "finger_count": result.gesture.finger_count,
                "action": result.debounced.action,
                "slide_index": result.slide.current_index,
            }

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
            result = self._controller.process(frame)
            results.append({
                "hand_detected": result.hand_detected,
                "gesture": result.gesture.gesture,
                "action": result.debounced.action,
            })
        cap.release()
        return {"frames": len(results), "results": results}

    def visualize(self, input_data, output, **kwargs):
        from visualize import draw_overlay

        if isinstance(input_data, np.ndarray):
            result = output.get("result")
            if result is None:
                return input_data.copy()
            return draw_overlay(
                input_data, result, self.cfg,
                detector=self._controller.detector,
            )
        return np.zeros((100, 400, 3), dtype=np.uint8)

    def setup(self, **kwargs) -> None:
        from config import GestureConfig, load_config
        from controller import SlideshowController
        from validator import GestureValidator

        config_path = kwargs.get("config")
        self.cfg = (
            load_config(config_path) if config_path else GestureConfig()
        )
        slide_dir = kwargs.get("slides")
        self._controller = SlideshowController(self.cfg)
        self._controller.load(slide_dir=slide_dir)
        self._validator = GestureValidator(self.cfg)
        self._loaded = True

    def train(self, **kwargs) -> None:
        from train import main as train_main
        train_main()

    def evaluate(self, **kwargs) -> None:
        from train import main as eval_main
        eval_main()
