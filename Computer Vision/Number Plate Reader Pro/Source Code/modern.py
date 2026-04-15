"""Modern registry entry for Number Plate Reader Pro.

Registers the project with the repo's ``core/registry.py`` decorator
so it can be discovered and launched via the unified CLI.

Pipeline: YOLO26m detect → crop + rectify → PaddleOCR-first OCR → regex cleanup → dedup.
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


@register("number_plate_reader_pro")
class NumberPlateReaderProModern(CVProject):
    """License plate detection + OCR + dedup pipeline."""

    project_type = "detection"
    description = "License plate detection (YOLO26m) + OCR (PaddleOCR-first) + frame dedup"
    legacy_tech = "N/A (new project)"
    modern_tech = "YOLO26m plate detection + PaddleOCR-first recognition + regex cleanup"

    def __init__(self) -> None:
        super().__init__()
        self._pipeline = None
        self._validator = None
        self.cfg = None

    def load(self) -> None:
        from config import PlateConfig
        from parser import PlateReaderPipeline
        from validator import PlateValidator

        self.cfg = PlateConfig()
        self._pipeline = PlateReaderPipeline(self.cfg)
        self._validator = PlateValidator(self.cfg)
        self._loaded = True

    def predict(self, input_data, **kwargs):
        if not self._loaded:
            self.load()

        if isinstance(input_data, np.ndarray):
            frame = input_data
        else:
            frame = cv2.imread(str(input_data))

        result = self._pipeline.process_frame(frame)
        report = self._validator.validate(result)

        return {
            "result": result,
            "report": report,
            "frame": frame,
        }

    def visualize(self, input_data, output, **kwargs):
        from visualize import draw_overlay

        frame = output.get("frame")
        if frame is None:
            if isinstance(input_data, np.ndarray):
                frame = input_data
            else:
                frame = cv2.imread(str(input_data))

        return draw_overlay(frame, output["result"], self.cfg)

    def setup(self, **kwargs) -> None:
        from config import PlateConfig, load_config
        from parser import PlateReaderPipeline
        from validator import PlateValidator

        config_path = kwargs.get("config")
        self.cfg = load_config(config_path) if config_path else PlateConfig()
        if kwargs.get("gpu"):
            self.cfg.use_gpu = True
        self._pipeline = PlateReaderPipeline(self.cfg)
        self._validator = PlateValidator(self.cfg)
        self._loaded = True

    def train(self, **kwargs) -> None:
        from train import main as train_main
        train_main()

    def evaluate(self, **kwargs) -> None:
        from train import main as eval_main
        eval_main()
