"""Modern registry entry for Exam Sheet Parser.

Registers the project with the repo's ``core/registry.py`` decorator
so it can be discovered and launched via the unified CLI.

Pipeline: PaddleOCR → layout classification → question extraction.
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


@register("exam_sheet_parser")
class ExamSheetParserModern(CVProject):
    """Exam sheet OCR + layout-aware question extraction."""

    project_type = "ocr"
    description = (
        "Exam sheet OCR with layout-aware parsing: headings, "
        "questions, MCQ options, marks extraction"
    )
    legacy_tech = "N/A (new project)"
    modern_tech = "PaddleOCR + rule-based layout classification"

    def __init__(self) -> None:
        super().__init__()
        self._pipeline = None
        self._validator = None
        self.cfg = None

    def load(self) -> None:
        from config import ExamSheetConfig
        from parser import ExamSheetPipeline
        from validator import ExamSheetValidator

        self.cfg = ExamSheetConfig()
        self._pipeline = ExamSheetPipeline(self.cfg)
        self._validator = ExamSheetValidator(self.cfg)
        self._loaded = True

    def predict(self, input_data, **kwargs):
        if not self._loaded:
            self.load()

        if isinstance(input_data, np.ndarray):
            img = input_data
        else:
            img = cv2.imread(str(input_data))

        result, blocks, elements = self._pipeline.process_with_details(img)
        report = self._validator.validate(result)

        return {
            "result": result,
            "blocks": blocks,
            "elements": elements,
            "report": report,
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
            elements=output["elements"],
        )

    def setup(self, **kwargs) -> None:
        from config import ExamSheetConfig, load_config
        from parser import ExamSheetPipeline
        from validator import ExamSheetValidator

        config_path = kwargs.get("config")
        self.cfg = load_config(config_path) if config_path else ExamSheetConfig()
        if kwargs.get("lang"):
            self.cfg.ocr_lang = kwargs["lang"]
        if kwargs.get("gpu"):
            self.cfg.use_gpu = True
        self._pipeline = ExamSheetPipeline(self.cfg)
        self._validator = ExamSheetValidator(self.cfg)
        self._loaded = True

    def train(self, **kwargs) -> None:
        from train import main as train_main
        train_main()

    def evaluate(self, **kwargs) -> None:
        from train import main as eval_main
        eval_main()
