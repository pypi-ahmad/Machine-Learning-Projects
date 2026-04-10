"""Modern registry entry for Form OCR Checkbox Extractor.

Registers the project with the repo's ``core/registry.py`` decorator
so it can be discovered and launched via the unified CLI.

Pipeline: PaddleOCR + OpenCV checkbox detection → label association → export
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


@register("form_ocr_checkbox_extractor")
class FormOCRCheckboxExtractor(CVProject):
    """Form checkbox/radio detection + OCR text extraction."""

    project_type = "detection"
    description = "Checkbox/radio detection + PaddleOCR form field extraction"
    legacy_tech = "N/A (new project)"
    modern_tech = "OpenCV morphology + PaddleOCR + label association"

    def __init__(self) -> None:
        super().__init__()
        self._parser = None
        self._validator = None
        self.cfg = None

    def load(self) -> None:
        from config import FormCheckboxConfig
        from parser import FormParser
        from validator import FormValidator

        self.cfg = FormCheckboxConfig()
        self._parser = FormParser(self.cfg)
        self._validator = FormValidator(self.cfg)
        self._loaded = True

    def predict(self, input_data, **kwargs):
        if not self._loaded:
            self.load()

        if isinstance(input_data, np.ndarray):
            img = input_data
        else:
            img = cv2.imread(str(input_data))

        result, blocks, controls = self._parser.parse_with_details(img)
        report = self._validator.validate(result)

        return {
            "result": result,
            "blocks": blocks,
            "controls": controls,
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
            ocr_blocks=output["blocks"],
            controls=output["controls"],
        )

    def setup(self, **kwargs) -> None:
        from config import FormCheckboxConfig, load_config
        from parser import FormParser
        from validator import FormValidator

        config_path = kwargs.get("config")
        self.cfg = load_config(config_path) if config_path else FormCheckboxConfig()
        if kwargs.get("lang"):
            self.cfg.ocr_lang = kwargs["lang"]
        if kwargs.get("gpu"):
            self.cfg.use_gpu = True
        if kwargs.get("fill_threshold"):
            self.cfg.fill_threshold = kwargs["fill_threshold"]
        self._parser = FormParser(self.cfg)
        self._validator = FormValidator(self.cfg)
        self._loaded = True

    def train(self, **kwargs) -> None:
        from train import main as train_main
        train_main()

    def evaluate(self, **kwargs) -> None:
        from train import main as eval_main
        eval_main()
