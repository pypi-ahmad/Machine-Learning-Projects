"""Modern registry entry for Scene Text Reader Translator.

Registers the project with the repo's ``core/registry.py`` decorator
so it can be discovered and launched via the unified CLI.

Pipeline: PaddleOCR-first OCR → optional translation hook → structured output.
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


@register("scene_text_reader_translator")
class SceneTextReaderTranslator(CVProject):
    """Scene text OCR + optional translation pipeline."""

    project_type = "ocr"
    description = (
        "Scene text detection and recognition (PaddleOCR-first OCR) with "
        "optional translation hook"
    )
    legacy_tech = "N/A (new project)"
    modern_tech = "PaddleOCR-first scene text detection + recognition + translation hook"

    def __init__(self) -> None:
        super().__init__()
        self._pipeline = None
        self._validator = None
        self.cfg = None

    def load(self) -> None:
        from config import SceneTextConfig
        from parser import SceneTextPipeline
        from validator import SceneTextValidator

        self.cfg = SceneTextConfig()
        self._pipeline = SceneTextPipeline(self.cfg)
        self._validator = SceneTextValidator(self.cfg)
        self._loaded = True

    def predict(self, input_data, **kwargs):
        if not self._loaded:
            self.load()

        if isinstance(input_data, np.ndarray):
            img = input_data
        else:
            img = cv2.imread(str(input_data))

        result, blocks = self._pipeline.process_with_blocks(img)
        report = self._validator.validate(result)

        return {
            "result": result,
            "blocks": blocks,
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
        )

    def setup(self, **kwargs) -> None:
        from config import SceneTextConfig, load_config
        from parser import SceneTextPipeline
        from validator import SceneTextValidator

        config_path = kwargs.get("config")
        self.cfg = load_config(config_path) if config_path else SceneTextConfig()
        if kwargs.get("lang"):
            self.cfg.ocr_lang = kwargs["lang"]
        if kwargs.get("gpu"):
            self.cfg.use_gpu = True
        if kwargs.get("translate"):
            self.cfg.translate_enabled = True
        if kwargs.get("target_lang"):
            self.cfg.translate_target_lang = kwargs["target_lang"]
            self.cfg.translate_enabled = True
        self._pipeline = SceneTextPipeline(self.cfg)
        self._validator = SceneTextValidator(self.cfg)
        self._loaded = True

    def train(self, **kwargs) -> None:
        from train import main as train_main
        train_main()

    def evaluate(self, **kwargs) -> None:
        from train import main as eval_main
        eval_main()
