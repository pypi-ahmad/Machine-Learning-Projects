"""Modern registry entry for Handwritten Note to Markdown.

Registers the project with the repo's ``core/registry.py`` decorator
so it can be discovered and launched via the unified CLI.

Pipeline: Line segmentation → TrOCR → Markdown formatting
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


@register("handwritten_note_to_markdown")
class HandwrittenNoteToMarkdown(CVProject):
    """Handwritten text recognition → Markdown / plain-text output."""

    project_type = "ocr"
    description = "Handwritten note OCR using TrOCR with markdown formatting"
    legacy_tech = "N/A (new project)"
    modern_tech = "TrOCR (VisionEncoderDecoder) + projection-profile line segmentation"

    def __init__(self) -> None:
        super().__init__()
        self._parser = None
        self._validator = None
        self.cfg = None

    def load(self) -> None:
        from config import NoteConfig
        from parser import NoteParser
        from validator import NoteValidator

        self.cfg = NoteConfig()
        self._parser = NoteParser(self.cfg)
        self._validator = NoteValidator(self.cfg)
        self._loaded = True

    def predict(self, input_data, **kwargs):
        if not self._loaded:
            self.load()

        if isinstance(input_data, np.ndarray):
            img = input_data
        else:
            img = cv2.imread(str(input_data))

        result = self._parser.parse(img)
        report = self._validator.validate(result)

        return {
            "result": result,
            "report": report,
        }

    def visualize(self, input_data, output, **kwargs):
        from visualize import draw_overlay

        if isinstance(input_data, np.ndarray):
            img = input_data
        else:
            img = cv2.imread(str(input_data))

        return draw_overlay(img, output["result"], self.cfg)

    def setup(self, **kwargs) -> None:
        from config import NoteConfig, load_config
        from parser import NoteParser
        from validator import NoteValidator

        config_path = kwargs.get("config")
        self.cfg = load_config(config_path) if config_path else NoteConfig()
        if kwargs.get("model"):
            self.cfg.model_name = kwargs["model"]
        if kwargs.get("gpu"):
            self.cfg.use_gpu = True
        if kwargs.get("no_segment"):
            self.cfg.enable_segmentation = False
        self._parser = NoteParser(self.cfg)
        self._validator = NoteValidator(self.cfg)
        self._loaded = True

    def train(self, **kwargs) -> None:
        from train import main as train_main
        train_main()

    def evaluate(self, **kwargs) -> None:
        from train import main as eval_main
        eval_main()
