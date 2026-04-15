"""Modern registry entry for Prescription OCR Parser.

Registers the project with the repo's ``core/registry.py`` decorator
so it can be discovered and launched via the unified CLI.

Pipeline: PaddleOCR-first OCR -> field extraction -> structured Rx output.

**DISCLAIMER:** This tool is for informational and educational
purposes only.  It does not provide medical advice.
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


@register("prescription_ocr_parser")
class PrescriptionOCRParser(CVProject):
    """Prescription OCR + medicine field extraction (informational only)."""

    project_type = "ocr"
    description = (
        "Prescription OCR with medicine name, dosage, frequency extraction "
        "(informational only -- not for clinical use)"
    )
    legacy_tech = "N/A (new project)"
    modern_tech = "PaddleOCR-first OCR + pattern-based medicine field extraction"

    def __init__(self) -> None:
        super().__init__()
        self._parser = None
        self._validator = None
        self.cfg = None

    def load(self) -> None:
        from config import PrescriptionConfig
        from parser import PrescriptionParser
        from validator import PrescriptionValidator

        self.cfg = PrescriptionConfig()
        self._parser = PrescriptionParser(self.cfg)
        self._validator = PrescriptionValidator(self.cfg)
        self._loaded = True

    def predict(self, input_data, **kwargs):
        if not self._loaded:
            self.load()

        if isinstance(input_data, np.ndarray):
            img = input_data
        else:
            img = cv2.imread(str(input_data))

        result, blocks = self._parser.parse_with_blocks(img)
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
        from config import PrescriptionConfig, load_config
        from parser import PrescriptionParser
        from validator import PrescriptionValidator

        config_path = kwargs.get("config")
        self.cfg = (
            load_config(config_path) if config_path else PrescriptionConfig()
        )
        if kwargs.get("lang"):
            self.cfg.ocr_lang = kwargs["lang"]
        if kwargs.get("gpu"):
            self.cfg.use_gpu = True
        self._parser = PrescriptionParser(self.cfg)
        self._validator = PrescriptionValidator(self.cfg)
        self._loaded = True

    def train(self, **kwargs) -> None:
        from train import main as train_main
        train_main()

    def evaluate(self, **kwargs) -> None:
        from train import main as eval_main
        eval_main()
