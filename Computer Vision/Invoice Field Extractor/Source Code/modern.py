"""Modern registry entry for Invoice Field Extractor.

Registers the project with the repo's ``core/registry.py`` decorator
so it can be discovered and launched via the unified CLI.

Pipeline: PaddleOCR → OCREngine → InvoiceParser → Validator → Export
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


@register("invoice_field_extractor")
class InvoiceFieldExtractorModern(CVProject):
    """OCR-based invoice field extraction with validation and export."""

    project_type = "detection"
    description = "Invoice OCR + rule-based field extraction to JSON"
    legacy_tech = "N/A (new project)"
    modern_tech = "PaddleOCR detection + recognition + regex field parsing"

    def __init__(self) -> None:
        super().__init__()
        self._engine = None
        self._parser = None
        self._validator = None
        self.cfg = None

    def load(self) -> None:
        from config import InvoiceConfig
        from ocr_engine import OCREngine
        from parser import InvoiceParser
        from validator import InvoiceValidator

        self.cfg = InvoiceConfig()
        self._engine = OCREngine(self.cfg)
        self._parser = InvoiceParser()
        self._validator = InvoiceValidator(self.cfg)
        self._loaded = True

    def predict(self, input_data, **kwargs):
        if not self._loaded:
            self.load()

        if isinstance(input_data, np.ndarray):
            img = input_data
        else:
            img = cv2.imread(str(input_data))

        blocks = self._engine.run(img)
        result = self._parser.parse(blocks)
        report = self._validator.validate(result)

        return {
            "blocks": blocks,
            "result": result,
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
        from config import InvoiceConfig, load_config
        from ocr_engine import OCREngine
        from parser import InvoiceParser
        from validator import InvoiceValidator

        config_path = kwargs.get("config")
        self.cfg = load_config(config_path) if config_path else InvoiceConfig()
        if kwargs.get("lang"):
            self.cfg.ocr_lang = kwargs["lang"]
        if kwargs.get("gpu"):
            self.cfg.use_gpu = True
        self._engine = OCREngine(self.cfg)
        self._parser = InvoiceParser()
        self._validator = InvoiceValidator(self.cfg)
        self._loaded = True

    def train(self, **kwargs) -> None:
        from train import main as train_main
        train_main()

    def evaluate(self, **kwargs) -> None:
        from train import main as eval_main
        eval_main()
