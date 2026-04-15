"""Modern registry entry for ID Card KYC Parser.

Registers the project with the repo's ``core/registry.py`` decorator
so it can be discovered and launched via the unified CLI.

Pipeline: CardDetector -> EasyOCR -> Template parser -> Validator -> Export
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


@register("id_card_kyc_parser")
class IDCardKYCParser(CVProject):
    """KYC-style ID card detection, rectification, OCR, and field extraction."""

    project_type = "detection"
    description = "ID card detection + perspective correction + OCR field extraction"
    legacy_tech = "N/A (new project)"
    modern_tech = "Contour detection + EasyOCR + template-based field parsing"

    def __init__(self) -> None:
        super().__init__()
        self._detector = None
        self._engine = None
        self._parser = None
        self._validator = None
        self.cfg = None

    def load(self) -> None:
        from card_detector import CardDetector
        from config import IDCardConfig
        from ocr_engine import OCREngine
        from parser import IDCardParser
        from validator import IDCardValidator

        self.cfg = IDCardConfig()
        self._detector = CardDetector(self.cfg)
        self._engine = OCREngine(self.cfg)
        self._parser = IDCardParser(template=self.cfg.template)
        self._validator = IDCardValidator(self.cfg)
        self._loaded = True

    def predict(self, input_data, **kwargs):
        if not self._loaded:
            self.load()

        if isinstance(input_data, np.ndarray):
            img = input_data
        else:
            img = cv2.imread(str(input_data))

        det = self._detector.detect_and_rectify(img)
        ocr_input = det.rectified if det.rectified is not None else img
        blocks = self._engine.run(ocr_input)
        result = self._parser.parse(blocks)
        report = self._validator.validate(result, card_detected=det.found)

        return {
            "detection": det,
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
            detection=output["detection"],
        )

    def setup(self, **kwargs) -> None:
        from card_detector import CardDetector
        from config import IDCardConfig, load_config
        from ocr_engine import OCREngine
        from parser import IDCardParser
        from validator import IDCardValidator

        config_path = kwargs.get("config")
        self.cfg = load_config(config_path) if config_path else IDCardConfig()
        if kwargs.get("template"):
            self.cfg.template = kwargs["template"]
        if kwargs.get("lang"):
            self.cfg.ocr_lang = kwargs["lang"]
        if kwargs.get("gpu"):
            self.cfg.use_gpu = True
        self._detector = CardDetector(self.cfg)
        self._engine = OCREngine(self.cfg)
        self._parser = IDCardParser(template=self.cfg.template)
        self._validator = IDCardValidator(self.cfg)
        self._loaded = True

    def train(self, **kwargs) -> None:
        from train import main as train_main
        train_main()

    def evaluate(self, **kwargs) -> None:
        from train import main as eval_main
        eval_main()
