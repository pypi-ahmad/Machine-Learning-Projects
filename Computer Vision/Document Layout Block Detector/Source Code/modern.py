"""Modern registry entry for Document Layout Block Detector.

Registers the project with the repo's ``core/registry.py`` decorator
so it can be discovered and launched via the unified CLI.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.base import CVProject
from core.registry import register


@register("document_layout_block_detector")
class DocumentLayoutBlockDetector(CVProject):
    """Detect document-layout blocks (titles, tables, figures, etc.) in scanned pages."""

    project_type = "detection"
    description = "Document layout analysis with block-level detection"
    modern_tech = "Ultralytics YOLO26m"

    def __init__(self) -> None:
        super().__init__()
        self.detector = None
        self.cfg = None

    def load(self) -> None:
        from config import LayoutConfig
        from detector import LayoutDetector

        self.cfg = LayoutConfig()
        self.detector = LayoutDetector(self.cfg)
        self._loaded = True

    def predict(self, input_data, **kwargs):
        if not self._loaded:
            self.load()
        return self.detector.process(input_data, page_idx=kwargs.get("page_idx", 0))

    def visualize(self, input_data, output, **kwargs):
        from visualize import draw_overlay
        return draw_overlay(input_data, output, self.cfg)

    def setup(self, **kwargs) -> None:
        from config import LayoutConfig, load_config
        from detector import LayoutDetector

        config_path = kwargs.get("config")
        self.cfg = load_config(config_path) if config_path else LayoutConfig()
        if kwargs.get("model"):
            self.cfg.model = kwargs["model"]
        if kwargs.get("conf"):
            self.cfg.conf_threshold = kwargs["conf"]
        self.detector = LayoutDetector(self.cfg)
        self._loaded = True

    def train(self, **kwargs) -> None:
        from train import main as train_main
        train_main()

    def evaluate(self, **kwargs) -> None:
        from evaluate import main as eval_main
        eval_main()
