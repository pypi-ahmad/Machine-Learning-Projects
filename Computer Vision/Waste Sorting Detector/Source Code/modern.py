"""Modern registry entry for Waste Sorting Detector.

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


@register("waste_sorting_detector")
class WasteSortingDetector(CVProject):
    """Waste detection with per-class counting and bin-zone validation."""

    project_type = "detection"

    def __init__(self) -> None:
        super().__init__()
        self.sorter = None
        self.cfg = None

    def setup(self, **kwargs) -> None:
        from config import WasteConfig, load_config

        config_path = kwargs.get("config")
        self.cfg = load_config(config_path) if config_path else WasteConfig()

        if kwargs.get("model"):
            self.cfg.model = kwargs["model"]
        if kwargs.get("conf"):
            self.cfg.conf_threshold = kwargs["conf"]

        from sorter import WasteSorter
        self.sorter = WasteSorter(self.cfg)

    def process_frame(self, frame, **kwargs):
        from visualize import draw_overlay

        frame_idx = kwargs.get("frame_idx", 0)
        result = self.sorter.process(frame, frame_idx=frame_idx)
        vis = draw_overlay(frame, result, self.cfg)
        return vis

    def train(self, **kwargs) -> None:
        from train import main as train_main
        train_main()

    def evaluate(self, **kwargs) -> None:
        from evaluate import main as eval_main
        eval_main()
