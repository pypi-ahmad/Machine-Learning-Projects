"""Modern registry entry for Drone Ship OBB Detector.

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


@register("drone_ship_obb_detector")
class DroneShipOBBDetector(CVProject):
    """Oriented bounding-box detection for ships and vehicles in aerial imagery."""

    project_type = "detection"
    description = "OBB detection for aerial/satellite images using YOLO-OBB"
    modern_tech = "Ultralytics YOLO26m-OBB"

    def __init__(self) -> None:
        super().__init__()
        self.detector = None
        self.cfg = None

    def load(self) -> None:
        from config import OBBConfig, load_config
        self.cfg = OBBConfig()
        from detector import OBBDetector
        self.detector = OBBDetector(self.cfg)
        self._loaded = True

    def predict(self, input_data, **kwargs):
        if not self._loaded:
            self.load()
        result = self.detector.process(input_data, frame_idx=kwargs.get("frame_idx", 0))
        return result

    def visualize(self, input_data, output, **kwargs):
        from visualize import draw_overlay
        return draw_overlay(input_data, output, self.cfg)

    def setup(self, **kwargs) -> None:
        from config import OBBConfig, load_config
        config_path = kwargs.get("config")
        self.cfg = load_config(config_path) if config_path else OBBConfig()
        if kwargs.get("model"):
            self.cfg.model = kwargs["model"]
        if kwargs.get("conf"):
            self.cfg.conf_threshold = kwargs["conf"]
        from detector import OBBDetector
        self.detector = OBBDetector(self.cfg)
        self._loaded = True

    def train(self, **kwargs) -> None:
        from train import main as train_main
        train_main()

    def evaluate(self, **kwargs) -> None:
        from evaluate import main as eval_main
        eval_main()
