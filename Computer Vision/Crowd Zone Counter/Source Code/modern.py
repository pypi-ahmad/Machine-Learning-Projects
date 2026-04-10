"""Modern registry entry for Crowd Zone Counter.

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


@register("crowd_zone_counter")
class CrowdZoneCounter(CVProject):
    """Zone-based people counting with configurable overcrowding alerts."""

    project_type = "detection"
    description = "Zone-based crowd counting and overcrowding alerts"
    modern_tech = "Ultralytics YOLO26m"

    def __init__(self) -> None:
        super().__init__()
        self._detector = None
        self._counter = None
        self.cfg = None

    def load(self) -> None:
        from config import CrowdConfig
        from detector import PersonDetector
        from zone_counter import ZoneCounter

        self.cfg = CrowdConfig()
        self._detector = PersonDetector(self.cfg)
        self._counter = ZoneCounter(self.cfg)
        self._loaded = True

    def predict(self, input_data, **kwargs):
        if not self._loaded:
            self.load()
        frame_idx = kwargs.get("frame_idx", 0)
        dets = self._detector.detect(input_data, frame_idx=frame_idx)
        result = self._counter.update(dets)
        return {"detections": dets, "result": result}

    def visualize(self, input_data, output, **kwargs):
        from visualize import draw_overlay
        return draw_overlay(input_data, output["detections"], output["result"], self.cfg)

    def setup(self, **kwargs) -> None:
        from config import CrowdConfig, load_config
        from detector import PersonDetector
        from zone_counter import ZoneCounter

        config_path = kwargs.get("config")
        self.cfg = load_config(config_path) if config_path else CrowdConfig()
        if kwargs.get("model"):
            self.cfg.model = kwargs["model"]
        if kwargs.get("conf"):
            self.cfg.conf_threshold = kwargs["conf"]
        self._detector = PersonDetector(self.cfg)
        self._counter = ZoneCounter(self.cfg)
        self._loaded = True

    def train(self, **kwargs) -> None:
        from train import main as train_main
        train_main()

    def evaluate(self, **kwargs) -> None:
        from evaluate import main as eval_main
        eval_main()
