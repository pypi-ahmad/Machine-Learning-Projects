"""Modern registry entry for Sports Ball Possession Tracker.

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


@register("sports_ball_possession_tracker")
class SportsBallPossessionTracker(CVProject):
    """Player + ball tracking with nearest-player possession estimation."""

    project_type = "tracking"
    description = "Sports ball possession tracking via YOLO detection + ByteTrack"
    modern_tech = "Ultralytics YOLO26m + ByteTrack"

    def __init__(self) -> None:
        super().__init__()
        self._tracker = None
        self._estimator = None
        self._visualizer = None
        self.cfg = None

    def load(self) -> None:
        from config import PossessionConfig
        from possession import PossessionEstimator
        from tracker import SportTracker
        from visualize import Visualizer

        self.cfg = PossessionConfig()
        self._tracker = SportTracker(self.cfg)
        self._estimator = PossessionEstimator(self.cfg)
        self._visualizer = Visualizer(self.cfg)
        self._loaded = True

    def predict(self, input_data, **kwargs):
        if not self._loaded:
            self.load()
        frame_idx = kwargs.get("frame_idx", 0)
        dets = self._tracker.update(input_data, frame_idx=frame_idx)
        state = self._estimator.update(dets)
        return {"detections": dets, "possession": state}

    def visualize(self, input_data, output, **kwargs):
        return self._visualizer.draw(input_data, output["detections"], output["possession"])

    def setup(self, **kwargs) -> None:
        from config import PossessionConfig, load_config
        from possession import PossessionEstimator
        from tracker import SportTracker
        from visualize import Visualizer

        config_path = kwargs.get("config")
        self.cfg = load_config(config_path) if config_path else PossessionConfig()
        if kwargs.get("model"):
            self.cfg.model = kwargs["model"]
        if kwargs.get("conf"):
            self.cfg.conf_threshold = kwargs["conf"]
        self._tracker = SportTracker(self.cfg)
        self._estimator = PossessionEstimator(self.cfg)
        self._visualizer = Visualizer(self.cfg)
        self._loaded = True

    def train(self, **kwargs) -> None:
        from train import main as train_main
        train_main()

    def evaluate(self, **kwargs) -> None:
        from evaluate import main as eval_main
        eval_main()
