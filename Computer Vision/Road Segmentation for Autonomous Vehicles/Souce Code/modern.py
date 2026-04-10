"""Modern v2 pipeline — Road Segmentation for Autonomous Vehicles.

Replaces: Notebook-only road segmentation
Uses:     YOLO26m-seg

Note: For video streams, add temporal smoothing to reduce
      frame-to-frame mask flickering.

The original notebook implementation is preserved in the .ipynb file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.base import CVProject
from core.registry import register
from utils.yolo import load_yolo
from models.registry import resolve


@register("road_segmentation")
class RoadSegmentationModern(CVProject):
    project_type = "segmentation"
    description = "Road segmentation for autonomous vehicles (add temporal smoothing for video)"
    legacy_tech = "Custom segmentation (notebook)"
    modern_tech = "YOLO26m-seg"

    def load(self):
        weights, ver, fallback = resolve("road_segmentation", "seg")
        print(f"Using model for road_segmentation: version={ver} weights={weights} pretrained_fallback={fallback}")
        self.model = load_yolo(weights)

    def predict(self, input_data):
        return self.model(input_data, verbose=False)

    def visualize(self, input_data, output):
        return output[0].plot()
