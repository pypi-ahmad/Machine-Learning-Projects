"""Modern v2 pipeline — Building Footprint Segmentation.

Replaces: Notebook-only segmentation model
Uses:     Ultralytics YOLO26-seg

The original notebook implementation is preserved in the .ipynb file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.base import CVProject
from core.registry import register
from utils.yolo import load_yolo
from models.registry import resolve


@register("building_footprint_segmentation")
class BuildingFootprintModern(CVProject):
    project_type = "segmentation"
    description = "Building footprint segmentation (instance masks)"
    legacy_tech = "Custom segmentation (notebook)"
    modern_tech = "YOLO26m-seg"

    def load(self):
        weights, ver, fallback = resolve("building_footprint_segmentation", "seg")
        print(f"Using model for building_footprint_segmentation: version={ver} weights={weights} pretrained_fallback={fallback}")
        self.model = load_yolo(weights)

    def predict(self, input_data):
        return self.model(input_data, verbose=False)

    def visualize(self, input_data, output):
        return output[0].plot()
