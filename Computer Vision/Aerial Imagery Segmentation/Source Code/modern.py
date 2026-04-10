"""Modern v2 pipeline — Aerial Imagery Segmentation.

Replaces: Notebook-only U-Net / custom segmentation
Uses:     YOLO26m-seg for instance segmentation

Note: If annotations are rotated bounding boxes (not masks),
      consider switching to YOLO26m-obb instead.

The original notebook implementation is preserved in the .ipynb file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.base import CVProject
from core.registry import register
from utils.yolo import load_yolo
from models.registry import resolve


@register("aerial_imagery_segmentation")
class AerialSegmentationModern(CVProject):
    project_type = "segmentation"
    description = "Aerial image segmentation (instance masks; use OBB for rotated boxes)"
    legacy_tech = "Custom U-Net (notebook)"
    modern_tech = "YOLO26m-seg (or YOLO26m-obb for rotated annotations)"

    def load(self):
        weights, ver, fallback = resolve("aerial_imagery_segmentation", "seg")
        print(f"Using model for aerial_imagery_segmentation: version={ver} weights={weights} pretrained_fallback={fallback}")
        self.model = load_yolo(weights)

    def predict(self, input_data):
        return self.model(input_data, verbose=False)

    def visualize(self, input_data, output):
        return output[0].plot()
