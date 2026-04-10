"""Modern v2 pipeline — Aerial Cactus Identification.

Replaces: Notebook-only CNN classifier
Uses:     Ultralytics YOLO26-cls (ImageNet pretrained)

The original notebook implementation is preserved in the .ipynb file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.base import CVProject
from core.registry import register
from utils.yolo import load_yolo
from models.registry import resolve


@register("aerial_cactus_identification")
class AerialCactusModern(CVProject):
    project_type = "classification"
    description = "Aerial cactus identification (image-level binary classification)"
    legacy_tech = "Custom CNN (notebook)"
    modern_tech = "YOLO26m-cls"

    def load(self):
        weights, ver, fallback = resolve("aerial_cactus_identification", "cls")
        print(f"Using model for aerial_cactus_identification: version={ver} weights={weights} pretrained_fallback={fallback}")
        self.model = load_yolo(weights)

    def predict(self, input_data):
        return self.model(input_data, verbose=False)

    def visualize(self, input_data, output):
        return output[0].plot()
