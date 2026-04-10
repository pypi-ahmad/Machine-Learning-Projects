"""Modern v2 pipeline — Plant Disease Prediction.

Replaces: Custom CNN notebook classifier
Uses:     YOLO26m-cls for leaf disease classification

Optional: leaf crop/segmentation preprocessing if backgrounds are messy.
The original notebook implementation is preserved in the .ipynb file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.base import CVProject
from core.registry import register
from utils.yolo import load_yolo
from models.registry import resolve


@register("plant_disease_prediction")
class PlantDiseaseModern(CVProject):
    project_type = "classification"
    description = "Plant disease classification (leaf image-level)"
    legacy_tech = "Custom CNN (notebook)"
    modern_tech = "YOLO26m-cls"

    def load(self):
        weights, ver, fallback = resolve("plant_disease_prediction", "cls")
        print(f"Using model for plant_disease_prediction: version={ver} weights={weights} pretrained_fallback={fallback}")
        self.model = load_yolo(weights)

    def predict(self, input_data):
        return self.model(input_data, verbose=False)

    def visualize(self, input_data, output):
        return output[0].plot()
