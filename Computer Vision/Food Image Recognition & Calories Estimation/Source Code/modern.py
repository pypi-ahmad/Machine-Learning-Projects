"""Modern v2 pipeline — Food Image Recognition & Calorie Estimation.

Replaces: InceptionV3 / EfficientNet Flask app
Uses:     YOLO26m-cls for image-level food classification

The original implementation is preserved in app.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.base import CVProject
from core.registry import register
from utils.yolo import load_yolo
from models.registry import resolve


@register("food_image_recognition")
class FoodImageRecognitionModern(CVProject):
    project_type = "classification"
    description = "Food image recognition + calorie estimation"
    legacy_tech = "InceptionV3 / EfficientNet (Flask)"
    modern_tech = "YOLO26m-cls"

    def load(self):
        weights, ver, fallback = resolve("food_image_recognition", "cls")
        print(f"Using model for food_image_recognition: version={ver} weights={weights} pretrained_fallback={fallback}")
        self.model = load_yolo(weights)

    def predict(self, input_data):
        return self.model(input_data, verbose=False)

    def visualize(self, input_data, output):
        return output[0].plot()
