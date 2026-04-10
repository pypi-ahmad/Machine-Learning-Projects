"""Modern v2 pipeline — Traffic Sign Recognition.

Replaces: Keras CNN + Flask app
Uses:     YOLO26m-cls for cropped sign images (image-level classification)
          YOLO26m detect would be needed for full road-scene sign detection

Conditional: If images are pre-cropped signs, cls is correct.
             If road scenes with signs at various positions, change to detect.

The original implementation is preserved in Traffic_app.py / run.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.base import CVProject
from core.registry import register
from utils.yolo import load_yolo
from models.registry import resolve


@register("traffic_sign_recognition")
class TrafficSignRecognitionModern(CVProject):
    project_type = "classification"
    description = "Traffic sign classification (cropped signs — image-level)"
    legacy_tech = "Keras CNN + Flask"
    modern_tech = "YOLO26m-cls (for road scenes: switch to YOLO26m detect)"

    def load(self):
        weights, ver, fallback = resolve("traffic_sign_recognition", "cls")
        print(f"Using model for traffic_sign_recognition: version={ver} weights={weights} pretrained_fallback={fallback}")
        if fallback:
            print("  [traffic_sign] Using cls mode for cropped signs. For full-scene detection, register detect weights.")
        self.model = load_yolo(weights)

    def predict(self, input_data):
        return self.model(input_data, verbose=False)

    def visualize(self, input_data, output):
        return output[0].plot()
