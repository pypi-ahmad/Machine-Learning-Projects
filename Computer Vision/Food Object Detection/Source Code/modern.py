"""Modern v2 pipeline — Food Object Detection.

Replaces: InceptionV3 Keras model (Streamlit app)
Uses:     YOLO26m detect filtered to food-related COCO classes

COCO food classes: banana(46), apple(47), sandwich(48), orange(49),
broccoli(50), carrot(51), hot dog(52), pizza(53), donut(54), cake(55)
Dining items: bottle(39), wine glass(40), cup(41), fork(42),
knife(43), spoon(44), bowl(45)

Note: For fine-grained food recognition (200+ categories), train on
Food-101 or UECFood-256.  COCO covers only 10 food item classes.

The original Streamlit implementation is preserved in food_app.py / run.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.base import CVProject
from core.registry import register
from utils.yolo import load_yolo
from models.registry import resolve

# COCO food-item classes (46-55) + dining context (39-45)
_FOOD_CLASSES = list(range(39, 56))


@register("food_object_detection")
class FoodObjectModern(CVProject):
    project_type = "detection"
    description = "Food object detection — COCO food class filter"
    legacy_tech = "InceptionV3 (Keras) + Streamlit"
    modern_tech = "YOLO26m detect (COCO food classes; train on Food-101 for 200+ categories)"

    CONF_THRESHOLD = 0.3

    def load(self):
        weights, ver, fallback = resolve("food_object_detection", "detect")
        if fallback:
            print("  [food] Using pretrained COCO — filtered to food classes (46-55)")
            print("  [food] For 200+ food categories, train on Food-101 or UECFood-256")
        self.model = load_yolo(weights)

    def predict(self, input_data):
        return self.model(input_data, classes=_FOOD_CLASSES, verbose=False, conf=self.CONF_THRESHOLD)

    def visualize(self, input_data, output):
        return output[0].plot()
