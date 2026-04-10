"""Modern v2 pipeline — Brain Tumour Detection.

Replaces: Notebook-only CNN classifier + data augmentation
Uses:     YOLO26m-cls for image-level tumor/no-tumor classification

Pipeline: YOLO26m-cls → whole-image binary classification (tumor / no-tumor)
          If localization is needed, consider moving to segmentation project.

Fine-tune: python train_classification.py --data brain_tumour/ --weights yolo26m-cls.pt

The original notebook implementation is preserved in the .ipynb files.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2

from core.base import CVProject
from core.registry import register
from utils.yolo import load_yolo
from models.registry import resolve


@register("brain_tumour_detection")
class BrainTumourModern(CVProject):
    project_type = "classification"
    description = "Brain tumour image-level classification (tumor / no-tumor)"
    legacy_tech = "Custom CNN + data augmentation (notebook)"
    modern_tech = "YOLO26m-cls (whole-image binary classifier)"

    def load(self):
        weights, ver, fallback = resolve("brain_tumour_detection", "cls")
        self.model = load_yolo(weights)
        print(f"  [brain_tumour] YOLO-cls loaded ({weights})")
        if fallback:
            print("  [brain_tumour] Fine-tune: python train_classification.py --data brain_tumour/")

    def predict(self, input_data):
        return self.model(input_data, verbose=False)

    def visualize(self, input_data, output):
        return output[0].plot()
