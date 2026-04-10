"""Modern v2 pipeline — Skin Cancer Detection.

Replaces: Custom CNN notebook classifier
Uses:     YOLO26m-cls for image-level lesion triage classification

DISCLAIMER: This is an educational demo only. It is NOT a diagnostic tool.
            Do NOT use for clinical decisions. Dermatological diagnosis requires
            expert evaluation and certified medical devices.
            Intended for learning purposes — not approved for any medical use.

Pipeline: YOLO26m-cls → whole-image classification (benign / malignant triage)

Fine-tune: python train_classification.py --data skin_lesion/ --weights yolo26m-cls.pt

The original notebook implementation is preserved in the .ipynb file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2

from core.base import CVProject
from core.registry import register
from utils.yolo import load_yolo
from models.registry import resolve


@register("skin_cancer_detection")
class SkinCancerDetectionModern(CVProject):
    project_type = "classification"
    description = "Skin lesion triage classification — EDUCATIONAL DEMO ONLY, not for diagnosis"
    legacy_tech = "Custom CNN (notebook)"
    modern_tech = "YOLO26m-cls (whole-image triage classifier)"

    def load(self):
        weights, ver, fallback = resolve("skin_cancer_detection", "cls")
        self.model = load_yolo(weights)
        print(f"  [skin_cancer] YOLO-cls loaded ({weights})")
        print("  [skin_cancer] DISCLAIMER: Educational demo only — NOT for clinical use")
        if fallback:
            print("  [skin_cancer] Fine-tune: python train_classification.py --data skin_lesion/")

    def predict(self, input_data):
        return self.model(input_data, verbose=False)

    def visualize(self, input_data, output):
        vis = output[0].plot()
        cv2.putText(vis, "DEMO ONLY - Not for clinical diagnosis", (10, vis.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return vis
