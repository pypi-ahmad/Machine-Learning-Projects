"""Ecommerce Item Attribute Tagger — CVProject registry entry.
"""Ecommerce Item Attribute Tagger — CVProject registry entry.

Thin adapter that plugs into the repo's global registry so the
project is discoverable via ``core.registry.PROJECT_REGISTRY``.
"""
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
_REPO = _SRC.parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register

from config import TaggerConfig, load_config


@register("ecommerce_item_attribute_tagger")
class EcommerceItemAttributeTagger(CVProject):
    """Predict structured attributes from product images."""

    project_type = "classification"
    description = (
        "Multi-head attribute tagger -- predicts category, colour, "
        "article type, season, usage, and gender from product images"
    )
    legacy_tech = "Manual data entry for product catalogs"
    modern_tech = "ResNet multi-head classifier with optional YOLO item isolation"

    def __init__(self, config: TaggerConfig | None = None) -> None:
        super().__init__()
        self._cfg = config or TaggerConfig()
        self._detector = None
        self._predictor = None

    # ── CVProject interface ────────────────────────────────

    def load(self) -> None:
        from detector import ItemDetector
        from attribute_predictor import AttributePredictor

        self._detector = ItemDetector(self._cfg)
        self._detector.load()

        self._predictor = AttributePredictor(self._cfg)
        self._predictor.load()

    def predict(self, input_data) -> dict:
        if self._predictor is None:
            self.load()

        frame = (
            input_data
            if isinstance(input_data, np.ndarray)
            else cv2.imread(str(input_data))
        )
        if frame is None:
            return {"error": "Could not read image"}

        crop, box = self._detector.isolate(frame)
        prediction = self._predictor.predict_proba(crop)

        return {
            "attributes": {
                attr: {"value": info["label"], "confidence": info["confidence"]}
                for attr, info in prediction.items()
            },
            "box": box,
        }

    def visualize(self, input_data, output=None) -> np.ndarray:
        if self._predictor is None:
            self.load()

        from visualize import draw_attributes

        frame = (
            input_data
            if isinstance(input_data, np.ndarray)
            else cv2.imread(str(input_data))
        )

        if output is None:
            output = self.predict(frame)

        # Reconstruct predict_proba format for visualize
        prediction = {}
        for attr, info in output.get("attributes", {}).items():
            prediction[attr] = {
                "label": info.get("value", ""),
                "confidence": info.get("confidence", 0.0),
            }

        return draw_attributes(frame, prediction, self._cfg)

    def setup(self, **kwargs) -> None:
        if kwargs:
            self._cfg = TaggerConfig.from_dict(kwargs)
        self.load()

    def train(self, **kwargs) -> None:
        from train import main as train_main
        train_main()

    def evaluate(self, **kwargs) -> None:
        from train import train_model
        data = Path(kwargs.get("data", "."))
        train_model(data, self._cfg)
