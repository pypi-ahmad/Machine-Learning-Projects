"""Plant Disease Severity Estimator -- CVProject registry entry."""

from __future__ import annotations

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from core.base import CVProject  # noqa: E402
from core.registry import register  # noqa: E402

_src = Path(__file__).resolve().parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


@register("plant_disease_severity_estimator")
class PlantDiseaseSeverityEstimator(CVProject):
    """Classify leaf disease type and estimate severity from images."""

    project_type = "classification"
    description = (
        "Classify leaf images into 38 PlantVillage disease classes "
        "and assign a severity bucket (none / mild / moderate / severe)"
    )
    legacy_tech = "Manual scouting by agronomists"
    modern_tech = "ResNet-18 transfer learning (ImageNet -> PlantVillage)"

    def __init__(self) -> None:
        super().__init__()
        self._ctrl = None

    def load(self) -> None:
        from controller import PlantDiseaseController

        self._ctrl = PlantDiseaseController()
        self._ctrl.load()

    def predict(self, input_data):
        if self._ctrl is None:
            self.load()

        if isinstance(input_data, str):
            result = self._ctrl.predict_file(input_data)
        else:
            result = self._ctrl.predict(input_data)

        return {
            "class_name": result.class_name,
            "plant": result.plant,
            "disease": result.disease,
            "severity_index": result.severity_index,
            "severity_name": result.severity_name,
            "confidence": result.confidence,
            "lesion_ratio": result.lesion_ratio,
        }
