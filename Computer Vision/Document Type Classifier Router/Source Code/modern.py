"""Document Type Classifier Router — CVProject registry entry."""

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


@register("document_type_classifier_router")
class DocumentTypeClassifierRouter(CVProject):
    """Classify document images and route them to downstream pipelines."""

    project_type = "classification"
    description = (
        "Classify scanned/photographed documents into 16 types "
        "(invoice, letter, form, resume, …) and route each to the "
        "appropriate downstream processing pipeline"
    )
    legacy_tech = "Manual document sorting and filing"
    modern_tech = "ResNet-18 transfer learning + configurable routing table"

    def __init__(self) -> None:
        super().__init__()
        self._ctrl = None

    def load(self) -> None:
        from controller import DocumentController

        self._ctrl = DocumentController()
        self._ctrl.load()

    def predict(self, input_data):
        if self._ctrl is None:
            self.load()

        if isinstance(input_data, str):
            cr, rd = self._ctrl.process_file(input_data)
        else:
            cr, rd = self._ctrl.classify_and_route(input_data)

        return {
            "document_type": cr.class_name,
            "display_label": cr.display_label,
            "confidence": cr.confidence,
            "pipeline": rd.pipeline,
            "routed": rd.routed,
            "reason": rd.reason,
        }
