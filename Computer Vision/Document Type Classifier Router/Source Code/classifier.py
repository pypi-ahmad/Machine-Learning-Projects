"""Document Type Classifier Router — classification model.

Wraps a torchvision classification model to predict document type
from an image.  This module is independent of the routing logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T

from config import CLASS_NAMES, DISPLAY_LABELS, RouterConfig


# ── Result dataclass ──────────────────────────────────────

@dataclass
class ClassificationResult:
    """Single document classification output."""

    class_name: str          # raw class label (e.g. "invoice")
    display_label: str       # human-friendly label
    confidence: float
    all_probabilities: dict[str, float]


# ── Model builder ─────────────────────────────────────────

_BUILDERS = {
    "resnet18":        models.resnet18,
    "resnet34":        models.resnet34,
    "resnet50":        models.resnet50,
    "efficientnet_b0": models.efficientnet_b0,
    "mobilenet_v2":    models.mobilenet_v2,
}


def _build_model(name: str, num_classes: int) -> torch.nn.Module:
    if name not in _BUILDERS:
        raise ValueError(f"Unknown model: {name}. Choose from {list(_BUILDERS)}")
    model = _BUILDERS[name](weights=None)
    if name.startswith("resnet"):
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif name.startswith("efficientnet"):
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features, num_classes)
    elif name == "mobilenet_v2":
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features, num_classes)
    return model


# ── Classifier ────────────────────────────────────────────

class DocumentClassifier:
    """Classify document images into one of 16 types."""

    def __init__(self, cfg: RouterConfig | None = None) -> None:
        self.cfg = cfg or RouterConfig()
        self.device = torch.device(
            self.cfg.device
            if self.cfg.device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model: torch.nn.Module | None = None
        self.class_names: list[str] = []
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((self.cfg.imgsz, self.cfg.imgsz)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    # ── Lifecycle ─────────────────────────────────────────

    def load(self, weights: str | Path | None = None) -> None:
        path = Path(weights or self.cfg.weights_path)
        if not path.exists():
            raise FileNotFoundError(f"Weights not found: {path}")

        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        if isinstance(ckpt, dict) and "class_names" in ckpt:
            self.class_names = ckpt["class_names"]
            state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        else:
            state_dict = ckpt
            self.class_names = list(CLASS_NAMES)

        num_classes = len(self.class_names) if self.class_names else self.cfg.num_classes
        self.model = _build_model(self.cfg.model_name, num_classes)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device).eval()

    def close(self) -> None:
        self.model = None
        self.class_names = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    # ── Prediction ────────────────────────────────────────

    def classify(self, image_bgr: np.ndarray) -> ClassificationResult:
        """Classify a single BGR image."""
        if not self.is_loaded:
            raise RuntimeError("Call load() before classify()")

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        idx = int(np.argmax(probs))
        class_name = self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}"
        display = DISPLAY_LABELS.get(class_name, class_name)
        all_probs = {
            self.class_names[i]: float(probs[i])
            for i in range(len(self.class_names))
        }

        return ClassificationResult(
            class_name=class_name,
            display_label=display,
            confidence=float(probs[idx]),
            all_probabilities=all_probs,
        )

    def classify_batch(
        self, images: Sequence[np.ndarray]
    ) -> list[ClassificationResult]:
        """Classify multiple BGR images."""
        return [self.classify(img) for img in images]
