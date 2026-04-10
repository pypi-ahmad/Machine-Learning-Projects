"""Plant Disease Classifier — wraps a torchvision classification model.

Loads a trained model and classifies leaf images into one of 38 PlantVillage
classes.  Each prediction is enriched with plant name, disease, and
severity bucket from the config mapping.
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

from config import (
    DISEASE_SEVERITY_MAP,
    SEVERITY_NAMES,
    SeverityConfig,
    estimate_lesion_ratio,
    parse_class,
)


# ── Result dataclass ──────────────────────────────────────

@dataclass
class PredictionResult:
    """A single image prediction with full metadata."""

    class_name: str                     # raw ImageFolder label
    plant: str                          # e.g. "Tomato"
    disease: str                        # e.g. "Late blight"
    severity_index: int                 # 0-3
    severity_name: str                  # none / mild / moderate / severe
    confidence: float
    all_probabilities: dict[str, float]
    lesion_ratio: float | None = None   # only when lesion proxy is enabled


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

    # Replace final classifier head
    if name.startswith("resnet"):
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif name.startswith("efficientnet"):
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features, num_classes
        )
    elif name == "mobilenet_v2":
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features, num_classes
        )
    return model


# ── Classifier ────────────────────────────────────────────

class PlantDiseaseClassifier:
    """High-level classifier: loads weights once, runs inference many times."""

    def __init__(self, cfg: SeverityConfig | None = None) -> None:
        self.cfg = cfg or SeverityConfig()
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
        """Load model weights from a checkpoint file."""
        path = Path(weights or self.cfg.weights_path)
        if not path.exists():
            raise FileNotFoundError(f"Weights not found: {path}")

        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        # Support both bare state_dict and full checkpoint dict
        if isinstance(ckpt, dict) and "class_names" in ckpt:
            self.class_names = ckpt["class_names"]
            state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        else:
            state_dict = ckpt
            self.class_names = sorted(DISEASE_SEVERITY_MAP.keys())

        num_classes = self.cfg.num_classes if not self.class_names else len(self.class_names)
        self.model = _build_model(self.cfg.model_name, num_classes)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device).eval()

    def close(self) -> None:
        self.model = None
        self.class_names = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Prediction ────────────────────────────────────────

    def classify(self, image_bgr: np.ndarray) -> PredictionResult:
        """Classify a single BGR image."""
        if self.model is None:
            raise RuntimeError("Call load() before classify()")

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        idx = int(np.argmax(probs))
        class_name = self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}"
        plant, disease, sev_idx, sev_name = parse_class(class_name)

        all_probs = {
            self.class_names[i]: float(probs[i])
            for i in range(len(self.class_names))
        }

        lesion = None
        if self.cfg.enable_lesion_proxy and sev_idx > 0:
            lesion = estimate_lesion_ratio(
                image_bgr, hue_range=self.cfg.lesion_hue_range
            )

        return PredictionResult(
            class_name=class_name,
            plant=plant,
            disease=disease,
            severity_index=sev_idx,
            severity_name=sev_name,
            confidence=float(probs[idx]),
            all_probabilities=all_probs,
            lesion_ratio=lesion,
        )

    def classify_batch(
        self, images: Sequence[np.ndarray]
    ) -> list[PredictionResult]:
        """Classify multiple BGR images."""
        return [self.classify(img) for img in images]
