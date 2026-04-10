"""Ecommerce Item Attribute Tagger — multi-head attribute predictor.

A shared backbone (ResNet / MobileNet / EfficientNet) feeds into
independent classification heads for each attribute (category, colour,
season, etc.). Fully separated from detection/isolation.

Usage::

    from attribute_predictor import AttributePredictor

    predictor = AttributePredictor(cfg, label_maps)
    predictor.load("runs/attribute_tagger/best_model.pt")
    attrs = predictor.predict(image)          # → dict[str, str]
    attrs = predictor.predict_proba(image)    # → dict[str, dict]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

from config import ATTRIBUTE_HEADS, TaggerConfig

log = logging.getLogger("attribute_tagger.predictor")

# Backbone registry
_BACKBONES = {
    "resnet18":     (models.resnet18,     models.ResNet18_Weights.DEFAULT,     512),
    "resnet50":     (models.resnet50,     models.ResNet50_Weights.DEFAULT,     2048),
    "mobilenet_v2": (models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT, 1280),
}


# ---------------------------------------------------------------------------
# Multi-head model
# ---------------------------------------------------------------------------

class MultiHeadAttributeModel(nn.Module):
    """Shared backbone → per-attribute classification heads."""

    def __init__(
        self,
        backbone_name: str,
        head_sizes: dict[str, int],
    ) -> None:
        super().__init__()
        if backbone_name not in _BACKBONES:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        constructor, weights, feat_dim = _BACKBONES[backbone_name]
        backbone = constructor(weights=weights)

        # Remove classification head
        if hasattr(backbone, "fc"):
            backbone.fc = nn.Identity()
        elif hasattr(backbone, "classifier"):
            backbone.classifier = nn.Identity()

        self.backbone = backbone
        self.feat_dim = feat_dim

        # Per-attribute heads
        self.heads = nn.ModuleDict()
        for attr_name, num_cls in head_sizes.items():
            self.heads[attr_name] = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(feat_dim, num_cls),
            )

    def forward(
        self, x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        feats = self.backbone(x)
        if feats.ndim > 2:
            feats = feats.mean(dim=[2, 3])  # global avg pool for MobileNet
        return {name: head(feats) for name, head in self.heads.items()}


# ---------------------------------------------------------------------------
# Predictor wrapper
# ---------------------------------------------------------------------------

class AttributePredictor:
    """High-level predictor wrapping the multi-head model."""

    def __init__(
        self,
        cfg: TaggerConfig,
        label_maps: dict[str, list[str]] | None = None,
    ) -> None:
        self.cfg = cfg
        self.label_maps: dict[str, list[str]] = label_maps or {}
        self._model: MultiHeadAttributeModel | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize((cfg.imgsz, cfg.imgsz)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def load(self, weights_path: str | Path | None = None) -> None:
        """Load model weights and label maps."""
        wpath = Path(weights_path or self.cfg.weights_path)
        if not wpath.is_absolute():
            wpath = Path(__file__).resolve().parent / wpath

        if not wpath.exists():
            raise FileNotFoundError(f"Weights not found: {wpath}")

        checkpoint = torch.load(str(wpath), map_location=self._device, weights_only=False)

        self.label_maps = checkpoint.get("label_maps", self.label_maps)
        head_sizes = {k: len(v) for k, v in self.label_maps.items()}

        self._model = MultiHeadAttributeModel(
            self.cfg.backbone, head_sizes,
        )
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.to(self._device)
        self._model.eval()
        log.info("Loaded attribute model from %s (%d heads)", wpath, len(head_sizes))

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self._transform(image).unsqueeze(0).to(self._device)

    def predict(self, image: np.ndarray) -> dict[str, str]:
        """Predict top-1 attribute labels for an image.

        Returns
        -------
        dict[str, str]
            Attribute name → predicted label.
        """
        proba = self.predict_proba(image)
        return {
            attr: info["label"]
            for attr, info in proba.items()
        }

    def predict_proba(self, image: np.ndarray) -> dict[str, dict]:
        """Predict attributes with confidence scores.

        Returns
        -------
        dict[str, dict]
            Each key is an attribute name, each value has:
            ``label``, ``confidence``, ``top3``.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        tensor = self._preprocess(image)
        with torch.no_grad():
            logits = self._model(tensor)

        results: dict[str, dict] = {}
        for attr_name, attr_logits in logits.items():
            probs = torch.softmax(attr_logits, dim=1)[0]
            labels = self.label_maps.get(attr_name, [])

            top_k = min(3, len(labels))
            vals, idxs = probs.topk(top_k)

            top3 = [(labels[i] if i < len(labels) else f"cls_{i}",
                      round(float(v), 4))
                     for v, i in zip(vals, idxs)]

            best_idx = int(idxs[0])
            best_conf = float(vals[0])
            best_label = labels[best_idx] if best_idx < len(labels) else f"cls_{best_idx}"

            results[attr_name] = {
                "label": best_label,
                "confidence": round(best_conf, 4),
                "top3": top3,
            }

        return results

    def predict_batch(
        self, images: list[np.ndarray],
    ) -> list[dict[str, dict]]:
        """Predict attributes for a batch of images."""
        return [self.predict_proba(img) for img in images]

    def to_structured_json(self, proba_result: dict[str, dict]) -> dict:
        """Convert a prediction to catalog-ready structured JSON."""
        structured: dict[str, Any] = {}
        for attr_name, info in proba_result.items():
            structured[attr_name] = {
                "value": info["label"],
                "confidence": info["confidence"],
            }
            if info["confidence"] < self.cfg.confidence_threshold:
                structured[attr_name]["uncertain"] = True
        return structured
