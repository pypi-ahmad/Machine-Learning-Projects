"""Wildlife Species Retrieval — optional classifier for reranking.

A separate classification model that can be used to boost retrieval
hits whose predicted species matches the query's predicted species.
This component is fully independent of the retrieval pipeline —
it is only wired in at the controller level when ``enable_rerank``
is set in the config.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T

from index import SearchHit


@dataclass
class ClassificationResult:
    """Predicted species with confidence."""
    species: str
    confidence: float
    all_probabilities: dict[str, float]


_MODEL_BUILDERS = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "efficientnet_b0": models.efficientnet_b0,
    "mobilenet_v2": models.mobilenet_v2,
}


def _build_model(name: str, num_classes: int) -> torch.nn.Module:
    if name not in _MODEL_BUILDERS:
        raise ValueError(f"Unknown model: {name}")
    model = _MODEL_BUILDERS[name](weights=None)
    if name.startswith("resnet"):
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif name.startswith("efficientnet"):
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features, num_classes)
    elif name == "mobilenet_v2":
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features, num_classes)
    return model


class WildlifeClassifier:
    """Species classifier for optional reranking of retrieval results."""

    def __init__(
        self,
        weights_path: str = "runs/wildlife_cls/best_model.pt",
        model_name: str = "resnet18",
        num_classes: int = 90,
        imgsz: int = 224,
        device: str | None = None,
    ) -> None:
        self.weights_path = weights_path
        self.model_name = model_name
        self.num_classes = num_classes
        self.imgsz = imgsz
        self.device = torch.device(
            device if device else
            ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model: torch.nn.Module | None = None
        self.class_names: list[str] = []
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((imgsz, imgsz)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    # ── lifecycle ─────────────────────────────────────────

    def load(self, weights: str | Path | None = None) -> None:
        path = Path(weights or self.weights_path)
        if not path.exists():
            raise FileNotFoundError(f"Classifier weights not found: {path}")

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and "class_names" in ckpt:
            self.class_names = ckpt["class_names"]
            state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        else:
            state_dict = ckpt
            self.class_names = [f"class_{i}" for i in range(self.num_classes)]

        n = len(self.class_names) if self.class_names else self.num_classes
        self.model = _build_model(self.model_name, n)
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

    # ── prediction ────────────────────────────────────────

    def classify(self, image_bgr: np.ndarray) -> ClassificationResult:
        if not self.is_loaded:
            raise RuntimeError("Call load() before classify()")

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        idx = int(np.argmax(probs))
        species = self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}"
        all_probs = {
            self.class_names[i]: float(probs[i])
            for i in range(len(self.class_names))
        }
        return ClassificationResult(
            species=species,
            confidence=float(probs[idx]),
            all_probabilities=all_probs,
        )

    # ── reranking ─────────────────────────────────────────

    def rerank(
        self,
        query_species: str,
        hits: list[SearchHit],
        weight: float = 0.4,
    ) -> list[SearchHit]:
        """Rerank retrieval hits by boosting same-species matches.

        New score = (1 - weight) * similarity + weight * species_match
        where species_match is 1.0 if hit species == query species, else 0.0.
        """
        scored = []
        for h in hits:
            match_bonus = 1.0 if h.species.lower() == query_species.lower() else 0.0
            new_score = (1.0 - weight) * h.score + weight * match_bonus
            scored.append((new_score, h))

        scored.sort(key=lambda x: x[0], reverse=True)
        reranked = []
        for rank, (score, h) in enumerate(scored, 1):
            reranked.append(SearchHit(
                path=h.path,
                species=h.species,
                score=round(score, 6),
                rank=rank,
            ))
        return reranked
