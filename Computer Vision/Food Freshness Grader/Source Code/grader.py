"""Food Freshness Grader — grading engine.
"""Food Freshness Grader — grading engine.

Loads a trained classification model and produces freshness grades
with confidence scores.
"""
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms

from config import CLASS_NAMES, FreshnessConfig, parse_label

logger = logging.getLogger(__name__)

_MODEL_BUILDERS = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "efficientnet_b0": models.efficientnet_b0,
    "mobilenet_v2": models.mobilenet_v2,
}


@dataclass
class GradeResult:
    """Result for a single image."""

    class_name: str
    freshness: str          # "fresh" | "stale"
    produce: str            # e.g. "apple"
    confidence: float       # 0–1
    class_index: int
    all_probabilities: list[float]


class FreshnessGrader:
    """Classify food images and produce freshness grades."""

    def __init__(self, cfg: FreshnessConfig | None = None) -> None:
        self.cfg = cfg or FreshnessConfig()
        self._model: torch.nn.Module | None = None
        self._device: torch.device | None = None
        self._transform: transforms.Compose | None = None
        self._class_names: list[str] = list(self.cfg.class_names)

    # ── lifecycle ──────────────────────────────────────────

    def load(self, weights_path: str | None = None) -> None:
        """Load model weights."""
        wp = weights_path or self.cfg.weights_path

        if self.cfg.device:
            self._device = torch.device(self.cfg.device)
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build model architecture
        builder = _MODEL_BUILDERS.get(self.cfg.model_name)
        if builder is None:
            raise ValueError(f"Unknown model: {self.cfg.model_name}")

        model = builder(weights=None)
        nc = self.cfg.num_classes

        if self.cfg.model_name.startswith("resnet"):
            in_feat = model.fc.in_features
            model.fc = torch.nn.Linear(in_feat, nc)
        elif self.cfg.model_name == "efficientnet_b0":
            in_feat = model.classifier[1].in_features
            model.classifier[1] = torch.nn.Linear(in_feat, nc)
        elif self.cfg.model_name == "mobilenet_v2":
            in_feat = model.classifier[1].in_features
            model.classifier[1] = torch.nn.Linear(in_feat, nc)

        # Load trained weights
        p = Path(wp)
        if p.exists():
            ckpt = torch.load(str(p), map_location=self._device, weights_only=True)
            # Handle both raw state_dict and wrapped checkpoint
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
                if "classes" in ckpt:
                    self._class_names = list(ckpt["classes"])
            elif isinstance(ckpt, dict) and "state_dict" in ckpt:
                model.load_state_dict(ckpt["state_dict"])
            else:
                model.load_state_dict(ckpt)
            logger.info("Loaded weights from %s", p)
        else:
            logger.warning("No weights found at %s -- using random init", p)

        model.eval()
        model.to(self._device)
        self._model = model

        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.cfg.imgsz + 32),
            transforms.CenterCrop(self.cfg.imgsz),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        logger.info("Grader ready: %s on %s (%d classes)",
                     self.cfg.model_name, self._device, nc)

    def close(self) -> None:
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── grading API ────────────────────────────────────────

    @torch.no_grad()
    def grade(self, image_bgr: np.ndarray) -> GradeResult:
        """Grade a single BGR image."""
        assert self._model is not None, "Call load() first"
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb).unsqueeze(0).to(self._device)

        logits = self._model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        cname = self._class_names[idx] if idx < len(self._class_names) else f"class_{idx}"
        freshness, produce = parse_label(cname)

        return GradeResult(
            class_name=cname,
            freshness=freshness,
            produce=produce,
            confidence=conf,
            class_index=idx,
            all_probabilities=probs.tolist(),
        )

    @torch.no_grad()
    def grade_batch(self, images_bgr: list[np.ndarray]) -> list[GradeResult]:
        """Grade a batch of BGR images."""
        assert self._model is not None, "Call load() first"
        tensors = []
        for img in images_bgr:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensors.append(self._transform(rgb))

        batch = torch.stack(tensors).to(self._device)
        logits = self._model(batch)
        probs_batch = F.softmax(logits, dim=1).cpu().numpy()

        results = []
        for probs in probs_batch:
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            cname = self._class_names[idx] if idx < len(self._class_names) else f"class_{idx}"
            freshness, produce = parse_label(cname)
            results.append(GradeResult(
                class_name=cname,
                freshness=freshness,
                produce=produce,
                confidence=conf,
                class_index=idx,
                all_probabilities=probs.tolist(),
            ))
        return results

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)
