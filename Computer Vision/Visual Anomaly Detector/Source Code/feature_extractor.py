"""Visual Anomaly Detector — feature extraction.

Wraps a pretrained CNN backbone (ResNet variants) as a feature
extractor, removing the final classification head. Supports batch
extraction for efficient training.

Usage::

    from feature_extractor import FeatureExtractor

    fe = FeatureExtractor(backbone="resnet18", imgsz=224)
    fe.load()
    vec = fe.extract(image)                  # single image → (D,)
    vecs = fe.extract_batch(image_list)      # batch → (N, D)
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T

log = logging.getLogger("visual_anomaly.feature_extractor")

# Backbone registry: name → (constructor, weights, feature_dim)
_BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, 512),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048),
    "wide_resnet50_2": (models.wide_resnet50_2, models.Wide_ResNet50_2_Weights.DEFAULT, 2048),
}


class FeatureExtractor:
    """Extract feature vectors from images using a pretrained CNN."""

    def __init__(self, backbone: str = "resnet18", imgsz: int = 224) -> None:
        self.backbone_name = backbone
        self.imgsz = imgsz
        self._model: torch.nn.Module | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize((imgsz, imgsz)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        self.feature_dim: int = 0

    def load(self) -> None:
        """Load the pretrained backbone and remove the classification head."""
        if self.backbone_name not in _BACKBONES:
            raise ValueError(
                f"Unknown backbone '{self.backbone_name}'. "
                f"Choose from: {list(_BACKBONES.keys())}"
            )

        constructor, weights, dim = _BACKBONES[self.backbone_name]
        model = constructor(weights=weights)
        model.fc = torch.nn.Identity()
        model.eval()
        model.to(self._device)
        self._model = model
        self.feature_dim = dim
        log.info("Loaded %s (dim=%d) on %s", self.backbone_name, dim, self._device)

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Convert a BGR numpy image to a normalised tensor."""
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self._transform(image)

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract a feature vector from a single image.

        Returns
        -------
        np.ndarray
            1-D feature vector of shape ``(feature_dim,)``.
        """
        tensor = self._preprocess(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            features = self._model(tensor)
        return features.cpu().numpy().flatten()

    def extract_batch(self, images: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """Extract features from a list of images.

        Returns
        -------
        np.ndarray
            Array of shape ``(N, feature_dim)``.
        """
        all_features: list[np.ndarray] = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i : i + batch_size]
            tensors = torch.stack([self._preprocess(img) for img in batch_imgs])
            tensors = tensors.to(self._device)
            with torch.no_grad():
                feats = self._model(tensors)
            all_features.append(feats.cpu().numpy())
        return np.concatenate(all_features, axis=0)

    def extract_from_paths(
        self, paths: list[str | Path], batch_size: int = 32,
    ) -> np.ndarray:
        """Load images from paths and extract features.

        Returns
        -------
        np.ndarray
            Array of shape ``(N, feature_dim)``.
        """
        images: list[np.ndarray] = []
        valid_paths: list[str] = []
        for p in paths:
            img = cv2.imread(str(p))
            if img is not None:
                images.append(img)
                valid_paths.append(str(p))
            else:
                log.warning("Cannot read image: %s", p)

        if not images:
            raise ValueError("No valid images found")

        log.info("Extracting features from %d images...", len(images))
        return self.extract_batch(images, batch_size=batch_size)
