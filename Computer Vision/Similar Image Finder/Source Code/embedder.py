"""Similar Image Finder — feature embedding extractor.

Uses a pretrained torchvision backbone (classification head removed)
to produce a fixed-length, L2-normalised embedding for any image.
This module is independent of the index and retrieval layers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from config import SimilarityConfig

# backbone → (weights enum, embedding dim)
_BACKBONE_INFO: dict[str, tuple[str, int]] = {
    "efficientnet_b0":    ("EfficientNet_B0_Weights",      1280),
    "efficientnet_b2":    ("EfficientNet_B2_Weights",      1408),
    "resnet50":           ("ResNet50_Weights",              2048),
    "resnet18":           ("ResNet18_Weights",               512),
    "mobilenet_v3_small": ("MobileNet_V3_Small_Weights",    576),
    "mobilenet_v3_large": ("MobileNet_V3_Large_Weights",    960),
}


class ImageEmbedder:
    """Extract embeddings from images using a pretrained CNN backbone."""

    def __init__(self, cfg: SimilarityConfig | None = None) -> None:
        if cfg is None:
            from config import SimilarityConfig
            cfg = SimilarityConfig()
        self.cfg = cfg
        self._model = None
        self._transform = None
        self._device = None

    # ── lifecycle ──────────────────────────────────────────

    def load(self) -> None:
        import torch
        import torchvision.models as models
        import torchvision.transforms as T

        device = self.cfg.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        name = self.cfg.backbone
        if name not in _BACKBONE_INFO:
            raise ValueError(
                f"Unknown backbone: {name}. Choose from: {list(_BACKBONE_INFO)}"
            )
        weights_cls_name, embed_dim = _BACKBONE_INFO[name]
        weights_cls = getattr(models, weights_cls_name)
        weights = weights_cls.DEFAULT

        model = getattr(models, name)(weights=weights)

        # Remove classification head
        if hasattr(model, "classifier"):
            model.classifier = torch.nn.Identity()
        elif hasattr(model, "fc"):
            model.fc = torch.nn.Identity()

        model.eval()
        model.to(self._device)
        self._model = model
        self.cfg.embedding_dim = embed_dim

        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize((self.cfg.imgsz, self.cfg.imgsz)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def close(self) -> None:
        self._model = None
        self._transform = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ── embedding API ─────────────────────────────────────

    def embed(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return a normalised 1-D float32 embedding vector."""
        import torch

        if not self.is_loaded:
            self.load()

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb).unsqueeze(0).to(self._device)

        with torch.inference_mode():
            feat = self._model(tensor)

        vec = feat.squeeze().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_batch(self, images: list[np.ndarray]) -> np.ndarray:
        """Embed a batch → (N, D) float32 matrix, L2-normalised rows."""
        import torch

        if not self.is_loaded:
            self.load()

        tensors = []
        for img in images:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensors.append(self._transform(rgb))

        batch = torch.stack(tensors).to(self._device)
        with torch.inference_mode():
            feats = self._model(batch)

        vecs = feats.cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs /= norms
        return vecs
