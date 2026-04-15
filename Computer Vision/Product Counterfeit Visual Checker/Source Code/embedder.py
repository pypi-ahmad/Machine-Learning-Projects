"""Product Counterfeit Visual Checker — feature extraction.
"""Product Counterfeit Visual Checker — feature extraction.

Extracts L2-normalised embeddings from product images using a
pretrained CNN backbone.
"""
"""

from __future__ import annotations

import logging
from typing import Sequence

import cv2
import numpy as np
import torch
from torchvision import models, transforms

logger = logging.getLogger(__name__)

_BACKBONES: dict[str, tuple[str, int]] = {
    "efficientnet_b0": ("efficientnet_b0", 1280),
    "efficientnet_b2": ("efficientnet_b2", 1408),
    "resnet50": ("resnet50", 2048),
    "resnet18": ("resnet18", 512),
    "mobilenet_v3_small": ("mobilenet_v3_small", 576),
    "mobilenet_v3_large": ("mobilenet_v3_large", 960),
}


class ProductEmbedder:
    """Extract embedding vectors from product images."""

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        embedding_dim: int = 1280,
        imgsz: int = 224,
        device: str | None = None,
    ) -> None:
        self.backbone_name = backbone
        self.embedding_dim = embedding_dim
        self.imgsz = imgsz
        self._device_name = device
        self._model: torch.nn.Module | None = None
        self._device: torch.device | None = None
        self._transform: transforms.Compose | None = None

    # ── lifecycle ──────────────────────────────────────────

    def load(self) -> None:
        if self._device_name:
            self._device = torch.device(self._device_name)
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        name, expected_dim = _BACKBONES.get(
            self.backbone_name, (self.backbone_name, self.embedding_dim)
        )
        weights_enum = getattr(models, f"{name.split('_')[0].capitalize()}_Weights", None)
        if weights_enum is None:
            weights_enum = getattr(models, "DEFAULT", None)

        builder = getattr(models, name)
        self._model = builder(weights="DEFAULT")

        # Remove classification head
        if hasattr(self._model, "classifier"):
            self._model.classifier = torch.nn.Identity()
        elif hasattr(self._model, "fc"):
            self._model.fc = torch.nn.Identity()

        self._model.eval()
        self._model.to(self._device)

        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.imgsz, self.imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        logger.info("Loaded %s on %s (dim=%d)", name, self._device, expected_dim)

    def close(self) -> None:
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── embedding API ──────────────────────────────────────

    @torch.no_grad()
    def embed(self, image_bgr: np.ndarray) -> np.ndarray:
        """Embed a single BGR image -> L2-normalised vector."""
        assert self._model is not None, "Call load() first"
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb).unsqueeze(0).to(self._device)
        vec = self._model(tensor).squeeze().cpu().numpy().astype(np.float32)
        vec = vec.flatten()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    @torch.no_grad()
    def embed_batch(self, images_bgr: Sequence[np.ndarray]) -> np.ndarray:
        """Embed a batch of BGR images -> (N, D) L2-normalised."""
        assert self._model is not None, "Call load() first"
        tensors = []
        for img in images_bgr:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensors.append(self._transform(rgb))
        batch = torch.stack(tensors).to(self._device)
        vecs = self._model(batch).cpu().numpy().astype(np.float32)
        vecs = vecs.reshape(len(images_bgr), -1)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        vecs /= norms
        return vecs

    @torch.no_grad()
    def embed_patches(
        self,
        image_bgr: np.ndarray,
        grid: tuple[int, int] = (3, 3),
    ) -> np.ndarray:
        """Embed image patches from a grid layout → (R*C, D).
        """Embed image patches from a grid layout → (R*C, D).

        Splits the image into *grid* (rows, cols) patches and embeds each.
        Used for region-aware comparison.
        """
        """
        assert self._model is not None, "Call load() first"
        rows, cols = grid
        h, w = image_bgr.shape[:2]
        ph, pw = h // rows, w // cols

        patches = []
        for r in range(rows):
            for c in range(cols):
                y0, y1 = r * ph, (r + 1) * ph if r < rows - 1 else h
                x0, x1 = c * pw, (c + 1) * pw if c < cols - 1 else w
                patch = image_bgr[y0:y1, x0:x1]
                patches.append(patch)

        return self.embed_batch(patches)
