"""Logo Retrieval Brand Match — feature embedding extractor.

Uses a pretrained torchvision backbone (classification head removed)
to produce a fixed-length embedding vector for any logo image.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from config import LogoConfig

# backbone name → (torchvision weights enum name, embedding dim)
_BACKBONE_INFO: dict[str, tuple[str, int]] = {
    "efficientnet_b0": ("EfficientNet_B0_Weights", 1280),
    "efficientnet_b2": ("EfficientNet_B2_Weights", 1408),
    "resnet50":        ("ResNet50_Weights",         2048),
    "resnet18":        ("ResNet18_Weights",          512),
    "mobilenet_v3_small": ("MobileNet_V3_Small_Weights", 576),
}


class LogoEmbedder:
    """Extract embeddings from logo images using a pretrained CNN backbone."""

    def __init__(self, cfg: LogoConfig | None = None) -> None:
        if cfg is None:
            from config import LogoConfig
            cfg = LogoConfig()
        self.cfg = cfg
        self._model = None
        self._transform = None
        self._device = None

    # ── lifecycle ──────────────────────────────────────────

    def load(self) -> None:
        """Load the backbone and set up the image transform."""
        import torch
        import torchvision.models as models
        import torchvision.transforms as T

        device = self.cfg.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        backbone_name = self.cfg.backbone
        if backbone_name not in _BACKBONE_INFO:
            raise ValueError(
                f"Unknown backbone: {backbone_name}. "
                f"Choose from: {list(_BACKBONE_INFO)}"
            )

        weights_cls_name, embed_dim = _BACKBONE_INFO[backbone_name]
        weights_cls = getattr(models, weights_cls_name)
        weights = weights_cls.DEFAULT

        model_fn = getattr(models, backbone_name)
        full_model = model_fn(weights=weights)

        # Remove classification head — keep only feature extractor
        if hasattr(full_model, "classifier"):
            full_model.classifier = torch.nn.Identity()
        elif hasattr(full_model, "fc"):
            full_model.fc = torch.nn.Identity()

        full_model.eval()
        full_model.to(self._device)
        self._model = full_model

        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize((self.cfg.imgsz, self.cfg.imgsz)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        self.cfg.embedding_dim = embed_dim

    def close(self) -> None:
        self._model = None
        self._transform = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ── embedding API ─────────────────────────────────────

    def embed(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return a normalised embedding vector (1-D float32)."""
        import torch

        if self._model is None:
            self.load()

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._transform(image_rgb).unsqueeze(0).to(self._device)

        with torch.inference_mode():
            feat = self._model(tensor)

        vec = feat.squeeze().cpu().numpy().astype(np.float32)
        # L2-normalise
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_batch(self, images: list[np.ndarray]) -> np.ndarray:
        """Embed a batch of BGR images → (N, D) float32 matrix."""
        import torch

        if self._model is None:
            self.load()

        tensors = []
        for img in images:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensors.append(self._transform(rgb))

        batch = torch.stack(tensors).to(self._device)

        with torch.inference_mode():
            feats = self._model(batch)

        vecs = feats.cpu().numpy().astype(np.float32)
        # L2-normalise rows
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs /= norms
        return vecs
