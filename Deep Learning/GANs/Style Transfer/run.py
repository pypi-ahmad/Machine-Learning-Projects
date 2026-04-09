#!/usr/bin/env python3
"""
Neural Style Transfer (PyTorch + VGG19)
========================================
Dataset : https://www.kaggle.com/datasets/steubk/wikiart
Run     : python "GANS/Style Transfer/run.py"

Optimisation-based neural style transfer using pre-trained VGG-19 features.
The input image is iteratively optimised to minimise a weighted combination
of content loss (feature-space MSE) and style loss (Gram-matrix MSE).

This is NOT a GAN — no generator/discriminator — but sits under the GANS/
umbrella because it is a generative image technique.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils as vutils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.utils import (
    download_kaggle_dataset, set_seed, setup_logging, project_paths,
    get_device, ensure_dir, dataset_prompt,
    parse_common_args, dataset_missing_metrics,
    configure_cuda_allocator, resolve_device_from_args, run_metadata,
    save_metrics, dataset_fingerprint, write_split_manifest,
    auto_batch_and_accum, get_gpu_mem_bytes,
)

logger = logging.getLogger(__name__)

# ── Hyper-parameters ─────────────────────────────────────────
KAGGLE_SLUG      = "steubk/wikiart"
IMAGE_SIZE       = 512
NUM_STEPS        = 300
STYLE_WEIGHT     = 1e6
CONTENT_WEIGHT   = 1.0
LR               = 0.01

# VGG layers to tap
CONTENT_LAYERS = ["conv_4"]
STYLE_LAYERS   = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]


# ═════════════════════════════════════════════════════════════
#  VGG feature extractor
# ═════════════════════════════════════════════════════════════

class VGGFeatures(nn.Module):
    """Extract content and style features from VGG-19."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights

        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
        for p in vgg.parameters():
            p.requires_grad_(False)

        # Rename layers for easy reference
        self.slices: nn.ModuleList = nn.ModuleList()
        self.layer_names: list = []
        conv_idx = 0
        current = nn.Sequential()
        for layer in vgg.children():
            if isinstance(layer, nn.Conv2d):
                conv_idx += 1
                name = f"conv_{conv_idx}"
            elif isinstance(layer, nn.ReLU):
                name = f"relu_{conv_idx}"
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f"pool_{conv_idx}"
            else:
                name = f"other_{conv_idx}"
            current.add_module(name, layer)

            if name in CONTENT_LAYERS or name in STYLE_LAYERS:
                self.slices.append(current)
                self.layer_names.append(name)
                current = nn.Sequential()

        self.slices = self.slices.to(device)

    def forward(self, x: torch.Tensor):
        features: dict = {}
        out = x
        for slc, name in zip(self.slices, self.layer_names):
            out = slc(out)
            features[name] = out
        return features


# ═════════════════════════════════════════════════════════════
#  Gram matrix
# ═════════════════════════════════════════════════════════════

def gram_matrix(tensor: torch.Tensor) -> torch.Tensor:
    b, c, h, w = tensor.size()
    features = tensor.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)


# ═════════════════════════════════════════════════════════════
#  Image I/O helpers
# ═════════════════════════════════════════════════════════════

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_loader = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

_unnorm = transforms.Normalize(
    mean=[-m / s for m, s in zip(_IMAGENET_MEAN, _IMAGENET_STD)],
    std=[1.0 / s for s in _IMAGENET_STD],
)


def load_image(path: Path, device: torch.device) -> torch.Tensor:
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return _loader(img).unsqueeze(0).to(device)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.cpu().clone().squeeze(0)
    img = _unnorm(img)
    img = img.clamp(0, 1)
    return np.transpose(img.numpy(), (1, 2, 0))


# ═════════════════════════════════════════════════════════════
#  Pick sample images from dataset
# ═════════════════════════════════════════════════════════════

def _find_sample_images(ds_path: Path):
    """Return (content_path, style_path) from the downloaded dataset."""
    all_imgs = sorted(
        list(ds_path.rglob("*.jpg")) + list(ds_path.rglob("*.png"))
    )
    if len(all_imgs) < 2:
        raise FileNotFoundError(
            f"Need at least 2 images in {ds_path}; found {len(all_imgs)}"
        )
    # Use the first image as content, second as style
    return all_imgs[0], all_imgs[1]


# ═════════════════════════════════════════════════════════════
#  Style transfer optimisation
# ═════════════════════════════════════════════════════════════

def run_style_transfer(
    content_path: Path,
    style_path: Path,
    device: torch.device,
    output_dir: Path,
    num_steps: int = NUM_STEPS,
) -> tuple:
    """Run neural style transfer and return (styled_tensor, content_loss, style_loss)."""
    content_img = load_image(content_path, device)
    style_img   = load_image(style_path, device)

    logger.info("Content image: %s", content_path.name)
    logger.info("Style   image: %s", style_path.name)

    # Start from a copy of the content image
    input_img = content_img.clone().requires_grad_(True)

    extractor = VGGFeatures(device)
    content_features = extractor(content_img)
    style_features   = extractor(style_img)

    # Pre-compute style Gram matrices
    style_grams = {
        layer: gram_matrix(style_features[layer])
        for layer in STYLE_LAYERS if layer in style_features
    }

    optimizer = optim.Adam([input_img], lr=LR)

    logger.info("Starting optimisation (%d steps) …", num_steps)
    for step in range(1, num_steps + 1):
        optimizer.zero_grad()

        feats = extractor(input_img)

        # Content loss
        content_loss = torch.tensor(0.0, device=device)
        for layer in CONTENT_LAYERS:
            if layer in feats and layer in content_features:
                content_loss += nn.functional.mse_loss(
                    feats[layer], content_features[layer]
                )

        # Style loss
        style_loss = torch.tensor(0.0, device=device)
        for layer in STYLE_LAYERS:
            if layer in feats and layer in style_grams:
                G = gram_matrix(feats[layer])
                style_loss += nn.functional.mse_loss(G, style_grams[layer])

        total_loss = CONTENT_WEIGHT * content_loss + STYLE_WEIGHT * style_loss
        total_loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == 1:
            logger.info(
                "Step %3d/%d  content=%.4f  style=%.6f  total=%.4f",
                step, num_steps,
                content_loss.item(), style_loss.item(), total_loss.item(),
            )

    # Save results
    styled_np = tensor_to_image(input_img)
    content_np = tensor_to_image(content_img)
    style_np = tensor_to_image(style_img)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(content_np); axes[0].set_title("Content"); axes[0].axis("off")
    axes[1].imshow(style_np);   axes[1].set_title("Style");   axes[1].axis("off")
    axes[2].imshow(styled_np);  axes[2].set_title("Result");  axes[2].axis("off")
    fig.tight_layout()
    comparison = output_dir / "style_transfer_comparison.png"
    fig.savefig(comparison, dpi=150)
    plt.close(fig)
    logger.info("Comparison saved → %s", comparison)

    # Save just the styled image
    result_path = output_dir / "styled_output.png"
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.imshow(styled_np)
    ax2.axis("off")
    fig2.tight_layout()
    fig2.savefig(result_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig2)
    logger.info("Styled output saved → %s", result_path)

    return input_img.detach(), content_loss.item(), style_loss.item()


# ═════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════

def main() -> None:
    setup_logging()
    args = parse_common_args("Neural Style Transfer (VGG19)")
    set_seed(args.seed)
    configure_cuda_allocator()
    paths = project_paths(__file__)

    device = resolve_device_from_args(args)
    num_steps = args.epochs or NUM_STEPS

    if args.mode == "smoke":
        num_steps = 50

    logger.info("=" * 60)
    logger.info("  Neural Style Transfer (PyTorch + VGG19)")
    logger.info("=" * 60)

    try:
        ds_path = download_kaggle_dataset(KAGGLE_SLUG, paths["data"],
                                          dataset_name="WikiArt")
        content_path, style_path = _find_sample_images(ds_path)
    except Exception as exc:
        logger.warning("Dataset loading failed: %s", exc)
        dataset_missing_metrics(
            paths["outputs"], "WikiArt",
            ["https://www.kaggle.com/datasets/steubk/wikiart"],
        )
        return

    if args.download_only:
        logger.info("Download complete — exiting (--download-only).")
        sys.exit(0)

    write_split_manifest(
        paths["outputs"],
        dataset_fp=dataset_fingerprint(paths["data"]),
        split_method="gan_full_train",
        seed=args.seed,
        counts={"train": 2},
        extras={"note": "Style transfer uses content+style image pair; no traditional dataset split"},
    )

    _, content_loss_final, style_loss_final = run_style_transfer(
        content_path, style_path, device, paths["outputs"],
        num_steps=num_steps,
    )

    metrics = {
        "g_loss_final": content_loss_final,
        "d_loss_final": style_loss_final,
        "epochs_ran": num_steps,
        "content_loss_final": content_loss_final,
        "style_loss_final": style_loss_final,
        "samples_saved": True,
    }
    metrics["run_metadata"] = run_metadata(args)
    save_metrics(paths["outputs"], metrics, task_type="gan", mode=args.mode)

    logger.info("Done ✓")


if __name__ == "__main__":
    main()
