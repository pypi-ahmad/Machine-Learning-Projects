#!/usr/bin/env python3
"""
Image Inpainting — Context Encoder (PyTorch)
=============================================
Dataset : https://www.kaggle.com/datasets/nickj26/places2-mit-dataset
Run     : python "GANS/Image Inpainting/run.py"

Trains a U-Net-style encoder-decoder (context encoder) alongside an
adversarial discriminator to reconstruct randomly masked rectangular
regions of 128×128 images.  Loss = MSE reconstruction + adversarial BCE.
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
KAGGLE_SLUG  = "nickj26/places2-mit-dataset"
IMAGE_SIZE   = 128
MASK_SIZE    = 48          # square mask side
BATCH_SIZE   = 32
NUM_EPOCHS   = 5
LR_G        = 2e-4
LR_D        = 2e-4
BETA1       = 0.5
LAMBDA_REC  = 0.999       # reconstruction weight
LAMBDA_ADV  = 0.001       # adversarial weight
MAX_IMAGES  = 20_000      # cap for demo speed


# ═════════════════════════════════════════════════════════════
#  Weight init
# ═════════════════════════════════════════════════════════════

def weights_init(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ═════════════════════════════════════════════════════════════
#  Masking utility
# ═════════════════════════════════════════════════════════════

def apply_random_mask(images: torch.Tensor, mask_size: int = MASK_SIZE):
    """Apply a random rectangular mask to a batch of images.

    Returns (masked_images, masks) where mask is 1 inside the hole.
    """
    b, c, h, w = images.shape
    masks = torch.zeros(b, 1, h, w, device=images.device)
    masked = images.clone()
    for i in range(b):
        y = torch.randint(0, h - mask_size, (1,)).item()
        x = torch.randint(0, w - mask_size, (1,)).item()
        masks[i, :, y:y + mask_size, x:x + mask_size] = 1.0
        masked[i, :, y:y + mask_size, x:x + mask_size] = 0.0
    return masked, masks


# ═════════════════════════════════════════════════════════════
#  Generator — U-Net-style Encoder-Decoder
# ═════════════════════════════════════════════════════════════

class _EncoderBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, use_bn: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, use_dropout: bool = False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    """U-Net encoder-decoder for inpainting.

    Input : masked image (3 ch) + mask (1 ch) = 4 channels, 128×128
    Output: reconstructed image 3 ch, 128×128
    """

    def __init__(self) -> None:
        super().__init__()
        # Encoder
        self.e1 = _EncoderBlock(4, 64, use_bn=False)    # 64×64
        self.e2 = _EncoderBlock(64, 128)                 # 32×32
        self.e3 = _EncoderBlock(128, 256)                # 16×16
        self.e4 = _EncoderBlock(256, 512)                # 8×8
        self.e5 = _EncoderBlock(512, 512)                # 4×4

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1, bias=False),   # 2×2
            nn.ReLU(True),
        )

        # Decoder (with skip connections)
        self.d1 = _DecoderBlock(512, 512, use_dropout=True)   # 4×4
        self.d2 = _DecoderBlock(1024, 512)                    # 8×8
        self.d3 = _DecoderBlock(1024, 256)                    # 16×16
        self.d4 = _DecoderBlock(512, 128)                     # 32×32
        self.d5 = _DecoderBlock(256, 64)                      # 64×64

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),  # 128×128
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        b  = self.bottleneck(e5)

        d1 = self.d1(b)
        d2 = self.d2(torch.cat([d1, e5], dim=1))
        d3 = self.d3(torch.cat([d2, e4], dim=1))
        d4 = self.d4(torch.cat([d3, e3], dim=1))
        d5 = self.d5(torch.cat([d4, e2], dim=1))
        return self.final(torch.cat([d5, e1], dim=1))


# ═════════════════════════════════════════════════════════════
#  Discriminator — PatchGAN
# ═════════════════════════════════════════════════════════════

class Discriminator(nn.Module):
    """PatchGAN discriminator operating on 128×128 images."""

    def __init__(self) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3, 64, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(256, 1, 4, 2, 1, bias=False)),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x).view(x.size(0), -1).mean(dim=1)


# ═════════════════════════════════════════════════════════════
#  Data loading
# ═════════════════════════════════════════════════════════════

def get_data(data_dir: Path) -> DataLoader:
    """Download Places2 (or subset) and return a DataLoader."""
    from torchvision.datasets import ImageFolder

    ds_path = download_kaggle_dataset(KAGGLE_SLUG, data_dir,
                                      dataset_name="Places2 MIT")

    # Find the first directory that contains images (recursive search)
    img_root = ds_path
    for candidate in sorted(ds_path.rglob("*")):
        if candidate.is_dir() and any(candidate.glob("*.jpg")) or any(candidate.glob("*.png")):
            # Need a parent of class-folders for ImageFolder
            if any(sub.is_dir() for sub in candidate.iterdir()):
                img_root = candidate
                break

    # If img_root still equals ds_path and has no sub-dirs with images,
    # wrap it so ImageFolder works.
    subdirs = [d for d in img_root.iterdir() if d.is_dir()]
    if not subdirs:
        wrapper = data_dir / "_places_wrapper" / "images"
        wrapper.mkdir(parents=True, exist_ok=True)
        if not wrapper.exists() or not any(wrapper.iterdir()):
            try:
                wrapper.symlink_to(img_root.resolve())
            except OSError:
                import shutil
                shutil.copytree(str(img_root), str(wrapper), dirs_exist_ok=True)
        img_root = wrapper.parent

    logger.info("Image root: %s", img_root)

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = ImageFolder(str(img_root), transform=transform)
    if len(dataset) > MAX_IMAGES:
        dataset = torch.utils.data.Subset(dataset, list(range(MAX_IMAGES)))
        logger.info("Capped dataset to %d images.", MAX_IMAGES)

    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    logger.info("DataLoader ready — %d images, %d batches.",
                len(dataset), len(loader))
    return loader


# ═════════════════════════════════════════════════════════════
#  Training
# ═════════════════════════════════════════════════════════════

def train(loader: DataLoader, device: torch.device, output_dir: Path,
          use_amp: bool = True, num_epochs: int = 5,
          grad_accum: int = 1) -> tuple:
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)
    logger.info("Generator params : %s", f"{sum(p.numel() for p in netG.parameters()):,}")
    logger.info("Discriminator params: %s", f"{sum(p.numel() for p in netD.parameters()):,}")

    criterion_bce = nn.BCELoss()
    criterion_mse = nn.MSELoss()

    optimG = optim.Adam(netG.parameters(), lr=LR_G, betas=(BETA1, 0.999))
    optimD = optim.Adam(netD.parameters(), lr=LR_D, betas=(BETA1, 0.999))

    scaler_g = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_d = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Keep a fixed batch for visual comparisons
    fixed_batch = None

    for epoch in range(1, num_epochs + 1):
        netD.zero_grad()
        netG.zero_grad()
        for i, (real, _) in enumerate(loader):
            real = real.to(device)
            b = real.size(0)

            if fixed_batch is None:
                fixed_batch = real[:min(8, b)].clone()

            masked, masks = apply_random_mask(real)
            gen_input = torch.cat([masked, masks], dim=1)  # 4-ch

            label_real = torch.ones(b, device=device)
            label_fake = torch.zeros(b, device=device)

            # ── Discriminator ──
            with torch.amp.autocast("cuda", enabled=use_amp):
                recon = netG(gen_input)
                d_real = netD(real)
                d_fake = netD(recon.detach())
                lossD = (criterion_bce(d_real, label_real) +
                         criterion_bce(d_fake, label_fake)) * 0.5

            scaler_d.scale(lossD / grad_accum).backward()
            if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
                scaler_d.step(optimD)
                scaler_d.update()
                netD.zero_grad()

            # ── Generator ──
            with torch.amp.autocast("cuda", enabled=use_amp):
                d_fake2 = netD(recon)
                loss_adv = criterion_bce(d_fake2, label_real)
                loss_rec = criterion_mse(recon, real)
                lossG = LAMBDA_ADV * loss_adv + LAMBDA_REC * loss_rec

            scaler_g.scale(lossG / grad_accum).backward()
            if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
                scaler_g.step(optimG)
                scaler_g.update()
                netG.zero_grad()

            if i % 50 == 0:
                logger.info(
                    "[%d/%d][%d/%d]  LossD=%.4f  LossG=%.4f  (rec=%.4f adv=%.4f)",
                    epoch, num_epochs, i, len(loader),
                    lossD.item(), lossG.item(),
                    loss_rec.item(), loss_adv.item(),
                )

        # Epoch-end visualisation
        _save_comparison(netG, fixed_batch, device, output_dir, epoch)

    torch.save(netG.state_dict(), str(output_dir / "inpainting_generator.pth"))
    torch.save(netD.state_dict(), str(output_dir / "inpainting_discriminator.pth"))
    logger.info("Model weights saved → %s", output_dir)
    return netG, lossG.item(), lossD.item()


# ═════════════════════════════════════════════════════════════
#  Visualisation
# ═════════════════════════════════════════════════════════════

def _save_comparison(
    netG: nn.Module,
    originals: torch.Tensor,
    device: torch.device,
    output_dir: Path,
    epoch: int,
) -> None:
    """Save original / masked / reconstructed comparison grid."""
    netG.eval()
    with torch.no_grad():
        masked, masks = apply_random_mask(originals)
        gen_input = torch.cat([masked.to(device), masks.to(device)], dim=1)
        recon = netG(gen_input).cpu()
    netG.train()

    n = originals.size(0)
    rows = []
    for j in range(n):
        rows.extend([originals[j], masked[j], recon[j]])

    grid = vutils.make_grid(rows, nrow=3, normalize=True, padding=2)

    fig, ax = plt.subplots(figsize=(9, 3 * n))
    ax.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
    ax.set_title(f"Epoch {epoch} — Original | Masked | Reconstructed")
    ax.axis("off")
    fig.tight_layout()
    path = output_dir / f"comparison_epoch_{epoch:02d}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Comparison grid saved → %s", path)


# ═════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════

def main() -> None:
    global NUM_EPOCHS, BATCH_SIZE, MAX_IMAGES

    setup_logging()
    args = parse_common_args("Image Inpainting — Context Encoder")
    set_seed(args.seed)
    configure_cuda_allocator()
    paths = project_paths(__file__)

    device = resolve_device_from_args(args)
    use_amp = not args.no_amp and device.type == "cuda"
    NUM_EPOCHS = args.epochs or 10
    BATCH_SIZE = args.batch_size or BATCH_SIZE

    if args.mode == "smoke":
        NUM_EPOCHS = 1
        MAX_IMAGES = 500

    batch_size, grad_accum = auto_batch_and_accum(args.gpu_mem_gb, BATCH_SIZE, min_batch=2)
    BATCH_SIZE = batch_size

    logger.info("=" * 60)
    logger.info("  Image Inpainting — Context Encoder (PyTorch)")
    logger.info("=" * 60)

    try:
        loader = get_data(paths["data"])
    except Exception as exc:
        logger.warning("Dataset loading failed: %s", exc)
        dataset_missing_metrics(
            paths["outputs"], "Places2 MIT",
            ["https://www.kaggle.com/datasets/nickj26/places2-mit-dataset"],
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
        counts={"train": len(loader.dataset)},
        extras={"note": "GAN training uses full dataset; no test split needed"},
    )

    _, g_loss_final, d_loss_final = train(
        loader, device, paths["outputs"], use_amp=use_amp,
        num_epochs=NUM_EPOCHS, grad_accum=grad_accum,
    )

    metrics = {
        "g_loss_final": g_loss_final,
        "d_loss_final": d_loss_final,
        "epochs_ran": NUM_EPOCHS,
        "samples_saved": True,
    }
    metrics["run_metadata"] = run_metadata(args)
    save_metrics(paths["outputs"], metrics, task_type="gan", mode=args.mode)

    logger.info("Done ✓")


if __name__ == "__main__":
    main()
