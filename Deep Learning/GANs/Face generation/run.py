#!/usr/bin/env python3
"""
Face Generation — DCGAN on CelebA
==================================
Dataset : https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
Run     : python "GANS/Face generation/run.py"

Generates 64×64 RGB faces using a Deep Convolutional GAN (DCGAN) trained
on the CelebA dataset.  Uses spectral normalisation on the discriminator
and mixed-precision training when a CUDA device is available.
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
KAGGLE_SLUG = "jessicali9530/celeba-dataset"
IMAGE_SIZE  = 64
BATCH_SIZE  = 128
NZ          = 100        # latent vector length
NGF         = 64         # generator feature-map depth
NDF         = 64         # discriminator feature-map depth
NUM_EPOCHS  = 5
LR          = 2e-4
BETA1       = 0.5
MAX_IMAGES  = 30_000     # cap to speed up demo runs


# ═════════════════════════════════════════════════════════════
#  Weight initialisation
# ═════════════════════════════════════════════════════════════

def weights_init(m: nn.Module) -> None:
    """DCGAN-style weight initialisation (mean=0, std=0.02)."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ═════════════════════════════════════════════════════════════
#  Generator
# ═════════════════════════════════════════════════════════════

class Generator(nn.Module):
    """DCGAN Generator: z(NZ) → 3×64×64 RGB image."""

    def __init__(self, nz: int = NZ, ngf: int = NGF) -> None:
        super().__init__()
        self.main = nn.Sequential(
            # input: z (nz × 1 × 1)
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state: (ngf*8) × 4 × 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state: (ngf*4) × 8 × 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state: (ngf*2) × 16 × 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state: ngf × 32 × 32
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
            # output: 3 × 64 × 64
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


# ═════════════════════════════════════════════════════════════
#  Discriminator  (with spectral normalisation)
# ═════════════════════════════════════════════════════════════

class Discriminator(nn.Module):
    """DCGAN Discriminator with spectral normalisation: 3×64×64 → scalar."""

    def __init__(self, ndf: int = NDF) -> None:
        super().__init__()
        self.main = nn.Sequential(
            # input: 3 × 64 × 64
            nn.utils.spectral_norm(nn.Conv2d(3, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state: ndf × 32 × 32
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (ndf*2) × 16 × 16
            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (ndf*4) × 8 × 8
            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (ndf*8) × 4 × 4
            nn.utils.spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x).view(-1, 1).squeeze(1)


# ═════════════════════════════════════════════════════════════
#  Data loading
# ═════════════════════════════════════════════════════════════

def get_data(data_dir: Path) -> DataLoader:
    """Download CelebA and return a DataLoader of 64×64 centre-cropped faces."""
    ds_path = download_kaggle_dataset(KAGGLE_SLUG, data_dir, dataset_name="CelebA")

    # CelebA images live under  <ds_path>/img_align_celeba/img_align_celeba/
    img_root = ds_path
    for candidate in [
        ds_path / "img_align_celeba" / "img_align_celeba",
        ds_path / "img_align_celeba",
        ds_path,
    ]:
        if candidate.is_dir() and any(candidate.glob("*.jpg")):
            img_root = candidate
            break

    logger.info("Image root: %s", img_root)

    from torchvision.datasets import ImageFolder

    # ImageFolder needs sub-dirs; create a symlink-style wrapper
    # by wrapping img_root inside a parent so ImageFolder sees one class.
    wrapper = data_dir / "_celeba_wrapper"
    wrapper.mkdir(parents=True, exist_ok=True)
    link = wrapper / "faces"
    if not link.exists():
        try:
            link.symlink_to(img_root.resolve())
        except OSError:
            # On Windows without symlink privilege, copy-approach fallback
            import shutil
            if not link.exists():
                shutil.copytree(str(img_root), str(link), dirs_exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = ImageFolder(str(wrapper), transform=transform)
    # Limit dataset size for fast demo
    if len(dataset) > MAX_IMAGES:
        dataset = torch.utils.data.Subset(
            dataset, list(range(MAX_IMAGES))
        )
        logger.info("Capped dataset to %d images for demo speed.", MAX_IMAGES)

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

def train(
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    use_amp: bool = True,
    num_epochs: int = 5,
    grad_accum: int = 1,
) -> tuple:
    """Train DCGAN and return (generator, g_loss_final, d_loss_final)."""
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)
    logger.info("Generator params : %s", f"{sum(p.numel() for p in netG.parameters()):,}")
    logger.info("Discriminator params: %s", f"{sum(p.numel() for p in netD.parameters()):,}")

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, NZ, 1, 1, device=device)

    optimG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))
    optimD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))

    scaler_g = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_d = torch.amp.GradScaler("cuda", enabled=use_amp)

    real_label, fake_label = 1.0, 0.0

    for epoch in range(1, num_epochs + 1):
        netD.zero_grad()
        netG.zero_grad()
        for i, (real, _) in enumerate(loader):
            real = real.to(device)
            b_size = real.size(0)
            label_real = torch.full((b_size,), real_label, device=device)
            label_fake = torch.full((b_size,), fake_label, device=device)

            # ── Update Discriminator ──
            with torch.amp.autocast("cuda", enabled=use_amp):
                out_real = netD(real)
                lossD_real = criterion(out_real, label_real)

                noise = torch.randn(b_size, NZ, 1, 1, device=device)
                fake = netG(noise)
                out_fake = netD(fake.detach())
                lossD_fake = criterion(out_fake, label_fake)
                lossD = lossD_real + lossD_fake

            scaler_d.scale(lossD / grad_accum).backward()
            if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
                scaler_d.step(optimD)
                scaler_d.update()
                netD.zero_grad()

            # ── Update Generator ──
            with torch.amp.autocast("cuda", enabled=use_amp):
                out_fake2 = netD(fake)
                lossG = criterion(out_fake2, label_real)

            scaler_g.scale(lossG / grad_accum).backward()
            if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
                scaler_g.step(optimG)
                scaler_g.update()
                netG.zero_grad()

            if i % 100 == 0:
                logger.info(
                    "[%d/%d][%d/%d]  Loss_D=%.4f  Loss_G=%.4f  D(x)=%.3f  D(G(z))=%.3f",
                    epoch, num_epochs, i, len(loader),
                    lossD.item(), lossG.item(),
                    out_real.mean().item(), out_fake2.mean().item(),
                )

        # Save sample grid after each epoch
        with torch.no_grad():
            samples = netG(fixed_noise).detach().cpu()
        grid = vutils.make_grid(samples, nrow=8, normalize=True, padding=2)
        save_path = output_dir / f"generated_epoch_{epoch:02d}.png"
        vutils.save_image(grid, str(save_path))
        logger.info("Saved sample grid → %s", save_path)

    # Save final models
    torch.save(netG.state_dict(), str(output_dir / "generator.pth"))
    torch.save(netD.state_dict(), str(output_dir / "discriminator.pth"))
    logger.info("Model weights saved to %s", output_dir)

    return netG, lossG.item(), lossD.item()


# ═════════════════════════════════════════════════════════════
#  Visualisation
# ═════════════════════════════════════════════════════════════

def save_final_grid(netG: nn.Module, device: torch.device, output_dir: Path) -> None:
    """Generate and save a high-quality 8×8 grid of faces."""
    netG.eval()
    with torch.no_grad():
        noise = torch.randn(64, NZ, 1, 1, device=device)
        fakes = netG(noise).cpu()

    grid = vutils.make_grid(fakes, nrow=8, normalize=True, padding=2)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
    ax.set_title("DCGAN — Generated Faces (CelebA)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "final_generated_faces.png", dpi=150)
    plt.close(fig)
    logger.info("Final grid saved → %s", output_dir / "final_generated_faces.png")


# ═════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════

def main() -> None:
    global NUM_EPOCHS, BATCH_SIZE, MAX_IMAGES

    setup_logging()
    args = parse_common_args("Face Generation — DCGAN on CelebA")
    set_seed(args.seed)
    configure_cuda_allocator()
    paths = project_paths(__file__)

    device = resolve_device_from_args(args)
    use_amp = not args.no_amp and device.type == "cuda"
    NUM_EPOCHS = args.epochs or 15
    BATCH_SIZE = args.batch_size or BATCH_SIZE

    if args.mode == "smoke":
        NUM_EPOCHS = 1
        MAX_IMAGES = 500

    batch_size, grad_accum = auto_batch_and_accum(args.gpu_mem_gb, BATCH_SIZE, min_batch=2)
    BATCH_SIZE = batch_size

    logger.info("=" * 60)
    logger.info("  Face Generation — DCGAN on CelebA")
    logger.info("=" * 60)

    try:
        loader = get_data(paths["data"])
    except Exception as exc:
        logger.warning("Dataset loading failed: %s", exc)
        dataset_missing_metrics(
            paths["outputs"], "CelebA",
            ["https://www.kaggle.com/datasets/jessicali9530/celeba-dataset"],
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

    netG, g_loss_final, d_loss_final = train(
        loader, device, paths["outputs"], use_amp=use_amp,
        num_epochs=NUM_EPOCHS, grad_accum=grad_accum,
    )
    save_final_grid(netG, device, paths["outputs"])

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
