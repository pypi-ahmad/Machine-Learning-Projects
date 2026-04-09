#!/usr/bin/env python3
"""
Image-to-Image Translation — Pix2Pix (PyTorch)
================================================
Dataset : https://www.kaggle.com/datasets/sabahesaraki/pix2pix-facades-dataset
Run     : python "GANS/Image-to-Image Translation/run.py"

Pix2Pix cGAN for paired image translation on the facades dataset.
  * Generator  : U-Net with skip connections
  * Discriminator : PatchGAN (70×70 receptive field)
  * Loss       : cGAN (BCE) + L1 reconstruction (λ = 100)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils as vutils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

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
KAGGLE_SLUG  = "sabahesaraki/pix2pix-facades-dataset"
IMAGE_SIZE   = 256
BATCH_SIZE   = 8
NUM_EPOCHS   = 10
LR           = 2e-4
BETA1        = 0.5
LAMBDA_L1    = 100


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
#  Paired-image dataset
# ═════════════════════════════════════════════════════════════

class PairedImageDataset(Dataset):
    """Loads images where left half = input, right half = target (or vice versa)."""

    def __init__(self, root: Path, image_size: int = IMAGE_SIZE) -> None:
        self.files = sorted(
            list(root.rglob("*.jpg")) + list(root.rglob("*.png"))
        )
        if not self.files:
            raise FileNotFoundError(f"No images found under {root}")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        img = Image.open(self.files[idx]).convert("RGB")
        w, h = img.size
        half = w // 2
        # Convention: left = condition (label map), right = real photo
        input_img = img.crop((0, 0, half, h))
        target_img = img.crop((half, 0, w, h))
        return self.transform(input_img), self.transform(target_img)


# ═════════════════════════════════════════════════════════════
#  Generator — U-Net with skip connections
# ═════════════════════════════════════════════════════════════

class UNetDown(nn.Module):
    def __init__(self, in_c: int, out_c: int, normalize: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_c: int, out_c: int, dropout: bool = False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_c),
            nn.ReLU(True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.model(x)
        return torch.cat([x, skip], dim=1)


class Generator(nn.Module):
    """U-Net generator: 3-ch input → 3-ch output at IMAGE_SIZE×IMAGE_SIZE."""

    def __init__(self) -> None:
        super().__init__()
        self.d1 = UNetDown(3, 64, normalize=False)    # 128
        self.d2 = UNetDown(64, 128)                    # 64
        self.d3 = UNetDown(128, 256)                   # 32
        self.d4 = UNetDown(256, 512)                   # 16
        self.d5 = UNetDown(512, 512)                   # 8
        self.d6 = UNetDown(512, 512)                   # 4
        self.d7 = UNetDown(512, 512)                   # 2
        self.d8 = UNetDown(512, 512, normalize=False)  # 1

        self.u1 = UNetUp(512, 512, dropout=True)       # 2
        self.u2 = UNetUp(1024, 512, dropout=True)      # 4
        self.u3 = UNetUp(1024, 512, dropout=True)      # 8
        self.u4 = UNetUp(1024, 512)                    # 16
        self.u5 = UNetUp(1024, 256)                    # 32
        self.u6 = UNetUp(512, 128)                     # 64
        self.u7 = UNetUp(256, 64)                      # 128

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),       # 256
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)

        u1 = self.u1(d8, d7)
        u2 = self.u2(u1, d6)
        u3 = self.u3(u2, d5)
        u4 = self.u4(u3, d4)
        u5 = self.u5(u4, d3)
        u6 = self.u6(u5, d2)
        u7 = self.u7(u6, d1)
        return self.final(u7)


# ═════════════════════════════════════════════════════════════
#  Discriminator — PatchGAN (70×70)
# ═════════════════════════════════════════════════════════════

class Discriminator(nn.Module):
    """PatchGAN discriminator.  Input = concatenation of condition + image (6 ch)."""

    def __init__(self) -> None:
        super().__init__()

        def block(in_c, out_c, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(6, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 1, 4, padding=1, bias=False),
        )

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        x = torch.cat([img_a, img_b], dim=1)
        return self.model(x)


# ═════════════════════════════════════════════════════════════
#  Data loading
# ═════════════════════════════════════════════════════════════

def get_data(data_dir: Path):
    """Return train and (optional) val DataLoaders."""
    ds_path = download_kaggle_dataset(KAGGLE_SLUG, data_dir,
                                      dataset_name="Pix2Pix Facades")

    # Look for a train/ subfolder; fall back to ds_path
    train_root = ds_path
    for cand in [ds_path / "train", ds_path / "facades" / "train", ds_path]:
        if cand.is_dir() and (list(cand.glob("*.jpg")) or list(cand.glob("*.png"))):
            train_root = cand
            break

    logger.info("Train image root: %s", train_root)

    train_ds = PairedImageDataset(train_root, IMAGE_SIZE)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    logger.info("Train loader: %d images, %d batches.",
                len(train_ds), len(train_loader))
    return train_loader


# ═════════════════════════════════════════════════════════════
#  Training
# ═════════════════════════════════════════════════════════════

def train(loader: DataLoader, device: torch.device, output_dir: Path,
          use_amp: bool = True, num_epochs: int = 10,
          grad_accum: int = 1) -> tuple:
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)
    logger.info("Generator params : %s", f"{sum(p.numel() for p in netG.parameters()):,}")
    logger.info("Discriminator params: %s", f"{sum(p.numel() for p in netD.parameters()):,}")

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1  = nn.L1Loss()

    optimG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))
    optimD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))

    scaler_g = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_d = torch.amp.GradScaler("cuda", enabled=use_amp)

    fixed_input = None
    fixed_target = None

    for epoch in range(1, num_epochs + 1):
        netG.zero_grad()
        netD.zero_grad()
        for i, (inp, target) in enumerate(loader):
            inp = inp.to(device)
            target = target.to(device)
            b = inp.size(0)

            if fixed_input is None:
                fixed_input = inp[:min(4, b)].clone()
                fixed_target = target[:min(4, b)].clone()

            real_label = torch.ones(b, 1, 1, 1, device=device)
            fake_label = torch.zeros(b, 1, 1, 1, device=device)

            # ── Generator ──
            with torch.amp.autocast("cuda", enabled=use_amp):
                fake = netG(inp)
                pred_fake = netD(inp, fake)
                real_label_expanded = real_label.expand_as(pred_fake)
                loss_gan = criterion_gan(pred_fake, real_label_expanded)
                loss_l1 = criterion_l1(fake, target)
                lossG = loss_gan + LAMBDA_L1 * loss_l1

            scaler_g.scale(lossG / grad_accum).backward()
            if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
                scaler_g.step(optimG)
                scaler_g.update()
                netG.zero_grad()

            # ── Discriminator ──
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_real = netD(inp, target)
                real_label_expanded = real_label.expand_as(pred_real)
                loss_real = criterion_gan(pred_real, real_label_expanded)

                pred_fake_d = netD(inp, fake.detach())
                fake_label_expanded = fake_label.expand_as(pred_fake_d)
                loss_fake = criterion_gan(pred_fake_d, fake_label_expanded)
                lossD = (loss_real + loss_fake) * 0.5

            scaler_d.scale(lossD / grad_accum).backward()
            if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
                scaler_d.step(optimD)
                scaler_d.update()
                netD.zero_grad()

            if i % 20 == 0:
                logger.info(
                    "[%d/%d][%d/%d]  LossD=%.4f  LossG=%.4f  (gan=%.4f  L1=%.4f)",
                    epoch, num_epochs, i, len(loader),
                    lossD.item(), lossG.item(),
                    loss_gan.item(), loss_l1.item(),
                )

        # Epoch-end comparison
        _save_comparison(netG, fixed_input, fixed_target, device, output_dir, epoch)

    torch.save(netG.state_dict(), str(output_dir / "pix2pix_generator.pth"))
    torch.save(netD.state_dict(), str(output_dir / "pix2pix_discriminator.pth"))
    logger.info("Model weights saved → %s", output_dir)

    return netG, lossG.item(), lossD.item()


# ═════════════════════════════════════════════════════════════
#  Visualisation
# ═════════════════════════════════════════════════════════════

def _save_comparison(
    netG: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    output_dir: Path,
    epoch: int,
) -> None:
    netG.eval()
    with torch.no_grad():
        fakes = netG(inputs.to(device)).cpu()
    netG.train()

    imgs = []
    for j in range(inputs.size(0)):
        imgs.extend([inputs[j].cpu(), targets[j].cpu(), fakes[j]])

    grid = vutils.make_grid(imgs, nrow=3, normalize=True, padding=2)

    fig, ax = plt.subplots(figsize=(12, 4 * inputs.size(0)))
    ax.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
    ax.set_title(f"Epoch {epoch} — Input | Target | Generated")
    ax.axis("off")
    fig.tight_layout()
    path = output_dir / f"pix2pix_epoch_{epoch:02d}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Comparison grid saved → %s", path)


# ═════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════

def main() -> None:
    global NUM_EPOCHS, BATCH_SIZE

    setup_logging()
    args = parse_common_args("Image-to-Image Translation — Pix2Pix")
    set_seed(args.seed)
    configure_cuda_allocator()
    paths = project_paths(__file__)

    device = resolve_device_from_args(args)
    use_amp = not args.no_amp and device.type == "cuda"
    NUM_EPOCHS = args.epochs or 15
    BATCH_SIZE = args.batch_size or BATCH_SIZE

    if args.mode == "smoke":
        NUM_EPOCHS = 1

    batch_size, grad_accum = auto_batch_and_accum(args.gpu_mem_gb, BATCH_SIZE, min_batch=2)
    BATCH_SIZE = batch_size

    logger.info("=" * 60)
    logger.info("  Image-to-Image Translation — Pix2Pix (PyTorch)")
    logger.info("=" * 60)

    try:
        loader = get_data(paths["data"])
    except Exception as exc:
        logger.warning("Dataset loading failed: %s", exc)
        dataset_missing_metrics(
            paths["outputs"], "Pix2Pix Facades",
            ["https://www.kaggle.com/datasets/sabahesaraki/pix2pix-facades-dataset"],
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
