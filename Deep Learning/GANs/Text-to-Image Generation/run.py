#!/usr/bin/env python3
"""
Text-to-Image Generation — Conditional GAN (PyTorch)
=====================================================
Dataset : https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset
Run     : python "GANS/Text-to-Image Generation/run.py"

A class-conditional GAN trained on COCO 2017.  Full text-to-image
synthesis is extremely complex, so this implementation conditions on
COCO category labels instead of raw captions:

  * Generator    : noise (NZ) + class embedding → 64×64 RGB
  * Discriminator: image + class embedding → real/fake

Trains for 5 epochs as a fast demo.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
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
KAGGLE_SLUG  = "awsaf49/coco-2017-dataset"
IMAGE_SIZE   = 64
BATCH_SIZE   = 128
NZ           = 100          # noise dimension
NGF          = 64
NDF          = 64
EMBED_DIM    = 128          # class embedding dimension
NUM_CLASSES  = 80           # COCO has 80 categories
NUM_EPOCHS   = 5
LR           = 2e-4
BETA1        = 0.5
MAX_IMAGES   = 30_000       # cap for fast demos


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
#  Generator — Conditional DCGAN
# ═════════════════════════════════════════════════════════════

class Generator(nn.Module):
    """Conditional Generator: z (NZ) + class embedding → 3×64×64."""

    def __init__(self, nz: int = NZ, ngf: int = NGF,
                 num_classes: int = NUM_CLASSES, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        in_dim = nz + embed_dim

        self.main = nn.Sequential(
            # (nz + embed_dim) × 1 × 1
            nn.ConvTranspose2d(in_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # ngf*8 × 4 × 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # ngf*4 × 8 × 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # ngf*2 × 16 × 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # ngf × 32 × 32
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
            # 3 × 64 × 64
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = self.label_emb(labels).unsqueeze(2).unsqueeze(3)  # (B, embed, 1, 1)
        x = torch.cat([noise, emb], dim=1)
        return self.main(x)


# ═════════════════════════════════════════════════════════════
#  Discriminator — Conditional DCGAN
# ═════════════════════════════════════════════════════════════

class Discriminator(nn.Module):
    """Conditional Discriminator: image (3ch) + embedded label → scalar."""

    def __init__(self, ndf: int = NDF,
                 num_classes: int = NUM_CLASSES, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        # Project label embedding to a spatial map that we concatenate with image
        self.label_proj = nn.Sequential(
            nn.Linear(embed_dim, IMAGE_SIZE * IMAGE_SIZE),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.main = nn.Sequential(
            # input: (3 + 1) × 64 × 64
            nn.utils.spectral_norm(nn.Conv2d(4, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf × 32 × 32
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf*2 × 16 × 16
            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf*4 × 8 × 8
            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf*8 × 4 × 4
            nn.utils.spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid(),
        )

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = self.label_emb(labels)                     # (B, embed_dim)
        label_map = self.label_proj(emb)                  # (B, H*W)
        label_map = label_map.view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
        x = torch.cat([images, label_map], dim=1)        # (B, 4, H, W)
        return self.main(x).view(-1, 1).squeeze(1)


# ═════════════════════════════════════════════════════════════
#  COCO dataset (image + category label)
# ═════════════════════════════════════════════════════════════

class COCOCategoryDataset(Dataset):
    """Loads COCO images with their *first* category label (0-79)."""

    def __init__(self, img_dir: Path, ann_file: Path,
                 transform=None, max_images: int = MAX_IMAGES) -> None:
        with open(ann_file, "r") as f:
            coco = json.load(f)

        # Build category id → contiguous index (0 … 79)
        cat_ids = sorted(c["id"] for c in coco["categories"])
        self.cat_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
        self.idx_to_name = {i: c["name"] for c in coco["categories"]
                            for i, cid in enumerate(cat_ids) if cid == c["id"]}

        # Map image_id → first annotation category
        img_to_cat: dict = {}
        for ann in coco["annotations"]:
            iid = ann["image_id"]
            if iid not in img_to_cat:
                cid = ann["category_id"]
                if cid in self.cat_to_idx:
                    img_to_cat[iid] = self.cat_to_idx[cid]

        # Build list of (path, label)
        self.samples = []
        for img_info in coco["images"]:
            iid = img_info["id"]
            if iid in img_to_cat:
                p = img_dir / img_info["file_name"]
                if p.exists():
                    self.samples.append((p, img_to_cat[iid]))
            if len(self.samples) >= max_images:
                break

        self.transform = transform
        logger.info("COCOCategoryDataset: %d samples, %d categories.",
                    len(self.samples), len(cat_ids))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ═════════════════════════════════════════════════════════════
#  Data loading
# ═════════════════════════════════════════════════════════════

def get_data(data_dir: Path):
    """Download COCO 2017 and return a DataLoader + category name map."""
    ds_path = download_kaggle_dataset(KAGGLE_SLUG, data_dir,
                                      dataset_name="COCO 2017")

    # Locate images and annotation file
    img_dir = None
    ann_file = None
    for candidate_img in [
        ds_path / "coco2017" / "train2017",
        ds_path / "train2017",
        ds_path / "images" / "train2017",
    ]:
        if candidate_img.is_dir():
            img_dir = candidate_img
            break

    for candidate_ann in [
        ds_path / "coco2017" / "annotations" / "instances_train2017.json",
        ds_path / "annotations" / "instances_train2017.json",
    ]:
        if candidate_ann.is_file():
            ann_file = candidate_ann
            break

    if img_dir is None or ann_file is None:
        # Fallback: scan recursively
        for p in sorted(ds_path.rglob("instances_train2017.json")):
            ann_file = p
            break
        for p in sorted(ds_path.rglob("train2017")):
            if p.is_dir():
                img_dir = p
                break

    if img_dir is None or ann_file is None:
        raise FileNotFoundError(
            f"Could not locate COCO train images / annotations in {ds_path}"
        )

    logger.info("COCO images : %s", img_dir)
    logger.info("COCO annots : %s", ann_file)

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = COCOCategoryDataset(img_dir, ann_file, transform, MAX_IMAGES)

    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    logger.info("DataLoader ready — %d images, %d batches.",
                len(dataset), len(loader))
    return loader, dataset.idx_to_name


# ═════════════════════════════════════════════════════════════
#  Training
# ═════════════════════════════════════════════════════════════

def train(
    loader: DataLoader,
    idx_to_name: dict,
    device: torch.device,
    output_dir: Path,
    use_amp: bool = True,
    num_epochs: int = 5,
    grad_accum: int = 1,
) -> tuple:
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)
    logger.info("Generator params : %s", f"{sum(p.numel() for p in netG.parameters()):,}")
    logger.info("Discriminator params: %s", f"{sum(p.numel() for p in netD.parameters()):,}")

    criterion = nn.BCELoss()

    optimG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))
    optimD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))

    scaler_g = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_d = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Fixed noise + labels for visual tracking
    n_vis = min(8, NUM_CLASSES)
    fixed_noise = torch.randn(n_vis * 8, NZ, 1, 1, device=device)
    fixed_labels = torch.arange(n_vis, device=device).repeat_interleave(8)

    real_label_val, fake_label_val = 1.0, 0.0

    for epoch in range(1, num_epochs + 1):
        netD.zero_grad()
        netG.zero_grad()
        for i, (real, labels) in enumerate(loader):
            real = real.to(device)
            labels = labels.to(device)
            b = real.size(0)

            label_real = torch.full((b,), real_label_val, device=device)
            label_fake = torch.full((b,), fake_label_val, device=device)

            # ── Discriminator ──
            with torch.amp.autocast("cuda", enabled=use_amp):
                out_real = netD(real, labels)
                lossD_real = criterion(out_real, label_real)

                noise = torch.randn(b, NZ, 1, 1, device=device)
                fake = netG(noise, labels)
                out_fake = netD(fake.detach(), labels)
                lossD_fake = criterion(out_fake, label_fake)
                lossD = lossD_real + lossD_fake

            scaler_d.scale(lossD / grad_accum).backward()
            if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
                scaler_d.step(optimD)
                scaler_d.update()
                netD.zero_grad()

            # ── Generator ──
            with torch.amp.autocast("cuda", enabled=use_amp):
                out_fake2 = netD(fake, labels)
                lossG = criterion(out_fake2, label_real)

            scaler_g.scale(lossG / grad_accum).backward()
            if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
                scaler_g.step(optimG)
                scaler_g.update()
                netG.zero_grad()

            if i % 100 == 0:
                logger.info(
                    "[%d/%d][%d/%d]  LossD=%.4f  LossG=%.4f  D(x)=%.3f  D(G(z))=%.3f",
                    epoch, num_epochs, i, len(loader),
                    lossD.item(), lossG.item(),
                    out_real.mean().item(), out_fake2.mean().item(),
                )

        # Epoch-end: generate class-conditional grid
        _save_conditional_grid(netG, fixed_noise, fixed_labels,
                               idx_to_name, output_dir, epoch)

    torch.save(netG.state_dict(), str(output_dir / "cgan_generator.pth"))
    torch.save(netD.state_dict(), str(output_dir / "cgan_discriminator.pth"))
    logger.info("Model weights saved → %s", output_dir)
    return netG, lossG.item(), lossD.item()


# ═════════════════════════════════════════════════════════════
#  Visualisation
# ═════════════════════════════════════════════════════════════

def _save_conditional_grid(
    netG: nn.Module,
    noise: torch.Tensor,
    labels: torch.Tensor,
    idx_to_name: dict,
    output_dir: Path,
    epoch: int,
) -> None:
    """Save a grid where each row is a different class."""
    netG.eval()
    with torch.no_grad():
        fakes = netG(noise, labels).cpu()
    netG.train()

    n_rows = labels.unique().size(0)
    grid = vutils.make_grid(fakes, nrow=8, normalize=True, padding=2)

    fig, ax = plt.subplots(figsize=(14, 2 * n_rows))
    ax.imshow(np.transpose(grid.numpy(), (1, 2, 0)))

    # Build row labels
    unique = labels.unique().cpu().tolist()
    y_labels = [idx_to_name.get(c, f"class_{c}") for c in unique]
    title = "Epoch {} — rows: {}".format(epoch, ", ".join(y_labels))
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    path = output_dir / f"conditional_grid_epoch_{epoch:02d}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Conditional grid saved → %s", path)


# ═════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════

def main() -> None:
    global NUM_EPOCHS, BATCH_SIZE, MAX_IMAGES

    setup_logging()
    args = parse_common_args("Text-to-Image — Conditional GAN on COCO")
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
    logger.info("  Text-to-Image — Conditional GAN on COCO (PyTorch)")
    logger.info("=" * 60)

    try:
        loader, idx_to_name = get_data(paths["data"])
    except Exception as exc:
        logger.warning("Dataset loading failed: %s", exc)
        dataset_missing_metrics(
            paths["outputs"], "COCO 2017",
            ["https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset"],
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
        loader, idx_to_name, device, paths["outputs"], use_amp=use_amp,
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
