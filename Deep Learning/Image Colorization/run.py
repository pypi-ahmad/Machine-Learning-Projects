#!/usr/bin/env python3
"""Project 44 -- Image Colorization

Dataset : VizWiz Colorization
Model   : Simple UNet autoencoder (grayscale -> color)

Usage:
    python run.py
    python run.py --smoke-test
    python run.py --epochs 10 --batch-size 8
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from shared.utils import (
    seed_everything, get_device, dataset_prompt, kaggle_download, ensure_dir,
    save_metrics, parse_common_args, load_profile, resolve_config,
    write_split_manifest, EarlyStopping)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / "data"
OUTPUT_DIR  = PROJECT_DIR / "outputs"
KAGGLE = "landrykezebou/vizwiz-colorization"
EPOCHS, LR, BATCH = 20, 1e-3, 16
IMG_SIZE = 128


class ColorDS(Dataset):
    def __init__(self, paths, size=128):
        self.paths, self.size = paths, size
        self.tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        color = self.tf(Image.open(self.paths[i]).convert("RGB"))
        gray  = color.mean(0, keepdim=True)
        return gray, color


class MiniUNet(nn.Module):
    def __init__(self):
        super().__init__()
        def block(ci, co): return nn.Sequential(nn.Conv2d(ci, co, 3, 1, 1), nn.BatchNorm2d(co), nn.ReLU())
        self.enc1 = block(1, 64)
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), block(64, 128))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), block(128, 256))
        self.up2  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = block(256, 128)
        self.up1  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = block(128, 64)
        self.head = nn.Conv2d(64, 3, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d2 = self.dec2(torch.cat([self.up2(e3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return torch.sigmoid(self.head(d1))



TASK_TYPE = 'custom_cv'

def get_data(max_imgs=5000):
    dataset_prompt("VizWiz Colorization",
                   ["https://www.kaggle.com/datasets/landrykezebou/vizwiz-colorization"])
    imgs = list(DATA_DIR.rglob("*.jpg")) + list(DATA_DIR.rglob("*.png"))
    if not imgs:
        kaggle_download(KAGGLE, DATA_DIR)
        imgs = list(DATA_DIR.rglob("*.jpg")) + list(DATA_DIR.rglob("*.png"))
    return ColorDS(imgs[:max_imgs])


def main():
    args = parse_common_args()
    profile = load_profile(args.profile)
    cfg = resolve_config(args, profile, TASK_TYPE)
    seed_everything(cfg.get('seed', 42))
    device = get_device(args.device)
    ensure_dir(OUTPUT_DIR)

    max_imgs = 100 if args.smoke_test else 5000
    ds = get_data(max_imgs=max_imgs)
    if args.download_only:
        print("  Dataset ready. Exiting (--download-only).")
        return

    is_full     = (args.mode == 'full')
    epochs      = args.epochs or cfg.get('epochs', EPOCHS)
    batch_size  = args.batch_size or cfg.get('batch_size', BATCH)
    num_workers = args.num_workers if args.num_workers is not None else cfg.get('num_workers', 2)
    use_amp     = not args.no_amp and cfg.get('amp', True)
    max_batches = 2 if args.smoke_test else None
    es_on       = cfg.get('early_stopping', False) if is_full else False
    es_patience = cfg.get('patience', 3)
    if args.smoke_test:
        epochs = 1

    n_val = int(0.15 * len(ds))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers)
    val_dl   = DataLoader(val_ds, batch_size, num_workers=num_workers)

    model = MiniUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()
    best_loss = float("inf")
    es = EarlyStopping(patience=es_patience, mode='min') if es_on else None

    for ep in range(epochs):
        model.train(); total_loss = 0
        for bi, (gray, color) in enumerate(tqdm(train_dl, desc=f"Epoch {ep+1}/{epochs}")):
            if max_batches and bi >= max_batches:
                break
            gray, color = gray.to(device), color.to(device)
            pred = model(gray)
            loss = crit(pred, color)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        # val
        model.eval(); vloss = 0; vcount = 0
        with torch.no_grad():
            for bi, (gray, color) in enumerate(val_dl):
                if max_batches and bi >= max_batches:
                    break
                gray, color = gray.to(device), color.to(device)
                vloss += crit(model(gray), color).item()
                vcount += 1
        vloss /= max(vcount, 1)
        print(f"  train_loss={total_loss / max(bi + 1, 1):.4f}  val_loss={vloss:.4f}")
        if vloss < best_loss:
            best_loss = vloss
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")
        if es and es.step(vloss):
            break

    save_metrics({"best_val_mse": best_loss}, OUTPUT_DIR)
    if is_full:
        write_split_manifest(OUTPUT_DIR, dataset_name=KAGGLE,
            seed=cfg.get('seed', 42))


if __name__ == "__main__":
    from shared.utils import guarded_main
    guarded_main(main, OUTPUT_DIR)
