#!/usr/bin/env python3
"""Project 16 -- Brain MRI Segmentation

Dataset : LGG MRI Segmentation
Model   : DeepLabV3-ResNet50 (torchvision)
Task    : Binary Segmentation

Usage:
    python run.py
    python run.py --smoke-test
    python run.py --epochs 10 --batch-size 4 --device cuda
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
from tqdm import tqdm
from shared.utils import (
    seed_everything, get_device, dataset_prompt, kaggle_download, ensure_dir,
    save_metrics, parse_common_args, load_profile, resolve_config,
    write_split_manifest, EarlyStopping)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / "data"
OUTPUT_DIR  = PROJECT_DIR / "outputs"
KAGGLE = "mateuszbuda/lgg-mri-segmentation"
EPOCHS, LR, BATCH = 20, 1e-4, 8


class MRIDataset(Dataset):
    def __init__(self, images, masks, size=256):
        self.images, self.masks, self.size = images, masks, size
        self.tf = transforms.Compose([transforms.Resize((size, size)),
                                      transforms.ToTensor()])
    def __len__(self): return len(self.images)
    def __getitem__(self, i):
        img = self.tf(Image.open(self.images[i]).convert("RGB"))
        msk = self.tf(Image.open(self.masks[i]).convert("L"))
        return img, (msk > 0.5).float()



TASK_TYPE = 'custom_cv'

def get_data():
    dataset_prompt("LGG MRI Segmentation",
                   ["https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation"])
    if not list(DATA_DIR.rglob("*_mask.tif")):
        kaggle_download(KAGGLE, DATA_DIR)
    imgs = sorted(DATA_DIR.rglob("*[!_mask].tif"))
    msks = sorted(DATA_DIR.rglob("*_mask.tif"))
    if not imgs:
        imgs = sorted(p for p in DATA_DIR.rglob("*.png") if "_mask" not in p.stem)
        msks = sorted(DATA_DIR.rglob("*_mask.png"))
    return imgs, msks


def dice_score(pred, target, eps=1e-7):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    return (2 * inter + eps) / (pred.sum() + target.sum() + eps)


def main():
    args = parse_common_args()
    profile = load_profile(args.profile)
    cfg = resolve_config(args, profile, TASK_TYPE)
    seed_everything(cfg.get('seed', 42))
    device = get_device(args.device)
    ensure_dir(OUTPUT_DIR)

    imgs, msks = get_data()
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

    ds = MRIDataset(imgs, msks)
    n_val = int(0.15 * len(ds))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

    model = deeplabv3_resnet50(weights="DEFAULT")
    model.classifier[4] = nn.Conv2d(256, 1, 1)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    es = EarlyStopping(patience=es_patience, mode='max') if es_on else None

    best_dice = 0.0

    for ep in range(epochs):
        model.train(); loss_sum = 0
        for bi, (X, y) in enumerate(tqdm(train_dl, desc=f"Epoch {ep+1}/{epochs}")):
            if max_batches and bi >= max_batches:
                break
            X, y = X.to(device), y.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                out = model(X)["out"]
                out = nn.functional.interpolate(out, size=y.shape[-2:], mode="bilinear")
                loss = bce(out, y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item()
        # val
        model.eval(); dsc = []
        with torch.no_grad():
            for bi, (X, y) in enumerate(val_dl):
                if max_batches and bi >= max_batches:
                    break
                X, y = X.to(device), y.to(device)
                out = torch.sigmoid(model(X)["out"])
                out = nn.functional.interpolate(out, size=y.shape[-2:], mode="bilinear")
                dsc.append(dice_score(out, y).item())
        mean_dice = np.mean(dsc) if dsc else 0.0
        print(f"  loss={loss_sum / max(bi + 1, 1):.4f}  dice={mean_dice:.4f}")
        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")
        if es and es.step(mean_dice if 'mean_dice' in dir() else vacc if 'vacc' in dir() else acc):
            break

    save_metrics({"best_dice": best_dice}, OUTPUT_DIR)
    if is_full:
        write_split_manifest(OUTPUT_DIR, dataset_name=KAGGLE,
            seed=cfg.get('seed', 42))


if __name__ == "__main__":
    from shared.utils import guarded_main
    guarded_main(main, OUTPUT_DIR)
