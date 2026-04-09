#!/usr/bin/env python3
"""Project 48 -- Fashion MNIST Clothing Classification

Dataset : Fashion-MNIST (auto-downloaded via torchvision)
Model   : efficientnet_b0 (timm, pretrained)

Usage:
    python run.py
    python run.py --smoke-test
    python run.py --epochs 5 --batch-size 128
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from shared.utils import (
    seed_everything, get_device, ensure_dir, parse_common_args, load_profile,
    resolve_config, write_split_manifest, EarlyStopping, run_with_oom_backoff)
from shared.cv import build_timm_model, train_model, evaluate_model

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / "data"
OUTPUT_DIR  = PROJECT_DIR / "outputs"

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
MODEL   = "efficientnet_b0.ra_in1k"
EPOCHS  = 10
BATCH   = 64



TASK_TYPE = 'cv'

def get_data(batch_size=64, num_workers=4):
    tf = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485] * 3, [0.229] * 3),
    ])
    train_full = datasets.FashionMNIST(str(DATA_DIR), train=True, download=True, transform=tf)
    test_ds    = datasets.FashionMNIST(str(DATA_DIR), train=False, download=True, transform=tf)
    n_val = int(0.15 * len(train_full))
    train_ds, val_ds = random_split(train_full, [len(train_full) - n_val, n_val])
    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=torch.cuda.is_available())
    return (DataLoader(train_ds, shuffle=True, **kw),
            DataLoader(val_ds, **kw),
            DataLoader(test_ds, **kw), CLASSES)


def main():
    args = parse_common_args()
    profile = load_profile(args.profile)
    cfg = resolve_config(args, profile, TASK_TYPE)
    seed_everything(cfg.get('seed', 42))
    device = get_device(args.device)
    ensure_dir(OUTPUT_DIR)

    is_full    = (args.mode == 'full')
    epochs     = args.epochs or cfg.get('epochs', EPOCHS)
    batch_size = args.batch_size or cfg.get('batch_size', BATCH)
    num_workers = args.num_workers if args.num_workers is not None else cfg.get('num_workers', 2)
    use_amp    = not args.no_amp and cfg.get('amp', True)
    img_size   = cfg.get('img_size', 224)
    max_batches = 2 if args.smoke_test else None
    grad_accum  = cfg.get('grad_accum_steps', 1) if is_full else 1
    freeze_bb   = cfg.get('freeze_backbone_epochs', 0) if is_full else 0
    es_on       = cfg.get('early_stopping', False) if is_full else False
    es_patience = cfg.get('patience', 3)
    if args.smoke_test:
        epochs = 1

    train_dl, val_dl, test_dl, classes = get_data(
        batch_size=batch_size, num_workers=num_workers,
    )

    if args.download_only:
        print("  Dataset ready. Exiting (--download-only).")
        return

    model = build_timm_model(MODEL, num_classes=10)
    model = train_model(model, train_dl, val_dl, epochs=epochs,
                        lr=1e-4, device=device, output_dir=OUTPUT_DIR,
                        use_amp=use_amp, max_batches=max_batches)
    evaluate_model(model, test_dl, classes, device=device,
                   output_dir=OUTPUT_DIR, max_batches=max_batches)


if __name__ == "__main__":
    from shared.utils import guarded_main
    guarded_main(main, OUTPUT_DIR)
