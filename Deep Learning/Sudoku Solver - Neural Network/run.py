#!/usr/bin/env python3
"""Project 23 -- Sudoku Solver with Neural Network

Dataset : 1M Sudoku Games
Model   : Custom CNN (PyTorch)

Usage:
    python run.py
    python run.py --smoke-test
    python run.py --epochs 5 --batch-size 128
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd
from shared.utils import (
    seed_everything, get_device, dataset_prompt, kaggle_download, ensure_dir,
    save_metrics, parse_common_args, load_profile, resolve_config,
    write_split_manifest, EarlyStopping)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / "data"
OUTPUT_DIR  = PROJECT_DIR / "outputs"
KAGGLE = "bryanpark/sudoku"
EPOCHS, LR, BATCH = 10, 1e-3, 256


class SudokuNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.head = nn.Conv2d(128, 9, 1)  # 9 classes per cell

    def forward(self, x):  # x: (B,1,9,9)
        return self.head(self.conv(x))  # (B,9,9,9)


def parse_sudoku(s):
    return np.array([int(c) for c in s]).reshape(9, 9).astype(np.float32)



TASK_TYPE = 'custom_special'

def get_data(nrows=100_000):
    dataset_prompt("1M Sudoku Games", ["https://www.kaggle.com/datasets/bryanpark/sudoku"])
    csvs = list(DATA_DIR.rglob("*.csv"))
    if not csvs:
        kaggle_download(KAGGLE, DATA_DIR)
        csvs = list(DATA_DIR.rglob("*.csv"))
    df = pd.read_csv(csvs[0], nrows=nrows)
    quizzes = np.stack([parse_sudoku(q) for q in df.iloc[:, 0]])
    solutions = np.stack([parse_sudoku(s) for s in df.iloc[:, 1]])
    X = torch.from_numpy(quizzes / 9.0).unsqueeze(1)  # (N,1,9,9)
    y = torch.from_numpy(solutions).long() - 1        # classes 0-8
    n = len(X); split = int(0.9 * n)
    return (TensorDataset(X[:split], y[:split]),
            TensorDataset(X[split:], y[split:]))


def main():
    args = parse_common_args()
    profile = load_profile(args.profile)
    cfg = resolve_config(args, profile, TASK_TYPE)
    seed_everything(cfg.get('seed', 42))
    device = get_device(args.device)
    ensure_dir(OUTPUT_DIR)

    nrows = 1000 if args.smoke_test else 100_000
    train_ds, val_ds = get_data(nrows=nrows)
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

    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers)
    val_dl   = DataLoader(val_ds, batch_size, num_workers=num_workers)

    model = SudokuNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    es = EarlyStopping(patience=es_patience, mode='max') if es_on else None

    best_acc = 0.0

    for ep in range(epochs):
        model.train(); total, correct = 0, 0
        for bi, (X, y) in enumerate(tqdm(train_dl, desc=f"Epoch {ep+1}/{epochs}")):
            if max_batches and bi >= max_batches:
                break
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = crit(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            correct += (out.argmax(1) == y).sum().item()
            total += y.numel()
        # val
        model.eval(); vc, vt = 0, 0
        with torch.no_grad():
            for bi, (X, y) in enumerate(val_dl):
                if max_batches and bi >= max_batches:
                    break
                X, y = X.to(device), y.to(device)
                pred = model(X).argmax(1)
                vc += (pred == y).sum().item(); vt += y.numel()
        vacc = vc / max(vt, 1)
        print(f"  train_acc={correct / max(total, 1):.4f}  val_acc={vacc:.4f}")
        if vacc > best_acc:
            best_acc = vacc
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")
        if es and es.step(mean_dice if 'mean_dice' in dir() else vacc if 'vacc' in dir() else acc):
            break

    save_metrics({"best_cell_accuracy": best_acc}, OUTPUT_DIR)
    if is_full:
        write_split_manifest(OUTPUT_DIR, dataset_name=KAGGLE,
            seed=cfg.get('seed', 42))


if __name__ == "__main__":
    from shared.utils import guarded_main
    guarded_main(main, OUTPUT_DIR)
