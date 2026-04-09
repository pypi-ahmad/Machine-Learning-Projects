#!/usr/bin/env python3
"""
Traffic Flow Prediction — METR-LA
==================================
PyTorch LSTM for traffic flow prediction on METR-LA dataset.

Dataset: https://www.kaggle.com/datasets/annnnguyen/metr-la-dataset
Run:     python run.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from shared.utils import (
    download_kaggle_dataset, set_seed, setup_logging, project_paths,
    get_device, save_regression_report,
    parse_common_args, configure_cuda_allocator, save_metrics, run_metadata,
    write_split_manifest, dataset_fingerprint,
    dataset_missing_metrics, resolve_device_from_args,
)

logger = logging.getLogger(__name__)
KAGGLE_SLUG = "annnnguyen/metr-la-dataset"
SEQ_LEN = 12
PRED_LEN = 1
BATCH_SIZE = 64
NUM_EPOCHS = 20
LR = 1e-3


class TrafficLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_layers=2, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def get_data(data_dir: Path):
    ds_path = download_kaggle_dataset(KAGGLE_SLUG, data_dir, dataset_name="METR-LA Traffic")
    csvs = sorted(ds_path.rglob("*.csv")) + sorted(data_dir.rglob("*.csv"))
    h5s = sorted(ds_path.rglob("*.h5")) + sorted(data_dir.rglob("*.h5"))

    if h5s:
        df = pd.read_hdf(h5s[0])
        logger.info("Loaded HDF5: %s — shape %s", h5s[0].name, df.shape)
    elif csvs:
        df = pd.read_csv(csvs[0])
        logger.info("Loaded CSV: %s — shape %s", csvs[0].name, df.shape)
    else:
        raise FileNotFoundError(f"No data files in {ds_path}")
    return df


def create_sequences(data: np.ndarray, seq_len: int, pred_len: int):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len:i + seq_len + pred_len].mean(axis=0))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def main():
    args = parse_common_args("Traffic Flow Prediction — METR-LA LSTM")
    setup_logging()
    set_seed(args.seed, deterministic=True)
    configure_cuda_allocator()
    paths = project_paths(__file__)
    device = resolve_device_from_args(args)

    logger.info("=== Traffic Flow Prediction — METR-LA (LSTM) ===")

    if args.download_only:
        try:
            get_data(paths["data"])
            logger.info("Download complete.")
        except Exception as e:
            logger.error("Download failed: %s", e)
        sys.exit(0)

    try:
        df = get_data(paths["data"])
    except (FileNotFoundError, Exception) as exc:
        logger.error("Dataset error: %s", exc)
        dataset_missing_metrics(paths["outputs"], "METR-LA Traffic", ["https://www.kaggle.com/datasets/annnnguyen/metr-la-dataset"])
        return

    # Use first N sensor columns (numeric only)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 20:
        numeric_cols = numeric_cols[:20]  # limit for speed
    values = df[numeric_cols].values.astype(np.float32)

    # Normalize
    mu, std = values.mean(axis=0, keepdims=True), values.std(axis=0, keepdims=True) + 1e-8
    values_norm = (values - mu) / std

    # Create sequences
    X, y = create_sequences(values_norm, SEQ_LEN, PRED_LEN)
    logger.info("Sequences: X=%s, y=%s", X.shape, y.shape)

    batch_size = args.batch_size or BATCH_SIZE
    num_epochs = args.epochs or NUM_EPOCHS
    use_amp = not args.no_amp

    if args.mode == "smoke":
        num_epochs = 1
        X = X[:500]
        y = y[:500]
        logger.info("SMOKE TEST: 1 epoch, %d sequences", len(X))

    # Split 70/15/15
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

    write_split_manifest(
        paths["outputs"],
        dataset_fp=dataset_fingerprint(paths["data"]),
        split_method="temporal",
        seed=args.seed,
        counts={"train": len(y_train), "val": len(y_val), "test": len(y_test)},
    )

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Model
    input_dim = X.shape[2]
    output_dim = y.shape[1] if y.ndim > 1 else 1
    model = TrafficLSTM(input_dim, hidden_dim=128, output_dim=output_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and use_amp))

    # Train
    model.train()
    for epoch in range(1, num_epochs + 1):
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and use_amp)):
                pred = model(xb)
                loss = criterion(pred, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * xb.size(0)
        logger.info("Epoch %d/%d — MSE=%.6f", epoch, num_epochs, total_loss / len(train_ds))

    # Evaluate
    model.eval()
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            all_preds.append(pred)
            all_true.append(yb.numpy())

    y_pred = np.concatenate(all_preds).ravel()
    y_true = np.concatenate(all_true).ravel()

    metrics = save_regression_report(y_true, y_pred, paths["outputs"], prefix="metr_la")
    torch.save(model.state_dict(), paths["outputs"] / "traffic_lstm.pth")

    # Write standardized metrics
    metrics["run_metadata"] = run_metadata(args)
    save_metrics(paths["outputs"], metrics, task_type="regression", mode=args.mode)
    logger.info("Done! Metrics: %s", metrics)


if __name__ == "__main__":
    main()
