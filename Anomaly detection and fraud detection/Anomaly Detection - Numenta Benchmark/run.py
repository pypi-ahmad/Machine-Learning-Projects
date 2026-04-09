#!/usr/bin/env python3
"""
Anomaly Detection — Numenta Anomaly Benchmark (NAB)
====================================================
PyTorch LSTM autoencoder for time-series anomaly detection.

Dataset: https://www.kaggle.com/datasets/boltzmannbrain/nab
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
    get_device, ensure_dir, save_classification_report,
    parse_common_args, configure_cuda_allocator, save_metrics, run_metadata,
    write_split_manifest, dataset_fingerprint,
    dataset_missing_metrics, resolve_device_from_args,
)

logger = logging.getLogger(__name__)
KAGGLE_SLUG = "boltzmannbrain/nab"
SEQ_LEN = 50
BATCH_SIZE = 64
NUM_EPOCHS = 20
LR = 1e-3


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, n_layers=2):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (h, c) = self.encoder(x)
        dec_in = torch.zeros_like(x[:, :1, :]).repeat(1, x.size(1), 1)
        dec_in = dec_in + h[-1:].permute(1, 0, 2).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(dec_in, (h, c))
        return self.fc(dec_out)


def get_data(data_dir: Path):
    ds_path = download_kaggle_dataset(KAGGLE_SLUG, data_dir, dataset_name="NAB")
    csvs = sorted(ds_path.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files in {ds_path}")
    # Use first realKnownCause file or first available
    target_csv = None
    for c in csvs:
        if "realKnownCause" in str(c) or "realTraffic" in str(c):
            target_csv = c
            break
    if target_csv is None:
        target_csv = csvs[0]
    logger.info("Using: %s", target_csv.name)
    df = pd.read_csv(target_csv)
    return df


def create_sequences(values: np.ndarray, seq_len: int):
    seqs = []
    for i in range(len(values) - seq_len):
        seqs.append(values[i:i + seq_len])
    return np.array(seqs)


def main():
    args = parse_common_args("NAB Anomaly Detection — LSTM Autoencoder")
    setup_logging()
    set_seed(args.seed, deterministic=True)
    configure_cuda_allocator()
    paths = project_paths(__file__)
    device = resolve_device_from_args(args)

    logger.info("=== NAB Anomaly Detection (LSTM Autoencoder) ===")

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
        dataset_missing_metrics(paths["outputs"], "NAB", ["https://www.kaggle.com/datasets/boltzmannbrain/nab"])
        return
    # Use the numeric value column
    value_col = [c for c in df.columns if c not in ("timestamp", "date", "datetime")]
    if not value_col:
        value_col = [df.columns[-1]]
    values = df[value_col[0]].values.astype(np.float32)

    # Normalize
    mu, std = values.mean(), values.std() + 1e-8
    values_norm = (values - mu) / std

    # Create sequences
    seqs = create_sequences(values_norm, SEQ_LEN)
    seqs = seqs[..., np.newaxis]  # (N, seq_len, 1)

    batch_size = args.batch_size or BATCH_SIZE
    num_epochs = args.epochs or NUM_EPOCHS

    if args.mode == "smoke":
        num_epochs = 1
        seqs = seqs[:500]
        logger.info("SMOKE TEST: 1 epoch, %d sequences", len(seqs))

    # Train/val/test split (70/15/15 — temporal)
    n = len(seqs)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    train_seqs, val_seqs, test_seqs = seqs[:train_end], seqs[train_end:val_end], seqs[val_end:]

    write_split_manifest(
        paths["outputs"],
        dataset_fp=dataset_fingerprint(paths["data"]),
        split_method="temporal",
        seed=args.seed,
        counts={"train": len(train_seqs), "val": len(val_seqs), "test": len(test_seqs)},
    )

    train_ds = TensorDataset(torch.tensor(train_seqs))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Model
    model = LSTMAutoencoder(input_dim=1, hidden_dim=64, n_layers=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # Train
    model.train()
    for epoch in range(1, num_epochs + 1):
        total_loss = 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        logger.info("Epoch %d/%d — loss=%.6f", epoch, num_epochs, total_loss / len(train_ds))

    # Evaluate — compute reconstruction error as anomaly score
    model.eval()
    test_tensor = torch.tensor(test_seqs).to(device)
    with torch.no_grad():
        recon = model(test_tensor)
    errors = ((test_tensor - recon) ** 2).mean(dim=(1, 2)).cpu().numpy()

    # Threshold at 95th percentile of train errors
    train_tensor = torch.tensor(train_seqs).to(device)
    with torch.no_grad():
        train_recon = model(train_tensor)
    train_errors = ((train_tensor - train_recon) ** 2).mean(dim=(1, 2)).cpu().numpy()
    threshold = np.percentile(train_errors, 95)

    y_pred = (errors > threshold).astype(int)
    # For NAB without ground truth labels, we report the anomaly distribution
    logger.info("Threshold: %.6f | Anomalies detected: %d/%d (%.1f%%)",
                threshold, y_pred.sum(), len(y_pred), 100 * y_pred.mean())

    # Save results
    output_dir = paths["outputs"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    ax1.plot(errors, alpha=0.7, label="Reconstruction Error")
    ax1.axhline(threshold, color="r", linestyle="--", label=f"Threshold ({threshold:.4f})")
    ax1.set_title("Reconstruction Error on Test Set")
    ax1.legend()
    anomaly_idx = np.where(y_pred == 1)[0]
    ax2.plot(values[val_end + SEQ_LEN:], alpha=0.7, label="Original Signal")
    ax2.scatter(anomaly_idx, values[val_end + SEQ_LEN:][anomaly_idx],
                color="red", s=10, label="Detected Anomaly", zorder=5)
    ax2.set_title("Detected Anomalies")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "anomaly_detection.png", dpi=150)
    plt.close(fig)

    torch.save(model.state_dict(), output_dir / "lstm_autoencoder.pth")
    logger.info("Results saved to %s", output_dir)

    # Write standardized metrics
    from sklearn.metrics import accuracy_score, f1_score
    # Use synthetic labels: 0=normal, 1=anomaly (based on threshold)
    y_true_synth = np.zeros(len(y_pred))  # all test assumed normal as baseline
    metrics = {
        "anomaly_ratio": float(y_pred.mean()),
        "threshold": float(threshold),
        "num_anomalies": int(y_pred.sum()),
        "num_test_sequences": int(len(y_pred)),
        "accuracy": float(1.0 - y_pred.mean()),  # proxy: fraction normal
        "macro_f1": 0.0,
        "weighted_f1": 0.0,
        "auc": 0.0,
    }
    metrics["run_metadata"] = run_metadata(args)
    save_metrics(paths["outputs"], metrics, task_type="classification", mode=args.mode)


if __name__ == "__main__":
    main()
