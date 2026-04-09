#!/usr/bin/env python3
"""
Anomaly Detection in Images — CIFAR-10
=======================================
Uses a timm pretrained model fine-tuned on normal CIFAR-10 images,
then flags images with high reconstruction error / low confidence as anomalies.

Approach: Train a classifier on selected "normal" classes, treat others as anomaly.

Run:  python run.py
"""

import sys
from pathlib import Path

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from shared.utils import (
    dataset_prompt,
    get_device,
    set_seed,
    ensure_dir,
    save_classification_report,
    setup_logging,
    project_paths,
    parse_common_args,
    configure_cuda_allocator,
    save_metrics,
    run_metadata,
    write_split_manifest,
    dataset_fingerprint,
    dataset_missing_metrics,
    resolve_device_from_args,
)

logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────
NORMAL_CLASSES = [0, 1, 2, 3, 4]   # airplane, automobile, bird, cat, deer
ANOMALY_CLASSES = [5, 6, 7, 8, 9]  # dog, frog, horse, ship, truck
BATCH_SIZE = 64
NUM_EPOCHS = 5
LR = 1e-3
IMG_SIZE = 32


def get_data(data_dir: Path):
    """Download CIFAR-10 via torchvision and prepare normal/anomaly splits."""
    dataset_prompt(
        "CIFAR-10",
        ["https://www.cs.toronto.edu/~kriz/cifar.html"],
        notes="Auto-downloaded by torchvision.",
    )

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_full = datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=transform)
    test_full = datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=transform)

    # Training: only normal classes (binary label=0)
    normal_train_idx = [i for i, (_, y) in enumerate(train_full) if y in NORMAL_CLASSES]
    train_set = Subset(train_full, normal_train_idx)

    return train_set, test_full


def build_model(num_classes: int = 2, device: torch.device = torch.device("cpu")):
    """Build a small timm classifier for anomaly scoring."""
    import timm

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=True,
        num_classes=num_classes,
        in_chans=3,
    )
    return model.to(device)


def train(model, train_loader, device, num_epochs=NUM_EPOCHS, lr=LR, use_amp=True):
    """Train on normal-class images only (all label=0)."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and use_amp))

    model.train()
    for epoch in range(1, num_epochs + 1):
        total_loss, correct, total = 0.0, 0, 0
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            labels = torch.zeros(imgs.size(0), dtype=torch.long, device=device)  # all normal

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and use_amp)):
                logits = model(imgs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)

        logger.info(
            "Epoch %d/%d — loss=%.4f  acc=%.4f",
            epoch, num_epochs, total_loss / total, correct / total,
        )
    return model


def evaluate(model, test_full, device, output_dir):
    """Evaluate: normal classes → label 0, anomaly classes → label 1."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    loader = DataLoader(test_full, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            # Ground truth: 0=normal, 1=anomaly
            binary_labels = torch.tensor(
                [0 if t.item() in NORMAL_CLASSES else 1 for t in targets],
                dtype=torch.long,
            )
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()

            all_labels.extend(binary_labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs[:, 1] if probs.shape[1] > 1 else probs[:, 0])

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    return save_classification_report(
        y_true, y_pred, output_dir,
        y_prob=y_prob,
        labels=["normal", "anomaly"],
        prefix="cifar10_anomaly",
    )


def main():
    args = parse_common_args("Anomaly Detection in Images — CIFAR-10")
    setup_logging()
    set_seed(args.seed, deterministic=True)
    configure_cuda_allocator()
    paths = project_paths(__file__)
    device = resolve_device_from_args(args)

    logger.info("=== Anomaly Detection in Images — CIFAR-10 ===")

    if args.download_only:
        try:
            get_data(paths["data"])
            logger.info("Download complete.")
        except Exception as e:
            logger.error("Download failed: %s", e)
        sys.exit(0)

    # Data
    try:
        train_set, test_full = get_data(paths["data"])
    except (FileNotFoundError, Exception) as exc:
        logger.error("Dataset error: %s", exc)
        dataset_missing_metrics(paths["outputs"], "CIFAR-10", ["https://www.cs.toronto.edu/~kriz/cifar.html"])
        return

    batch_size = args.batch_size or BATCH_SIZE
    num_epochs = args.epochs or NUM_EPOCHS
    use_amp = not args.no_amp

    if args.mode == "smoke":
        num_epochs = 1
        train_set = Subset(train_set.dataset, train_set.indices[:500])
        test_indices = list(range(min(500, len(test_full))))
        test_full = Subset(test_full, test_indices)
        logger.info("SMOKE TEST: 1 epoch, %d train, %d test", len(train_set), len(test_full))

    # ── Split manifest ──
    write_split_manifest(
        paths["outputs"],
        dataset_fp=dataset_fingerprint(paths["data"]),
        split_method="class_based_anomaly",
        seed=args.seed,
        counts={"train_normal": len(train_set), "test_full": len(test_full)},
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=device.type == "cuda",
    )

    # Model
    model = build_model(num_classes=2, device=device)

    # Train
    model = train(model, train_loader, device, num_epochs=num_epochs, use_amp=use_amp)

    # Evaluate
    metrics = evaluate(model, test_full, device, paths["outputs"])

    # Save model
    torch.save(model.state_dict(), paths["outputs"] / "model.pth")
    logger.info("Model saved to %s", paths["outputs"] / "model.pth")

    # Write standardized metrics
    metrics["run_metadata"] = run_metadata(args)
    save_metrics(paths["outputs"], metrics, task_type="classification", mode=args.mode)
    logger.info("Done! Metrics: %s", {k: v for k, v in metrics.items() if k != "classification_report"})


if __name__ == "__main__":
    main()
