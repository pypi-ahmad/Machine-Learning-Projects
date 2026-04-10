"""Ecommerce Item Attribute Tagger — training pipeline.

Trains a multi-head attribute classifier using the Fashion Product
Images dataset.

Usage::

    python train.py
    python train.py --data path/to/dataset --epochs 20
    python train.py --backbone resnet50 --batch 32
    python train.py --force-download
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

_SRC = Path(__file__).resolve().parent
_REPO = _SRC.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_SRC))

from config import TaggerConfig, load_config, ATTRIBUTE_HEADS
from attribute_predictor import MultiHeadAttributeModel
from data_bootstrap import (
    ensure_tagger_dataset,
    find_images_dir,
    ATTRIBUTE_COLUMNS,
)

log = logging.getLogger("attribute_tagger.train")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FashionAttributeDataset(Dataset):
    """PyTorch Dataset loading images + multi-attribute labels."""

    def __init__(
        self,
        images_dir: Path,
        styles_csv: Path,
        label_maps: dict[str, list[str]],
        imgsz: int = 224,
    ) -> None:
        import torchvision.transforms as T

        self.images_dir = images_dir
        self.label_maps = label_maps
        self.imgsz = imgsz

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((imgsz, imgsz)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.val_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((imgsz, imgsz)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self._is_val = False

        # Parse CSV
        self.samples: list[dict] = []
        label_to_idx = {
            col: {lbl: i for i, lbl in enumerate(labels)}
            for col, labels in label_maps.items()
        }

        with open(styles_csv, "r", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                img_id = row.get("id", "").strip()
                if not img_id:
                    continue

                img_path = images_dir / f"{img_id}.jpg"
                if not img_path.exists():
                    continue

                labels = {}
                for col in label_maps:
                    val = row.get(col, "<other>").strip()
                    idx = label_to_idx[col].get(val, label_to_idx[col].get("<other>", 0))
                    labels[col] = idx

                self.samples.append({
                    "path": str(img_path),
                    "labels": labels,
                })

        log.info("Loaded %d samples from %s", len(self.samples), styles_csv)

    def set_val_mode(self, val: bool = True) -> None:
        self._is_val = val

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, int]]:
        sample = self.samples[idx]
        img = cv2.imread(sample["path"])
        if img is None:
            img = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        tfm = self.val_transform if self._is_val else self.transform
        tensor = tfm(img)
        return tensor, sample["labels"]


def _collate(batch):
    """Custom collate for dict labels."""
    images = torch.stack([b[0] for b in batch])
    labels = {}
    keys = batch[0][1].keys()
    for k in keys:
        labels[k] = torch.tensor([b[1][k] for b in batch], dtype=torch.long)
    return images, labels


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(
    data_root: Path,
    cfg: TaggerConfig,
) -> dict:
    """Train the multi-head attribute model.

    Returns
    -------
    dict
        Training summary with metrics and paths.
    """
    # Locate data
    processed = data_root / "processed"
    label_maps_path = processed / "label_maps.json"
    styles_path = processed / "styles_clean.csv"

    if not label_maps_path.exists():
        raise FileNotFoundError(
            f"label_maps.json not found in {processed}. Run data_bootstrap first."
        )

    label_maps = json.loads(label_maps_path.read_text(encoding="utf-8"))
    images_dir = find_images_dir(data_root)

    if images_dir is None:
        raise FileNotFoundError(f"Images directory not found under {data_root}")

    log.info("Images: %s | Labels: %s", images_dir, styles_path)

    # Build dataset
    ds = FashionAttributeDataset(images_dir, styles_path, label_maps, imgsz=cfg.imgsz)

    val_size = int(len(ds) * cfg.val_split)
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=_collate, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=_collate,
    )

    # Build model
    head_sizes = {k: len(v) for k, v in label_maps.items() if k in cfg.attribute_heads}
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = MultiHeadAttributeModel(cfg.backbone, head_sizes)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss()

    save_dir = _SRC / "runs" / "attribute_tagger"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(cfg.epochs):
        # ── Train ──────────────────────────────────────────
        model.train()
        total_loss = 0.0
        n_batches = 0

        for images, labels in train_loader:
            images = images.to(device)
            logits = model(images)

            loss = torch.tensor(0.0, device=device)
            for attr_name in head_sizes:
                if attr_name in logits and attr_name in labels:
                    loss += criterion(logits[attr_name], labels[attr_name].to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # ── Validate ───────────────────────────────────────
        model.eval()
        correct: dict[str, int] = {k: 0 for k in head_sizes}
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                logits = model(images)
                total += images.size(0)

                for attr_name in head_sizes:
                    if attr_name in logits and attr_name in labels:
                        preds = logits[attr_name].argmax(dim=1)
                        correct[attr_name] += (
                            preds == labels[attr_name].to(device)
                        ).sum().item()

        per_head_acc = {k: correct[k] / max(total, 1) for k in head_sizes}
        mean_acc = sum(per_head_acc.values()) / max(len(per_head_acc), 1)

        log.info(
            "Epoch %d/%d — loss=%.4f  mean_acc=%.3f  %s",
            epoch + 1, cfg.epochs, avg_loss, mean_acc,
            " ".join(f"{k}={v:.3f}" for k, v in per_head_acc.items()),
        )

        if mean_acc > best_acc:
            best_acc = mean_acc
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "label_maps": label_maps,
                "backbone": cfg.backbone,
                "head_sizes": head_sizes,
                "epoch": epoch + 1,
                "mean_acc": mean_acc,
                "per_head_acc": per_head_acc,
            }
            torch.save(checkpoint, str(save_dir / "best_model.pt"))
            log.info("  → Saved best model (acc=%.3f)", mean_acc)

    print(f"\nTraining complete! Best mean accuracy: {best_acc:.2%}")
    return {
        "best_acc": best_acc,
        "epochs": cfg.epochs,
        "backbone": cfg.backbone,
        "heads": list(head_sizes.keys()),
        "weights": str(save_dir / "best_model.pt"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Ecommerce Item Attribute Tagger — training",
    )
    parser.add_argument("--data", type=str, default=None,
                        help="Path to dataset root")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML/JSON config file")
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.backbone:
        cfg.backbone = args.backbone
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch:
        cfg.batch_size = args.batch
    if args.lr:
        cfg.lr = args.lr
    if args.device:
        cfg.device = args.device

    if args.data:
        data_root = Path(args.data)
    else:
        data_root = ensure_tagger_dataset(force=args.force_download)

    train_model(data_root, cfg)


if __name__ == "__main__":
    main()
