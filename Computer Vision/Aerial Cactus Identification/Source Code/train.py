"""Train Aerial Cactus Identification — classification (binary).

Usage::

    python train.py
    python train.py --data path/to/dataset --epochs 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.datasets import DatasetResolver
from train.train_classification import train_classification


def _create_synthetic_dataset() -> Path:
    """Create a tiny synthetic binary classification dataset as fallback."""
    import cv2
    import numpy as np

    base = Path(__file__).resolve().parents[2] / "data" / "aerial_cactus" / "synthetic"
    if (base / ".ready").exists():
        return base

    rng = np.random.RandomState(42)
    for cls_name, color_range in [("cactus", (40, 120)), ("no_cactus", (160, 220))]:
        cls_dir = base / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(50):
            img = rng.randint(color_range[0], color_range[1], (32, 32, 3), dtype=np.uint8)
            cv2.imwrite(str(cls_dir / f"{i:04d}.jpg"), img)

    (base / ".ready").touch()
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Aerial Cactus Identification (classification)")
    parser.add_argument("--data", type=str, default=None, help="Path to image-folder dataset")
    parser.add_argument("--model", type=str, default="resnet18", help="resnet18|resnet50|efficientnet_b0|mobilenet_v2")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--force-download", action="store_true", help="Force re-download dataset")
    args = parser.parse_args()

    if args.data is None:
        try:
            data_path = DatasetResolver().resolve("aerial_cactus", force=args.force_download)
            data_dir = str(data_path)
        except RuntimeError:
            print("[WARN] Kaggle download failed; generating synthetic dataset")
            data_dir = str(_create_synthetic_dataset())
        print(f"[INFO] Using dataset at {data_dir}")
    else:
        data_dir = args.data

    train_classification(
        data_dir=data_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        lr=args.lr,
        device=args.device,
        save_path=str(Path(__file__).parent / "runs" / "aerial_cactus_cls" / "best_model.pt"),
        registry_project="aerial_cactus",
    )


if __name__ == "__main__":
    main()
