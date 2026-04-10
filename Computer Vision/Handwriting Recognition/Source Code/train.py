"""Train Handwriting Recognition — classification.

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Handwriting Recognition (classification)")
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
        data_path = DatasetResolver().resolve("handwriting_recognition", force=args.force_download)
        data_dir = str(data_path)
        print(f"[INFO] Resolved dataset → {data_path}")
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
        save_path=str(Path(__file__).parent / "runs" / "handwriting_cls" / "best_model.pt"),
        registry_project="handwriting_recognition",
    )


if __name__ == "__main__":
    main()
