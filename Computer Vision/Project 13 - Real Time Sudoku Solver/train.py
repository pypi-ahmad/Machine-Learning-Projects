"""
Train — P13 Sudoku Digit Classifier
======================================
Train a digit classifier (0-9) on MNIST using ResNet-18 transfer learning.

The Sudoku Solver project needs a digit recognition model.  This script
downloads MNIST via ``torchvision.datasets``, restructures it into an
ImageFolder layout, and fine-tunes a lightweight classifier.

Usage::

    python train.py                          # Default: 15 epochs
    python train.py --epochs 30 --model resnet18
    python train.py --model mobilenet_v3_small --epochs 20
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.train_classification import train_classification

logger = logging.getLogger(__name__)

MNIST_DIR = ROOT / "mnist_imagefolder"


def prepare_mnist_imagefolder(output_dir: Path = MNIST_DIR) -> Path:
    """Download MNIST and restructure into ImageFolder layout.

    Creates::

        output_dir/
            train/
                0/ 1/ 2/ ... 9/
            val/
                0/ 1/ 2/ ... 9/

    Returns the output_dir path.
    """
    from torchvision import datasets as tv_datasets

    if (output_dir / "train" / "0").exists():
        logger.info("MNIST ImageFolder already exists at %s", output_dir)
        return output_dir

    logger.info("Downloading & preparing MNIST...")

    from PIL import Image

    for split_name, is_train in [("train", True), ("val", False)]:
        ds = tv_datasets.MNIST(
            root=str(output_dir / "_raw"),
            train=is_train,
            download=True,
        )
        for idx in range(len(ds)):
            img_tensor, label = ds[idx]
            cls_dir = output_dir / split_name / str(label)
            cls_dir.mkdir(parents=True, exist_ok=True)
            img_path = cls_dir / f"{idx:06d}.png"
            if not img_path.exists():
                if isinstance(img_tensor, Image.Image):
                    img_tensor.save(str(img_path))
                else:
                    # numpy array
                    arr = np.array(img_tensor)
                    Image.fromarray(arr).save(str(img_path))

    logger.info("MNIST ImageFolder ready at %s", output_dir)
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="P13 Sudoku — Digit Classifier Training (MNIST)",
    )
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--device", default=None)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )

    # Prepare MNIST as ImageFolder
    data_dir = prepare_mnist_imagefolder()

    # Train
    logger.info("Training digit classifier (10 classes)...")
    result = train_classification(
        data_dir=data_dir,
        num_classes=10,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=args.img_size,
        device=args.device,
        output_dir=str(ROOT / "runs" / "classify" / "digits"),
        patience=args.patience,
        workers=args.workers,
        registry_project="sudoku_digits",
    )
    logger.info(
        "Done! best_acc=%.4f @ epoch %d  →  %s",
        result["best_acc"], result["best_epoch"], result["model_path"],
    )


if __name__ == "__main__":
    main()
