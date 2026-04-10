"""
Train — P41 Image Segmentation (YOLO-Seg / DeepLabV3)
========================================================
Fine-tune a segmentation model.

Supports two backends:
    - ``yolo``: YOLO26-Seg (instance segmentation)
    - ``deeplab``: DeepLabV3 (semantic segmentation)

Usage::

    # YOLO-Seg
    python train.py yolo --data path/to/data.yaml

    # DeepLabV3
    python train.py deeplab --data path/to/seg_dataset --num-classes 21
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.train_segmentation import (
    train_segmentation_deeplabv3,
    train_segmentation_yolo,
)


def main():
    parser = argparse.ArgumentParser(description="P41 Segmentation Training")
    sub = parser.add_subparsers(dest="backend", required=True)

    # YOLO-Seg
    yp = sub.add_parser("yolo", help="YOLO instance segmentation")
    yp.add_argument("--data", required=True, help="Path to data.yaml")
    yp.add_argument("--model", default="yolo26n-seg.pt")
    yp.add_argument("--epochs", type=int, default=50)
    yp.add_argument("--imgsz", type=int, default=640)
    yp.add_argument("--batch", type=int, default=16)
    yp.add_argument("--device", default=None)
    yp.add_argument("--patience", type=int, default=10)
    yp.add_argument("--lr0", type=float, default=0.01)
    yp.add_argument("--workers", type=int, default=4)
    yp.add_argument("--resume", action="store_true")

    # DeepLabV3
    dp = sub.add_parser("deeplab", help="DeepLabV3 semantic segmentation")
    dp.add_argument("--data", required=True, help="Dataset root (images/ + masks/)")
    dp.add_argument("--num-classes", type=int, default=21)
    dp.add_argument("--backbone", default="resnet50", choices=["resnet50", "resnet101"])
    dp.add_argument("--epochs", type=int, default=30)
    dp.add_argument("--batch-size", type=int, default=8)
    dp.add_argument("--lr", type=float, default=1e-3)
    dp.add_argument("--img-size", type=int, default=512)
    dp.add_argument("--device", default=None)
    dp.add_argument("--patience", type=int, default=7)
    dp.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    if args.backend == "yolo":
        train_segmentation_yolo(
            data_yaml=args.data, model=args.model, epochs=args.epochs,
            imgsz=args.imgsz, batch=args.batch, device=args.device,
            project=str(ROOT / "runs" / "segment"), name="yolo_seg",
            patience=args.patience, lr0=args.lr0, workers=args.workers,
            resume=args.resume,
            registry_project="image_segmentation",
        )
    else:
        train_segmentation_deeplabv3(
            data_dir=args.data, num_classes=args.num_classes,
            backbone=args.backbone, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr,
            img_size=args.img_size, device=args.device,
            output_dir=str(ROOT / "runs" / "segment_deeplab"),
            patience=args.patience, workers=args.workers,
            registry_project="image_segmentation",
        )


if __name__ == "__main__":
    main()
