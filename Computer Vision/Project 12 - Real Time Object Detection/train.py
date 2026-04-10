"""
Train — P12 Object Detection (YOLO)
======================================
Fine-tune YOLO26 on a custom object detection dataset.

Usage::

    python train.py --data path/to/data.yaml
    python train.py --data path/to/data.yaml --model yolo26s.pt --epochs 100
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

from train.train_detection import train_detection


def main():
    parser = argparse.ArgumentParser(description="P12 Object Detection — YOLO Fine-Tuning")
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    train_detection(
        data_yaml=args.data, model=args.model, epochs=args.epochs,
        imgsz=args.imgsz, batch=args.batch, device=args.device,
        project=str(ROOT / "runs" / "detect"), name="object_detect",
        patience=args.patience, lr0=args.lr0, workers=args.workers,
        resume=args.resume,
        registry_project="object_detection",
    )


if __name__ == "__main__":
    main()
