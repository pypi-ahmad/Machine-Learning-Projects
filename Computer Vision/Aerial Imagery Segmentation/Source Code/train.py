"""Train Aerial Imagery Segmentation — YOLO segmentation.

Usage::

    python train.py
    python train.py --data path/to/data.yaml --epochs 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.datasets import DatasetResolver
from train.train_segmentation import train_segmentation


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Aerial Imagery Segmentation (YOLO-seg)")
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolo26n-seg.pt", help="Base YOLO seg model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--force-download", action="store_true", help="Force re-download dataset")
    args = parser.parse_args()

    if args.data is None:
        data_path = DatasetResolver().resolve("aerial_imagery_seg", force=args.force_download)
        data_yaml = str(data_path / "data.yaml")
        print(f"[INFO] Resolved dataset -> {data_path}")
        print("[INFO] Ensure data.yaml exists with YOLO-format segmentation annotations.")
    else:
        data_yaml = args.data

    train_segmentation(
        data_yaml=data_yaml,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(Path(__file__).parent / "runs"),
        name="aerial_imagery_seg",
        registry_project="aerial_imagery_seg",
    )


if __name__ == "__main__":
    main()
