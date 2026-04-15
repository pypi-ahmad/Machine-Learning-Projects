"""Train Number Plate Reader Pro.

Usage::

    python train.py
    python train.py --data path/to/data.yaml --epochs 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_bootstrap import ensure_plate_dataset
from train.train_detection import train_detection


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train Number Plate Reader Pro (YOLO)")
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolo26m.pt", help="Base YOLO model")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--force-download", action="store_true", help="Force re-download dataset")
    args = parser.parse_args(argv)

    if args.data is None:
        data_path = ensure_plate_dataset(force=args.force_download)
        data_yaml = str(data_path / "data.yaml")
        print(f"[INFO] Resolved dataset -> {data_path}")
        print(f"[INFO] Using YOLO dataset config -> {data_yaml}")
    else:
        data_yaml = args.data

    train_detection(
        data_yaml=data_yaml,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(Path(__file__).parent / "runs"),
        name="plate_detect",
        registry_project="number_plate_reader_pro",
    )


if __name__ == "__main__":
    main()
