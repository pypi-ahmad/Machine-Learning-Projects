"""Train retail shelf stockout detector — YOLO detection.

Automatically downloads and prepares the dataset on first run via
``data_bootstrap.ensure_retail_dataset()``.

Usage::

    python train.py
    python train.py --data path/to/data.yaml --epochs 100
    python train.py --force-download   # re-download dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
# Project root (for local imports)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_bootstrap import ensure_retail_dataset
from train.train_detection import train_detection


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Retail Shelf Stockout Detector (YOLO)")
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolo26m.pt", help="Base YOLO model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--force-download", action="store_true", help="Force re-download dataset")
    args = parser.parse_args()

    # Dataset bootstrap — idempotent, skips if already prepared
    data_root = ensure_retail_dataset(force=args.force_download)

    if args.data is None:
        data_yaml = str(data_root / "data.yaml")
        if not Path(data_yaml).exists():
            # Try processed/
            alt = data_root / "processed" / "data.yaml"
            if alt.exists():
                data_yaml = str(alt)
        print(f"[INFO] Dataset → {data_root}")
        print(f"[INFO] data.yaml → {data_yaml}")
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
        name="retail_shelf_detect",
        registry_project="retail_shelf_stockout",
    )


if __name__ == "__main__":
    main()
