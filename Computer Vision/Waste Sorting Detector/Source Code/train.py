"""Training script for Waste Sorting Detector.

Downloads the waste detection dataset (via data_bootstrap) and
delegates training to the shared ``train/train_detection.py`` helper.

Usage::

    python train.py                          # defaults — 50 epochs, imgsz 640
    python train.py --epochs 100 --batch 16
    python train.py --force-download         # re-download dataset
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_bootstrap import ensure_waste_dataset
from train.train_detection import train_detection

log = logging.getLogger("waste_sorting.train")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train waste detection model")
    p.add_argument("--model", default="yolo26m.pt", help="Base model weights")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="", help="CUDA device or 'cpu'")
    p.add_argument("--project", default=str(Path(__file__).resolve().parent / "runs"), help="Output project dir")
    p.add_argument("--name", default="train", help="Run name")
    p.add_argument("--force-download", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    args = _parse_args(argv)

    # 1. Ensure dataset exists
    data_root = ensure_waste_dataset(force=args.force_download)
    data_yaml = data_root / "data.yaml"
    if not data_yaml.exists():
        data_yaml = data_root / "processed" / "data.yaml"
    if not data_yaml.exists():
        log.error("data.yaml not found under %s — cannot train", data_root)
        sys.exit(1)

    # 2. Train
    log.info("Training with data_yaml=%s  model=%s  epochs=%d", data_yaml, args.model, args.epochs)
    train_detection(
        data_yaml=str(data_yaml),
        model=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device or None,
        project=args.project,
        name=args.name,
        registry_project="waste_sorting_detector",
    )


if __name__ == "__main__":
    main()
