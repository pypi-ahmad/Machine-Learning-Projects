"""Training script for Drone Ship OBB Detector.

Downloads the OBB aerial dataset (via data_bootstrap) and delegates
training to the shared ``train/train_obb.py`` helper.

Usage::

    python train.py                          # defaults — 100 epochs, imgsz 1024
    python train.py --epochs 200 --batch 4
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

from data_bootstrap import ensure_obb_dataset

log = logging.getLogger("drone_ship_obb.train")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train OBB drone/ship detection model")
    p.add_argument("--model", default="yolo26m-obb.pt", help="Base OBB model weights")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--device", default="", help="CUDA device or 'cpu'")
    p.add_argument("--project", default=str(Path(__file__).resolve().parent / "runs"),
                    help="Output project dir")
    p.add_argument("--name", default="train", help="Run name")
    p.add_argument("--force-download", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    args = _parse_args(argv)

    # 1. Ensure dataset
    data_root = ensure_obb_dataset(force=args.force_download)
    data_yaml = data_root / "data.yaml"
    if not data_yaml.exists():
        data_yaml = data_root / "processed" / "data.yaml"
    if not data_yaml.exists():
        log.error("data.yaml not found under %s — cannot train", data_root)
        sys.exit(1)

    # 2. Train via shared OBB pipeline
    from train.train_obb import train_obb

    log.info("Training OBB model: data_yaml=%s  model=%s  epochs=%d", data_yaml, args.model, args.epochs)
    train_obb(
        data_yaml=str(data_yaml),
        model=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device or None,
        project=args.project,
        name=args.name,
        registry_project="drone_ship_obb_detector",
    )


if __name__ == "__main__":
    main()
