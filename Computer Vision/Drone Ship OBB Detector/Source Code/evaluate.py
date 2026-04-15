"""Evaluation script for Drone Ship OBB Detector.

Runs YOLO OBB ``val()`` on the aerial dataset and prints per-class
OBB metrics (mAP@50, mAP@50-95, precision, recall).

OBB evaluation uses rotated-box IoU which properly accounts for
object orientation — critical for thin, elongated objects like ships.

Usage::

    python evaluate.py
    python evaluate.py --model runs/obb_ship/train/weights/best.pt
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
from utils.yolo import load_yolo

log = logging.getLogger("drone_ship_obb.evaluate")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate OBB drone/ship detection model")
    p.add_argument("--model", default="yolo26m-obb.pt", help="OBB model weights")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="", help="CUDA device or 'cpu'")
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
        log.error("data.yaml not found under %s — cannot evaluate", data_root)
        sys.exit(1)

    # 2. Load model and validate
    model = load_yolo(args.model, device=args.device or None)
    log.info("Evaluating OBB model %s on %s", args.model, data_yaml)

    metrics = model.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
    )

    # 3. Print summary
    box = metrics.box
    log.info("---- OBB Evaluation Results ----")
    log.info("mAP@50    : %.4f", box.map50)
    log.info("mAP@50-95 : %.4f", box.map)
    log.info("Precision : %.4f", box.mp)
    log.info("Recall    : %.4f", box.mr)

    if hasattr(box, "ap_class_index") and box.ap_class_index is not None:
        names = model.names
        log.info("---- Per-class AP@50 ----")
        for i, cls_idx in enumerate(box.ap_class_index):
            ap50 = box.ap50[i] if hasattr(box, "ap50") else 0.0
            log.info("  %-20s  %.4f", names.get(int(cls_idx), cls_idx), ap50)


if __name__ == "__main__":
    main()
