"""Evaluation script for Crowd Zone Counter.

Runs YOLO ``val()`` on the person detection dataset and prints metrics.

Usage::

    python evaluate.py
    python evaluate.py --model runs/crowd_zone/train/weights/best.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_bootstrap import ensure_crowd_dataset
from utils.yolo import load_yolo

log = logging.getLogger("crowd_zone.evaluate")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate crowd/person detection model")
    p.add_argument("--model", default="yolo26m.pt", help="Model weights")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="", help="CUDA device or 'cpu'")
    p.add_argument("--force-download", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    args = _parse_args(argv)

    data_root = ensure_crowd_dataset(force=args.force_download)
    data_yaml = data_root / "data.yaml"
    if not data_yaml.exists():
        data_yaml = data_root / "processed" / "data.yaml"
    if not data_yaml.exists():
        log.error("data.yaml not found under %s -- cannot evaluate", data_root)
        sys.exit(1)

    model = load_yolo(args.model, device=args.device or None)
    log.info("Evaluating %s on %s", args.model, data_yaml)

    metrics = model.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
    )

    box = metrics.box
    log.info("---- Detection Results ----")
    log.info("mAP@50    : %.4f", box.map50)
    log.info("mAP@50-95 : %.4f", box.map)
    log.info("Precision : %.4f", box.mp)
    log.info("Recall    : %.4f", box.mr)

    if hasattr(box, "ap_class_index") and box.ap_class_index is not None:
        names = model.names
        log.info("---- Per-class AP@50 ----")
        for i, cls_idx in enumerate(box.ap_class_index):
            ap50 = box.ap50[i] if hasattr(box, "ap50") else 0.0
            log.info("  %-15s  %.4f", names.get(int(cls_idx), cls_idx), ap50)


if __name__ == "__main__":
    main()
