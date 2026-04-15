"""Building Footprint Change Detector — training & evaluation.

Usage::

    # Train YOLO-seg on building segmentation data
    python train.py --data path/to/data.yaml --epochs 50

    # Evaluate change detection on dataset pairs
    python train.py --eval

    # Force re-download dataset
    python train.py --eval --force-download
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _train(args: argparse.Namespace) -> None:
    """Fine-tune YOLO26m-seg on building segmentation data."""
    from train.train_segmentation import train_segmentation
    from utils.datasets import DatasetResolver

    if args.data is None:
        data_path = DatasetResolver().resolve(
            "building_footprint_change_detector", force=args.force_download,
        )
        data_yaml = str(data_path / "data.yaml")
        print(f"[INFO] Resolved dataset -> {data_path}")
        print("[INFO] Ensure data.yaml exists with YOLO-seg annotations.")
    else:
        data_yaml = args.data

    train_segmentation(
        data_yaml=data_yaml,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(_SRC / "runs"),
        name="building_change_seg",
        registry_project="building_footprint_change_detector",
    )


def _evaluate(args: argparse.Namespace) -> None:
    """Run change detection on dataset image pairs and report metrics."""
    from data_bootstrap import ensure_change_dataset

    data_root = ensure_change_dataset(force=args.force_download)
    before_dir = data_root / "processed" / "A"
    after_dir = data_root / "processed" / "B"

    if not before_dir.is_dir() or not after_dir.is_dir():
        # Fall back to raw directory structure
        for sub in data_root.rglob("*"):
            if sub.is_dir() and sub.name == "A":
                before_dir = sub
            if sub.is_dir() and sub.name == "B":
                after_dir = sub

    if not before_dir.is_dir() or not after_dir.is_dir():
        print("[WARN] Could not find A/ and B/ subdirectories in dataset.")
        print(f"       Data root: {data_root}")
        print("       Run change detection manually with --before / --after.")
        return

    from config import ChangeConfig
    from controller import ChangeDetectorController
    from validator import validate_directory_pair

    report, pairs = validate_directory_pair(before_dir, after_dir)
    if not report.ok:
        for w in report.warnings:
            print(f"ERROR: {w}", file=sys.stderr)
        return

    max_pairs = min(len(pairs), args.max_pairs)
    pairs = pairs[:max_pairs]
    print(f"Evaluating change detection on {max_pairs} pair(s) ...")

    cfg = ChangeConfig()
    ctrl = ChangeDetectorController(cfg)
    ctrl.load()

    total_new = 0
    total_demo = 0
    total_iou = 0.0

    for idx, (bp, ap) in enumerate(pairs):
        result = ctrl.process_pair(bp, ap)
        m = result.metrics
        total_new += m.num_new_regions
        total_demo += m.num_demolished_regions
        total_iou += m.iou

        if (idx + 1) % 10 == 0 or idx == len(pairs) - 1:
            print(f"  [{idx + 1}/{max_pairs}] {bp.name}: "
                  f"IoU={m.iou:.4f}  new={m.num_new_regions}  demo={m.num_demolished_regions}")

    ctrl.close()

    avg_iou = total_iou / max_pairs if max_pairs > 0 else 0.0
    print(f"\n{'=' * 50}")
    print(f"Pairs evaluated:       {max_pairs}")
    print(f"Total new regions:     {total_new}")
    print(f"Total demolished:      {total_demo}")
    print(f"Average IoU:           {avg_iou:.4f}")
    print(f"{'=' * 50}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Building Footprint Change Detector -- train / evaluate",
    )
    parser.add_argument("--eval", action="store_true",
                        help="Run evaluation on dataset pairs")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to data.yaml for YOLO training")
    parser.add_argument("--model", type=str, default="yolo26m-seg.pt",
                        help="Base YOLO-seg model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-pairs", type=int, default=50,
                        help="Max pairs to evaluate (default: 50)")
    parser.add_argument("--force-download", action="store_true",
                        help="Force re-download dataset")
    args = parser.parse_args()

    if args.eval:
        _evaluate(args)
    else:
        _train(args)


if __name__ == "__main__":
    main()
