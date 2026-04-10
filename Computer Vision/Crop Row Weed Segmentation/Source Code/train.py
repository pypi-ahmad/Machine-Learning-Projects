"""Crop Row & Weed Segmentation — training & evaluation.

Usage::

    # Fine-tune YOLO26m-seg on crop/weed segmentation data
    python train.py --data path/to/data.yaml --epochs 80

    # Evaluate segmentation on dataset images
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
    """Fine-tune YOLO26m-seg on crop/weed segmentation data."""
    from train.train_segmentation import train_segmentation
    from utils.datasets import DatasetResolver

    if args.data is None:
        data_path = DatasetResolver().resolve(
            "crop_row_weed_segmentation", force=args.force_download,
        )
        data_yaml = str(data_path / "data.yaml")
        print(f"[INFO] Resolved dataset → {data_path}")
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
        name="cropweed_seg",
        registry_project="crop_row_weed_segmentation",
    )


def _evaluate(args: argparse.Namespace) -> None:
    """Run segmentation on dataset images and report class distribution."""
    import cv2

    from class_stats import compute_area_stats
    from config import CropWeedConfig
    from controller import CropWeedController
    from data_bootstrap import ensure_cropweed_dataset

    data_root = ensure_cropweed_dataset(force=args.force_download)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images: list[Path] = []
    for d in (data_root / "processed" / "media", data_root / "raw"):
        if d.is_dir():
            images.extend(f for f in sorted(d.rglob("*")) if f.suffix.lower() in exts)
    if not images:
        images = [f for f in sorted(data_root.rglob("*"))
                  if f.suffix.lower() in exts and "processed" not in str(f)]

    max_imgs = min(len(images), args.max_images)
    images = images[:max_imgs]
    print(f"Evaluating crop/weed segmentation on {max_imgs} image(s) ...")

    cfg = CropWeedConfig()
    ctrl = CropWeedController(cfg)
    ctrl.load()

    class_totals: dict[str, int] = {}
    class_instances: dict[str, int] = {}
    total_instances = 0

    for idx, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        result = ctrl.process(frame)
        ar = result.area_report

        total_instances += ar.total_instances
        for cn, cs in ar.per_class.items():
            class_totals[cn] = class_totals.get(cn, 0) + cs.total_area_px
            class_instances[cn] = class_instances.get(cn, 0) + cs.instance_count

        if (idx + 1) % 10 == 0 or idx == len(images) - 1:
            print(f"  [{idx + 1}/{max_imgs}] {img_path.name}: "
                  f"{ar.total_instances} instances")

    ctrl.close()

    print(f"\n{'=' * 50}")
    print(f"Images evaluated:     {max_imgs}")
    print(f"Total instances:      {total_instances}")
    for cn in sorted(class_totals):
        print(f"  {cn:20s}  instances={class_instances.get(cn, 0):5d}  "
              f"area_px={class_totals[cn]:,}")
    print(f"{'=' * 50}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop Row & Weed Segmentation — train / evaluate",
    )
    parser.add_argument("--eval", action="store_true",
                        help="Run evaluation on dataset images")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to data.yaml for YOLO training")
    parser.add_argument("--model", type=str, default="yolo26m-seg.pt",
                        help="Base YOLO-seg model")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-images", type=int, default=50,
                        help="Max images to evaluate (default: 50)")
    parser.add_argument("--force-download", action="store_true",
                        help="Force re-download dataset")
    args = parser.parse_args()

    if args.eval:
        _evaluate(args)
    else:
        _train(args)


if __name__ == "__main__":
    main()
