"""Road Pothole Segmentation — training & evaluation.
"""Road Pothole Segmentation — training & evaluation.

Usage::

    # Fine-tune YOLO26m-seg on pothole segmentation data
    python train.py --data path/to/data.yaml --epochs 50

    # Evaluate on dataset images
    python train.py --eval

    # Force re-download dataset
    python train.py --eval --force-download
"""
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
    """Fine-tune YOLO26m-seg on pothole segmentation data."""
    from train.train_segmentation import train_segmentation
    from utils.datasets import DatasetResolver

    if args.data is None:
        data_path = DatasetResolver().resolve(
            "road_pothole_segmentation", force=args.force_download,
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
        name="pothole_seg",
        registry_project="road_pothole_segmentation",
    )


def _evaluate(args: argparse.Namespace) -> None:
    """Run pothole segmentation on dataset images and report statistics."""
    import cv2

    from config import PotholeConfig
    from controller import PotholeController
    from data_bootstrap import ensure_pothole_dataset
    from severity import SEVERITY_MINOR, SEVERITY_MODERATE, SEVERITY_SEVERE

    data_root = ensure_pothole_dataset(force=args.force_download)

    # Find images in dataset
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
    print(f"Evaluating pothole segmentation on {max_imgs} image(s) ...")

    cfg = PotholeConfig()
    ctrl = PotholeController(cfg)
    ctrl.load()

    counts = {SEVERITY_MINOR: 0, SEVERITY_MODERATE: 0, SEVERITY_SEVERE: 0}
    total_potholes = 0
    conditions: dict[str, int] = {}

    for idx, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        result = ctrl.process(frame)
        sev = result.severity

        total_potholes += sev.total_count
        counts[SEVERITY_MINOR] += sev.minor_count
        counts[SEVERITY_MODERATE] += sev.moderate_count
        counts[SEVERITY_SEVERE] += sev.severe_count
        conditions[sev.road_condition] = conditions.get(sev.road_condition, 0) + 1

        if (idx + 1) % 10 == 0 or idx == len(images) - 1:
            print(f"  [{idx + 1}/{max_imgs}] {img_path.name}: "
                  f"{sev.total_count} potholes, condition={sev.road_condition}")

    ctrl.close()

    print(f"\n{'=' * 50}")
    print(f"Images evaluated:    {max_imgs}")
    print(f"Total potholes:      {total_potholes}")
    print(f"  Minor:             {counts[SEVERITY_MINOR]}")
    print(f"  Moderate:          {counts[SEVERITY_MODERATE]}")
    print(f"  Severe:            {counts[SEVERITY_SEVERE]}")
    print(f"Road conditions:     {conditions}")
    print(f"{'=' * 50}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Road Pothole Segmentation -- train / evaluate",
    )
    parser.add_argument("--eval", action="store_true",
                        help="Run evaluation on dataset images")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to data.yaml for YOLO training")
    parser.add_argument("--model", type=str, default="yolo26m-seg.pt",
                        help="Base YOLO-seg model")
    parser.add_argument("--epochs", type=int, default=50)
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
