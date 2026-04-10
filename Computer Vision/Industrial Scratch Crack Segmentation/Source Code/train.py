"""Industrial Scratch / Crack Segmentation — training & evaluation.

Usage::

    # Fine-tune YOLO26m-seg on defect data
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
    """Fine-tune YOLO26m-seg on surface defect data."""
    from train.train_segmentation import train_segmentation
    from utils.datasets import DatasetResolver

    if args.data is None:
        data_path = DatasetResolver().resolve(
            "industrial_scratch_crack_segmentation", force=args.force_download,
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
        name="defect_seg",
        registry_project="industrial_scratch_crack_segmentation",
    )


def _evaluate(args: argparse.Namespace) -> None:
    """Run segmentation on dataset images and report defect stats."""
    import cv2

    from config import DefectConfig
    from controller import DefectController
    from data_bootstrap import ensure_defect_dataset

    data_root = ensure_defect_dataset(force=args.force_download)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images: list[Path] = []
    for d in sorted(data_root.rglob("*")):
        if d.is_file() and d.suffix.lower() in exts and "processed" not in str(d):
            images.append(d)

    max_imgs = min(len(images), args.max_images)
    images = images[:max_imgs]
    print(f"Evaluating defect segmentation on {max_imgs} image(s) ...")

    cfg = DefectConfig()
    ctrl = DefectController(cfg)
    ctrl.load()

    total_defects = 0
    total_area = 0
    total_px = 0
    severity_counts = {"none": 0, "low": 0, "medium": 0, "high": 0}

    for idx, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        result = ctrl.process(frame, source=str(img_path))
        m = result.metrics

        total_defects += m.defect_count
        total_area += m.total_defect_area_px
        total_px += m.total_image_px
        severity_counts[m.severity] = severity_counts.get(m.severity, 0) + 1

        if (idx + 1) % 10 == 0 or idx == len(images) - 1:
            print(f"  [{idx + 1}/{max_imgs}] {img_path.name}: "
                  f"{m.defect_count} defect(s) severity={m.severity.upper()}")

    ctrl.close()

    avg_cov = total_area / total_px if total_px else 0
    print(f"\n{'=' * 50}")
    print(f"Images evaluated:      {max_imgs}")
    print(f"Total defects found:   {total_defects}")
    print(f"Average coverage:      {avg_cov:.2%}")
    print(f"Severity distribution: {severity_counts}")
    print(f"{'=' * 50}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Industrial Scratch / Crack Segmentation — train / evaluate",
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
