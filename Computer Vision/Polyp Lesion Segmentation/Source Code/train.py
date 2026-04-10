"""Polyp Lesion Segmentation — training & evaluation.

Usage::

    # Fine-tune YOLO26m-seg on polyp data
    python train.py --data path/to/data.yaml --epochs 80

    # Evaluate segmentation on dataset images (with GT masks)
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
    """Fine-tune YOLO26m-seg on polyp segmentation data."""
    from train.train_segmentation import train_segmentation
    from utils.datasets import DatasetResolver

    if args.data is None:
        data_path = DatasetResolver().resolve(
            "polyp_lesion_segmentation", force=args.force_download,
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
        name="polyp_seg",
        registry_project="polyp_lesion_segmentation",
    )


def _evaluate(args: argparse.Namespace) -> None:
    """Run segmentation on dataset images and report polyp stats."""
    import cv2

    from config import PolypConfig
    from controller import PolypController
    from data_bootstrap import ensure_polyp_dataset

    data_root = ensure_polyp_dataset(force=args.force_download)

    # Locate images and masks
    img_dir = _find_subdir(data_root, "images")
    mask_dir = _find_subdir(data_root, "masks")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images: list[Path] = []
    if img_dir and img_dir.is_dir():
        images = sorted(f for f in img_dir.iterdir() if f.suffix.lower() in exts)
    if not images:
        images = [f for f in sorted(data_root.rglob("*"))
                  if f.suffix.lower() in exts and "mask" not in f.parent.name.lower()]

    gt_lookup: dict[str, Path] = {}
    if mask_dir and mask_dir.is_dir():
        for f in mask_dir.iterdir():
            if f.suffix.lower() in exts:
                gt_lookup[f.stem] = f

    max_imgs = min(len(images), args.max_images)
    images = images[:max_imgs]
    print(f"Evaluating polyp segmentation on {max_imgs} image(s) ...")
    if gt_lookup:
        print(f"  Ground-truth masks: {len(gt_lookup)} available")

    cfg = PolypConfig()
    cfg.backend = args.backend
    ctrl = PolypController(cfg)
    ctrl.load()

    total_polyp_px = 0
    total_image_px = 0
    total_regions = 0
    total_dice = 0.0
    total_iou = 0.0
    n_gt = 0

    for idx, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        gt = None
        gt_path = gt_lookup.get(img_path.stem)
        if gt_path:
            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)

        result = ctrl.process(frame, source=str(img_path), gt_mask=gt)
        m = result.metrics

        total_polyp_px += m.polyp_area_px
        total_image_px += m.total_image_px
        total_regions += m.polyp_count

        if m.dice is not None:
            total_dice += m.dice
            total_iou += m.iou
            n_gt += 1

        if (idx + 1) % 10 == 0 or idx == len(images) - 1:
            dice_str = f"  Dice={m.dice:.4f}" if m.dice is not None else ""
            print(f"  [{idx + 1}/{max_imgs}] {img_path.name}: "
                  f"polyp={m.polyp_coverage:.2%} ({m.polyp_count} region(s))"
                  f"{dice_str}")

    ctrl.close()

    avg_cov = total_polyp_px / total_image_px if total_image_px else 0
    print(f"\n{'=' * 50}")
    print(f"Images evaluated:     {max_imgs}")
    print(f"Total polyp regions:  {total_regions}")
    print(f"Total polyp area:     {total_polyp_px:,} px")
    print(f"Average coverage:     {avg_cov:.2%}")
    if n_gt > 0:
        print(f"Mean Dice:            {total_dice / n_gt:.4f}")
        print(f"Mean IoU:             {total_iou / n_gt:.4f}")
        print(f"GT images evaluated:  {n_gt}")
    print(f"{'=' * 50}")


def _find_subdir(root: Path, name: str) -> Path | None:
    """Find a subdirectory by name (case-insensitive) under root."""
    for d in root.rglob("*"):
        if d.is_dir() and d.name.lower() == name.lower():
            return d
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Polyp Lesion Segmentation — train / evaluate",
    )
    parser.add_argument("--eval", action="store_true",
                        help="Run evaluation on dataset images")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to data.yaml for YOLO training")
    parser.add_argument("--model", type=str, default="yolo26m-seg.pt",
                        help="Base YOLO-seg model")
    parser.add_argument("--backend", type=str, default="yolo",
                        help="Segmentation backend for evaluation")
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
