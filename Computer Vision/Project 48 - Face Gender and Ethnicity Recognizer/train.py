"""
Train — P48 Face, Gender & Ethnicity Recognition
===================================================
Multi-output classification: age (regression), gender (2), ethnicity (5).

The UTKFace dataset uses filename-encoded labels::

    age_gender_race_date.jpg
    e.g.  25_0_3_20170109211032.jpg  →  age=25, gender=0(M), race=3

Steps:
    1. Extract the local zip or download from Kaggle via DatasetResolver.
    2. Parse filename labels and split into train/val ImageFolder layout.
    3. Train a ResNet-18 gender classifier (simplest task).

For multi-output (age + gender + ethnicity), a custom training script
is needed. This template demonstrates single-task gender classification.

Usage::

    python train.py                      # Auto-extract local zip + train
    python train.py --task ethnicity     # Train ethnicity classifier
    python train.py --task age           # Train age-bucket classifier
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.train_classification import train_classification

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
GENDER_MAP = {0: "male", 1: "female"}
RACE_MAP = {0: "white", 1: "black", 2: "asian", 3: "indian", 4: "other"}
AGE_BUCKETS = [(0, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 60), (61, 120)]
AGE_LABELS = ["0-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61+"]


def _age_bucket(age: int) -> str:
    for (lo, hi), label in zip(AGE_BUCKETS, AGE_LABELS):
        if lo <= age <= hi:
            return label
    return "61+"


def _parse_utk_filename(filename: str):
    """Parse UTKFace filename → (age, gender, race) or None."""
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    try:
        age = int(parts[0])
        gender = int(parts[1])
        race = int(parts[2])
    except ValueError:
        return None
    if gender not in GENDER_MAP or race not in RACE_MAP:
        return None
    return age, gender, race


def _find_images(search_dir: Path) -> list[Path]:
    """Recursively find image files."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = []
    for p in search_dir.rglob("*"):
        if p.suffix.lower() in exts and _parse_utk_filename(p.name) is not None:
            images.append(p)
    return images


def prepare_dataset(
    task: str = "gender",
    train_ratio: float = 0.8,
) -> Path:
    """Extract zip, parse filenames, and create ImageFolder layout.

    Returns the output directory ready for ``train_classification``.
    """
    import random

    # Try to find local zip
    data_dir = REPO_ROOT / "data" / "project_48"
    zip_candidates = list(data_dir.glob("*.zip")) if data_dir.exists() else []

    extracted_dir = ROOT / "extracted_data"

    if not extracted_dir.exists() or not any(extracted_dir.iterdir()):
        if zip_candidates:
            logger.info("Extracting %s ...", zip_candidates[0].name)
            extracted_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_candidates[0], "r") as zf:
                zf.extractall(extracted_dir)
        else:
            # Try DatasetResolver
            try:
                from utils.datasets import resolve
                ds_path = resolve("utkface")
                extracted_dir = Path(ds_path)
                logger.info("Using DatasetResolver dataset at %s", ds_path)
            except Exception:
                logger.error(
                    "No local zip found in %s and DatasetResolver failed.\n"
                    "Place UTKFace images in %s or configure Kaggle.",
                    data_dir, extracted_dir,
                )
                sys.exit(1)

    # Find all parseable images
    images = _find_images(extracted_dir)
    if not images:
        logger.error("No UTKFace images found in %s", extracted_dir)
        sys.exit(1)
    logger.info("Found %d UTKFace images", len(images))

    # Create ImageFolder layout
    output_dir = ROOT / f"dataset_{task}"
    if output_dir.exists():
        logger.info("Dataset dir %s already exists, reusing", output_dir)
        return output_dir

    random.seed(42)
    random.shuffle(images)
    split = int(len(images) * train_ratio)

    for i, img_path in enumerate(images):
        parsed = _parse_utk_filename(img_path.name)
        if parsed is None:
            continue
        age, gender, race = parsed

        if task == "gender":
            label = GENDER_MAP[gender]
        elif task == "ethnicity":
            label = RACE_MAP[race]
        elif task == "age":
            label = _age_bucket(age)
        else:
            raise ValueError(f"Unknown task: {task}")

        split_name = "train" if i < split else "val"
        dst_dir = output_dir / split_name / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / img_path.name
        if not dst.exists():
            shutil.copy2(img_path, dst)

    logger.info("ImageFolder dataset created at %s", output_dir)
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="P48 Face Attributes — Classification Training",
    )
    parser.add_argument(
        "--task",
        default="gender",
        choices=["gender", "ethnicity", "age"],
        help="Which attribute to classify",
    )
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--device", default=None)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )

    # Prepare data
    data_dir = prepare_dataset(task=args.task, train_ratio=args.train_ratio)

    # Determine num_classes
    cls_map = {"gender": 2, "ethnicity": 5, "age": len(AGE_LABELS)}
    num_classes = cls_map[args.task]

    # Train
    logger.info("Training %s classifier (%d classes)...", args.task, num_classes)
    result = train_classification(
        data_dir=data_dir,
        num_classes=num_classes,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=args.img_size,
        device=args.device,
        output_dir=str(ROOT / "runs" / "classify" / args.task),
        patience=args.patience,
        freeze_backbone=args.freeze_backbone,
        workers=args.workers,
        registry_project=f"face_attributes_{args.task}",
    )
    logger.info(
        "Done! best_acc=%.4f @ epoch %d  →  %s",
        result["best_acc"], result["best_epoch"], result["model_path"],
    )


if __name__ == "__main__":
    main()
