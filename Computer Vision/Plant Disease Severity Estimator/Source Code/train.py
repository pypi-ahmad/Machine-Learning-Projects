"""Plant Disease Severity Estimator — training script.

Trains a classification model on PlantVillage (38 classes) using
the shared training pipeline.

Usage::

    python train.py
    python train.py --data path/to/dataset --epochs 30
    python train.py --model efficientnet_b0 --batch 16
    python train.py --force-download
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.datasets import DatasetResolver
from train.train_classification import train_classification


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train Plant Disease Severity Estimator"
    )
    ap.add_argument("--data", type=str, default=None,
                    help="Path to image-folder dataset (auto-downloads if omitted)")
    ap.add_argument("--model", type=str, default="resnet18",
                    help="resnet18|resnet34|resnet50|efficientnet_b0|mobilenet_v2")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--force-download", action="store_true",
                    help="Force re-download dataset")
    args = ap.parse_args()

    if args.data is None:
        data_path = DatasetResolver().resolve(
            "plant_disease_severity_estimator", force=args.force_download
        )
        data_dir = str(data_path)
        print(f"[INFO] Resolved dataset → {data_path}")
    else:
        data_dir = args.data

    save_dir = Path(__file__).parent / "runs" / "plant_disease_cls"

    stats = train_classification(
        data_dir=data_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        lr=args.lr,
        device=args.device,
        save_path=str(save_dir / "best_model.pt"),
        registry_project="plant_disease_severity_estimator",
    )

    print(f"\nTraining complete!")
    print(f"  Best accuracy: {stats['best_acc']:.2%}")
    print(f"  Classes:       {stats['classes']}")
    print(f"  Weights:       {stats['weights']}")


if __name__ == "__main__":
    main()
