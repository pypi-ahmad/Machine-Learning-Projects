"""Train Face Landmark Detection — YOLO pose estimation.

Usage::

    python train.py
    python train.py --data path/to/data.yaml --epochs 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.datasets import DatasetResolver
from train.train_pose import train_pose


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Face Landmark Detection (YOLO-pose)")
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml (keypoint)")
    parser.add_argument("--model", type=str, default="yolo26n-pose.pt", help="Base YOLO pose model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--force-download", action="store_true", help="Force re-download dataset")
    args = parser.parse_args()

    if args.data is None:
        data_path = DatasetResolver().resolve("face_landmark_detection", force=args.force_download)
        data_yaml = str(data_path / "data.yaml")
        print(f"[INFO] Resolved dataset → {data_path}")
        print("[INFO] Ensure data.yaml exists with YOLO keypoint annotations.")
    else:
        data_yaml = args.data

    train_pose(
        data_yaml=data_yaml,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(Path(__file__).parent / "runs"),
        name="face_landmark_pose",
        registry_project="face_landmark_detection",
    )


if __name__ == "__main__":
    main()
