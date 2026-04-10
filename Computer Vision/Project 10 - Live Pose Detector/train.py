"""
Train — P10 Pose Detection (YOLO-Pose)
=========================================
Fine-tune YOLO26-Pose for body pose estimation.

Usage::

    python train.py --data path/to/pose_data.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)


def train_pose(data_yaml, model="yolo26n-pose.pt", epochs=50, imgsz=640,
               batch=16, device=None, project="runs/pose", name="train",
               patience=10, lr0=0.01, workers=4, resume=False,
               registry_project=None):
    from ultralytics import YOLO
    data_yaml = str(Path(data_yaml).resolve())
    yolo = YOLO(model)
    kwargs = dict(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch,
                  project=project, name=name, exist_ok=True,
                  patience=patience, lr0=lr0, workers=workers,
                  resume=resume, verbose=True)
    if device:
        kwargs["device"] = device
    results = yolo.train(**kwargs)
    best_path = str(yolo.trainer.best) if yolo.trainer.best else None
    if registry_project and best_path:
        try:
            from models.registry import get_registry
            reg = get_registry()
            version = reg.next_version(registry_project)
            reg.register(project=registry_project, version=version,
                         path=best_path, metrics={'epochs': epochs, 'base_model': model},
                         copy_weights=True)
        except Exception as exc:
            logging.getLogger(__name__).warning("Registration failed: %s", exc)
    return results


def main():
    parser = argparse.ArgumentParser(description="P10 PoseDetector — YOLO-Pose Fine-Tuning")
    parser.add_argument("--data", required=True, help="Path to YOLO-Pose data.yaml")
    parser.add_argument("--model", default="yolo26n-pose.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    train_pose(args.data, args.model, args.epochs, args.imgsz, args.batch,
               args.device, str(ROOT / "runs" / "pose"), "pose_detect",
               args.patience, args.lr0, args.workers, args.resume,
               registry_project="pose_detection")


if __name__ == "__main__":
    main()
