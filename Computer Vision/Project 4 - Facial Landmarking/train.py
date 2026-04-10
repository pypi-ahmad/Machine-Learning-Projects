"""
Train — P04 Facial Landmarking (YOLO-Pose)
=============================================
Fine-tune YOLO26-Pose on a custom facial landmark dataset.

The project uses a pretrained YOLO-Pose model for facial keypoints.
To fine-tune, provide a YOLO-Pose ``data.yaml`` with keypoint annotations.

Data format: YOLO-Pose (class cx cy w h kp1_x kp1_y kp1_v ... per line).

Usage::

    python train.py --data path/to/pose_data.yaml
    python train.py --data path/to/pose_data.yaml --model yolo26s-pose.pt
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


def train_pose(
    data_yaml: str | Path,
    model: str = "yolo26n-pose.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str | None = None,
    project: str = "runs/pose",
    name: str = "train",
    patience: int = 10,
    lr0: float = 0.01,
    workers: int = 4,
    resume: bool = False,
    registry_project: str | None = None,
):
    """Train a YOLO-Pose model (same API as YOLO detection training)."""
    from ultralytics import YOLO

    data_yaml = str(Path(data_yaml).resolve())
    logger.info("=" * 60)
    logger.info("YOLO-Pose Training")
    logger.info("  data:   %s", data_yaml)
    logger.info("  model:  %s", model)
    logger.info("  epochs: %d", epochs)
    logger.info("=" * 60)

    yolo = YOLO(model)
    kwargs = dict(
        data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch,
        project=project, name=name, exist_ok=True,
        patience=patience, lr0=lr0, workers=workers, resume=resume,
        verbose=True,
    )
    if device is not None:
        kwargs["device"] = device
    results = yolo.train(**kwargs)
    best_path = str(yolo.trainer.best) if yolo.trainer.best else None
    logger.info("Pose training complete. Best: %s", best_path)

    if registry_project and best_path:
        try:
            from models.registry import get_registry
            reg = get_registry()
            version = reg.next_version(registry_project)
            metrics = {'epochs': epochs, 'base_model': model}
            if hasattr(results, 'results_dict'):
                rd = results.results_dict
                for key in ('metrics/mAP50(B)', 'metrics/mAP50-95(B)'):
                    if key in rd:
                        short = 'mAP50' if '50(B)' in key and '95' not in key else 'mAP'
                        metrics[short] = round(float(rd[key]), 4)
            reg.register(project=registry_project, version=version,
                         path=best_path, metrics=metrics, copy_weights=True)
        except Exception as exc:
            logger.warning("Model registration failed (non-fatal): %s", exc)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="P04 Facial Landmarking — YOLO-Pose Fine-Tuning",
    )
    parser.add_argument("--data", required=True,
                        help="Path to YOLO-Pose data.yaml")
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )

    train_pose(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(ROOT / "runs" / "pose"),
        name="facial_landmarks",
        patience=args.patience,
        lr0=args.lr0,
        workers=args.workers,
        resume=args.resume,
        registry_project="facial_landmarks",
    )


if __name__ == "__main__":
    main()
