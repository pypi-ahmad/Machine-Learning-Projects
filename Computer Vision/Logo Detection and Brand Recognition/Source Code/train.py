"""Train Logo Detection & Brand Recognition — preferred detection trainer.

This project's modern default is in-scene logo localization with YOLO
detection, optionally followed by brand recognition. If you only have cropped
logo images for a closed-set classifier, keep that as a separate baseline; the
primary path here expects a YOLO-format ``data.yaml``.

Usage::

    python train.py --data path/to/data.yaml
    python train.py --data path/to/data.yaml --epochs 50 --model yolo26m.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.datasets import DatasetResolver
from train.train_detection import train_detection


def _resolve_data_yaml(data_arg: str | None, force_download: bool) -> Path:
    if data_arg is None:
        data_path = Path(DatasetResolver().resolve("logo_detection", force=force_download))
        print(f"[INFO] Resolved dataset → {data_path}")
    else:
        data_path = Path(data_arg)

    if data_path.is_file() and data_path.suffix.lower() in {".yaml", ".yml"}:
        return data_path

    if data_path.is_dir():
        for candidate in (data_path / "data.yaml", data_path / "processed" / "data.yaml"):
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"No YOLO data.yaml found under {data_path}. Prepare YOLO-format logo annotations or pass --data path/to/data.yaml."
        )

    raise FileNotFoundError(
        f"Expected --data to be a YOLO data.yaml or a directory containing one, got: {data_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Logo Detection (YOLO detection)")
    parser.add_argument("--data", type=str, default=None, help="Path to YOLO data.yaml or a directory containing it")
    parser.add_argument("--model", type=str, default="yolo26m.pt", help="YOLO base weights, e.g. yolo26m.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--project", type=str, default=str(Path(__file__).parent / "runs" / "logo_detect"))
    parser.add_argument("--name", type=str, default="train")
    parser.add_argument("--force-download", action="store_true", help="Force re-download dataset")
    args = parser.parse_args()

    data_yaml = _resolve_data_yaml(args.data, args.force_download)

    train_detection(
        data_yaml=str(data_yaml),
        model=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        registry_project="logo_detection",
    )


if __name__ == "__main__":
    main()
