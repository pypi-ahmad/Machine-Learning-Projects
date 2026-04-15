"""PPE Compliance Monitor — YOLO training entry-point.

Downloads the PPE detection dataset (if needed) and delegates to the
shared ``train/train_detection.py`` pipeline.

Usage::

    python train.py
    python train.py --force-download
    python train.py --epochs 50 --batch 16
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_bootstrap import ensure_ppe_dataset  # noqa: E402
from train.train_detection import train_detection  # noqa: E402

log = logging.getLogger("ppe_compliance.train")


def main() -> None:
    parser = argparse.ArgumentParser(description="PPE Compliance Monitor — Training")
    parser.add_argument("--force-download", action="store_true",
                        help="Force dataset re-download")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size")
    parser.add_argument("--model", type=str, default="yolo26m.pt",
                        help="Base YOLO model")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # Ensure dataset is available
    data_root = ensure_ppe_dataset(force=args.force_download)
    data_yaml = data_root / "data.yaml"

    if not data_yaml.exists():
        data_yaml = data_root / "processed" / "data.yaml"

    if not data_yaml.exists():
        log.error("data.yaml not found in %s — run with --force-download", data_root)
        sys.exit(1)

    log.info("Dataset ready at %s", data_root)
    log.info("Training with model=%s, epochs=%d, batch=%d, imgsz=%d",
             args.model, args.epochs, args.batch, args.imgsz)

    train_detection(
        data_yaml=str(data_yaml),
        model=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=str(Path(__file__).parent / "runs"),
        name="ppe_detect",
        registry_project="ppe_compliance_monitor",
    )


if __name__ == "__main__":
    main()
