"""Parking Occupancy Monitor — evaluation harness.

Runs YOLO ``val()`` on the test/validation split and writes per-class
metrics to a JSON report.

Usage::

    python evaluate.py
    python evaluate.py --model runs/parking_occupancy_monitor/train/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_bootstrap import ensure_parking_dataset  # noqa: E402
from utils.yolo import load_yolo                   # noqa: E402

log = logging.getLogger("parking_occupancy.evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parking Occupancy Monitor — Evaluation")
    parser.add_argument("--model", type=str, default="yolo11m.pt",
                        help="Model weights path")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    data_root = ensure_parking_dataset(force=args.force_download)
    data_yaml = data_root / "data.yaml"
    if not data_yaml.exists():
        data_yaml = data_root / "processed" / "data.yaml"
    if not data_yaml.exists():
        log.error("data.yaml not found — run with --force-download")
        sys.exit(1)

    model = load_yolo(args.model)
    metrics = model.val(data=str(data_yaml), imgsz=args.imgsz, verbose=False)

    # Build report
    report: dict = {
        "model": args.model,
        "data_yaml": str(data_yaml),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "per_class": {},
    }

    names = metrics.names if hasattr(metrics, "names") else {}
    for idx, cls_name in names.items():
        report["per_class"][cls_name] = {
            "precision": float(metrics.box.p[idx]) if idx < len(metrics.box.p) else None,
            "recall": float(metrics.box.r[idx]) if idx < len(metrics.box.r) else None,
            "ap50": float(metrics.box.ap50[idx]) if idx < len(metrics.box.ap50) else None,
        }

    out_path = Path("outputs") / "eval_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Evaluation report → %s", out_path)

    print(f"\nmAP@50  : {report['mAP50']:.4f}")
    print(f"mAP@50-95: {report['mAP50_95']:.4f}")
    for cls_name, vals in report["per_class"].items():
        print(f"  {cls_name:20s}  P={vals['precision']:.3f}  R={vals['recall']:.3f}  AP50={vals['ap50']:.3f}")


if __name__ == "__main__":
    main()
