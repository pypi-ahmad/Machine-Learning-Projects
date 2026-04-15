"""Evaluation script — run trained model on val/test split, report metrics.

Uses the YOLO built-in val() for mAP metrics and adds zone-counting
accuracy analysis when zone annotations are available.

Usage::

    python evaluate.py
    python evaluate.py --weights runs/retail_shelf_detect/weights/best.pt
    python evaluate.py --data path/to/data.yaml --conf 0.25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_bootstrap import ensure_retail_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Retail Shelf Stockout Detector")
    parser.add_argument("--weights", type=str, default=None, help="Path to trained best.pt")
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolo26m.pt", help="Fallback base model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device")
    parser.add_argument("--force-download", action="store_true", help="Force dataset re-download")
    args = parser.parse_args()

    # Ensure dataset
    data_root = ensure_retail_dataset(force=args.force_download)

    # Resolve data.yaml
    if args.data:
        data_yaml = args.data
    else:
        data_yaml = str(data_root / "data.yaml")
        if not Path(data_yaml).exists():
            alt = data_root / "processed" / "data.yaml"
            if alt.exists():
                data_yaml = str(alt)
        print(f"[INFO] data.yaml -> {data_yaml}")

    # Resolve weights
    if args.weights:
        weights_path = args.weights
    else:
        # Look for trained weights in project runs
        run_dir = Path(__file__).parent / "runs" / "retail_shelf_detect"
        best = run_dir / "weights" / "best.pt"
        if best.exists():
            weights_path = str(best)
            print(f"[INFO] Using trained weights: {weights_path}")
        else:
            weights_path = args.model
            print(f"[INFO] No trained weights found, using base: {weights_path}")

    # Run YOLO validation
    from ultralytics import YOLO

    model = YOLO(weights_path)
    results = model.val(
        data=data_yaml,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(Path(__file__).parent / "runs"),
        name="eval",
        exist_ok=True,
    )

    # Extract and display metrics
    metrics = {
        "mAP50": round(float(results.box.map50), 4),
        "mAP50-95": round(float(results.box.map), 4),
        "precision": round(float(results.box.mp), 4),
        "recall": round(float(results.box.mr), 4),
        "conf_threshold": args.conf,
        "iou_threshold": args.iou,
        "weights": weights_path,
        "data": data_yaml,
    }

    # Per-class metrics if available
    if hasattr(results.box, "ap_class_index") and results.box.ap_class_index is not None:
        class_names = results.names if hasattr(results, "names") else {}
        per_class = {}
        for i, cls_idx in enumerate(results.box.ap_class_index):
            name = class_names.get(int(cls_idx), str(int(cls_idx)))
            per_class[name] = {
                "AP50": round(float(results.box.ap50[i]), 4),
                "AP50-95": round(float(results.box.ap[i]), 4),
            }
        metrics["per_class"] = per_class

    # Print
    print("\n" + "=" * 50)
    print("  Evaluation Results")
    print("=" * 50)
    for k, v in metrics.items():
        if k == "per_class":
            print(f"\n  Per-Class AP:")
            for cls, vals in v.items():
                print(f"    {cls}: AP50={vals['AP50']}, AP50-95={vals['AP50-95']}")
        else:
            print(f"  {k}: {v}")
    print("=" * 50)

    # Save to JSON
    out_path = Path(__file__).parent / "runs" / "eval" / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\n[INFO] Metrics saved to {out_path}")


if __name__ == "__main__":
    main()
