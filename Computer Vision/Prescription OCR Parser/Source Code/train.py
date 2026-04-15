"""Train / Evaluate Prescription OCR Parser.

This project uses an OCR-first engine with PaddleOCR preferred and
EasyOCR fallback, so no training is needed. This script downloads the
dataset and evaluates the full pipeline on sample images.

Usage::

    python train.py
    python train.py --data path/to/dataset
    python train.py --force-download
    python train.py --max-samples 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_bootstrap import ensure_prescription_dataset


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Train/Evaluate Prescription OCR Parser",
    )
    ap.add_argument("--data", type=str, default=None,
                    help="Path to prescription dataset")
    ap.add_argument("--max-samples", type=int, default=20,
                    help="Max samples to evaluate")
    ap.add_argument("--force-download", action="store_true",
                    help="Force re-download dataset")
    args = ap.parse_args(argv)

    if args.data is None:
        data_path = ensure_prescription_dataset(force=args.force_download)
        data_dir = str(data_path)
        print(f"[INFO] Resolved dataset -> {data_path}")
    else:
        data_dir = args.data

    print(f"[INFO] Dataset ready at {data_dir}")
    print("[INFO] OCR engine is pre-trained; evaluating extraction on images.")
    print(
        "[INFO] DISCLAIMER: This tool is for informational purposes only. "
        "It does NOT provide medical advice."
    )

    try:
        from modern import PrescriptionOCRParser

        proj = PrescriptionOCRParser()
        proj.setup()

        data_root = Path(data_dir)
        images = sorted(
            list(data_root.rglob("*.jpg"))
            + list(data_root.rglob("*.png"))
            + list(data_root.rglob("*.jpeg"))
        )
        if not images:
            print("[WARN] No images found in dataset directory.")
            return

        results = []
        for img_path in images[: args.max_samples]:
            out = proj.predict(str(img_path))
            result = out["result"]
            report = out["report"]
            entry = {
                "file": img_path.name,
                "num_blocks": result.num_blocks,
                "num_medicines": result.num_medicines,
                "medicines": [m.medicine_name for m in result.medicines],
                "mean_confidence": round(result.mean_confidence, 3),
                "valid": report.valid,
                "warnings": len(report.warnings),
            }
            results.append(entry)
            med_str = ", ".join(m.medicine_name[:20] for m in result.medicines)
            print(
                f"  {img_path.name}: "
                f"{result.num_blocks} blocks, "
                f"{result.num_medicines} meds=[{med_str}], "
                f"conf={result.mean_confidence:.2f}"
            )

        # Summary
        out_path = Path(__file__).parent / "runs" / "eval_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

        total_meds = sum(r["num_medicines"] for r in results)
        avg_conf = (
            sum(r["mean_confidence"] for r in results) / max(len(results), 1)
        )
        print(f"\n[INFO] Evaluated {len(results)} prescriptions")
        print(f"[INFO] Total medicines found: {total_meds}")
        print(f"[INFO] Avg confidence: {avg_conf:.2f}")
        print(f"[INFO] Results saved to {out_path}")

    except ImportError as e:
        print(f"[WARN] Could not run evaluation: {e}")
        print("[INFO] Install OCR deps: pip install paddleocr paddlepaddle easyocr")


if __name__ == "__main__":
    main()
