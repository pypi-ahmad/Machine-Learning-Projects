"""Train / Evaluate ID Card KYC Parser — OCR + field extraction.

Note: This project uses PaddleOCR (pre-trained) and template-based
field extraction, so training is optional.  This script downloads
the dataset and evaluates the full pipeline on sample images.

Usage::

    python train.py
    python train.py --data path/to/dataset
    python train.py --force-download
    python train.py --template passport --max-samples 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.datasets import DatasetResolver


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train/Evaluate ID Card KYC Parser")
    ap.add_argument("--data", type=str, default=None, help="Path to ID card dataset")
    ap.add_argument("--template", type=str, default="generic", help="Template to use")
    ap.add_argument("--max-samples", type=int, default=20, help="Max samples to evaluate")
    ap.add_argument("--force-download", action="store_true", help="Force re-download dataset")
    args = ap.parse_args(argv)

    if args.data is None:
        data_path = DatasetResolver().resolve("id_card_kyc_parser", force=args.force_download)
        data_dir = str(data_path)
        print(f"[INFO] Resolved dataset → {data_path}")
    else:
        data_dir = args.data

    print(f"[INFO] Dataset ready at {data_dir}")
    print("[INFO] PaddleOCR is pre-trained; evaluating extraction on dataset images.")
    print(f"[INFO] Template: {args.template}")

    try:
        from modern import IDCardKYCParser

        proj = IDCardKYCParser()
        proj.setup(template=args.template)

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
            det = out["detection"]
            entry = {
                "file": img_path.name,
                "card_detected": det.found,
                "template": result.template_used,
                "num_blocks": result.num_blocks,
                "fields_found": len(result.fields),
                "fields": list(result.fields.keys()),
                "valid": report.valid,
                "warnings": len(report.warnings),
            }
            results.append(entry)
            fields_str = ", ".join(
                f"{n}={ef.value}" for n, ef in result.fields.items()
            )
            card_s = "Y" if det.found else "N"
            print(
                f"  {img_path.name}: card={card_s}, "
                f"{result.num_blocks} blocks, fields=[{fields_str}]"
            )

        # Summary
        out_path = Path(__file__).parent / "runs" / "eval_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

        n_detected = sum(1 for r in results if r["card_detected"])
        avg_fields = sum(r["fields_found"] for r in results) / max(len(results), 1)
        print(f"\n[INFO] Evaluated {len(results)} cards")
        print(f"[INFO] Cards detected: {n_detected}/{len(results)}")
        print(f"[INFO] Avg fields extracted: {avg_fields:.1f}")
        print(f"[INFO] Results saved to {out_path}")

    except ImportError as e:
        print(f"[WARN] Could not run evaluation: {e}")
        print("[INFO] Install paddleocr: pip install paddleocr paddlepaddle")


if __name__ == "__main__":
    main()
