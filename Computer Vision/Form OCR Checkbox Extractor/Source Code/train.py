"""Train / Evaluate Form OCR Checkbox Extractor.

This project uses PaddleOCR (pre-trained) and OpenCV-based checkbox
detection, so training is optional.  This script downloads the
dataset and evaluates the full pipeline on sample images.

Usage::

    python train.py
    python train.py --data path/to/dataset
    python train.py --force-download
    python train.py --max-samples 50
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
    ap = argparse.ArgumentParser(
        description="Train/Evaluate Form OCR Checkbox Extractor",
    )
    ap.add_argument("--data", type=str, default=None, help="Path to form dataset")
    ap.add_argument("--max-samples", type=int, default=20,
                    help="Max samples to evaluate")
    ap.add_argument("--force-download", action="store_true",
                    help="Force re-download dataset")
    ap.add_argument("--fill-threshold", type=float, default=None,
                    help="Override fill-ratio threshold")
    args = ap.parse_args(argv)

    if args.data is None:
        data_path = DatasetResolver().resolve(
            "form_ocr_checkbox_extractor", force=args.force_download,
        )
        data_dir = str(data_path)
        print(f"[INFO] Resolved dataset → {data_path}")
    else:
        data_dir = args.data

    print(f"[INFO] Dataset ready at {data_dir}")
    print("[INFO] PaddleOCR is pre-trained; evaluating checkbox + OCR on form images.")

    try:
        from modern import FormOCRCheckboxExtractor

        proj = FormOCRCheckboxExtractor()
        setup_kwargs: dict = {}
        if args.fill_threshold is not None:
            setup_kwargs["fill_threshold"] = args.fill_threshold
        proj.setup(**setup_kwargs)

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
                "ocr_blocks": result.num_ocr_blocks,
                "checkboxes": result.num_checkboxes,
                "checked": result.num_checked,
                "text_fields": list(result.text_fields.keys()),
                "valid": report.valid,
                "warnings": len(report.warnings),
            }
            results.append(entry)
            checked_labels = [
                cb.label[:20]
                for cb in result.checkbox_fields
                if cb.state == "checked"
            ]
            print(
                f"  {img_path.name}: "
                f"{result.num_ocr_blocks} blocks, "
                f"{result.num_checkboxes} checkboxes "
                f"({result.num_checked} checked), "
                f"text={list(result.text_fields.keys())}, "
                f"checked={checked_labels}"
            )

        # Summary
        out_path = Path(__file__).parent / "runs" / "eval_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

        total_cb = sum(r["checkboxes"] for r in results)
        total_checked = sum(r["checked"] for r in results)
        avg_blocks = sum(r["ocr_blocks"] for r in results) / max(len(results), 1)
        print(f"\n[INFO] Evaluated {len(results)} forms")
        print(f"[INFO] Total checkboxes: {total_cb} ({total_checked} checked)")
        print(f"[INFO] Avg OCR blocks: {avg_blocks:.1f}")
        print(f"[INFO] Results saved to {out_path}")

    except ImportError as e:
        print(f"[WARN] Could not run evaluation: {e}")
        print("[INFO] Install paddleocr: pip install paddleocr paddlepaddle")


if __name__ == "__main__":
    main()
