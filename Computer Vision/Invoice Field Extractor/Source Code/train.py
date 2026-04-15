"""Train / evaluate Invoice Field Extractor.

This project uses PaddleOCR (pre-trained) + rule-based field parsing,
so there is no model training.  This script downloads the dataset and
runs the OCR + extraction pipeline on sample images to evaluate quality.

Usage::

    python train.py
    python train.py --force-download
    python train.py --max-samples 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_bootstrap import ensure_invoice_dataset

log = logging.getLogger("invoice_extractor.train")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Invoice Field Extractor")
    p.add_argument("--max-samples", type=int, default=20, help="Max images to evaluate")
    p.add_argument("--force-download", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    args = _parse_args(argv)

    # 1. Ensure dataset
    data_root = ensure_invoice_dataset(force=args.force_download)
    log.info("Dataset at %s", data_root)

    # 2. Collect images
    images: list[Path] = []
    for ext in IMAGE_EXTS:
        images.extend(data_root.rglob(f"*{ext}"))
    images.sort()

    if not images:
        log.error("No images found in %s", data_root)
        sys.exit(1)

    images = images[: args.max_samples]
    log.info("Evaluating on %d images", len(images))

    # 3. Run OCR + extraction
    try:
        import cv2
        from config import InvoiceConfig
        from ocr_engine import OCREngine
        from parser import InvoiceParser
        from validator import InvoiceValidator

        cfg = InvoiceConfig()
        engine = OCREngine(cfg)
        parser_inst = InvoiceParser()
        validator = InvoiceValidator(cfg)

        results = []
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            blocks = engine.run(img)
            result = parser_inst.parse(blocks)
            report = validator.validate(result)

            field_names = list(result.fields.keys())
            results.append({
                "file": img_path.name,
                "num_blocks": len(blocks),
                "fields_found": len(result.fields),
                "field_names": field_names,
                "line_items": len(result.line_items),
                "valid": report.valid,
                "warnings": len(report.warnings),
            })
            log.info("  %s: %d blocks, %d fields [%s], %d line items",
                     img_path.name, len(blocks), len(field_names),
                     ", ".join(field_names), len(result.line_items))

        # 4. Save results
        out_path = Path(__file__).resolve().parent / "runs" / "eval_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        log.info("Evaluation results saved to %s", out_path)

        # 5. Summary
        total_fields = sum(r["fields_found"] for r in results)
        avg_fields = total_fields / len(results) if results else 0
        log.info("Summary: %d images, avg %.1f fields/image", len(results), avg_fields)

    except ImportError as exc:
        log.error("Cannot run evaluation: %s", exc)
        log.info("Install paddleocr: pip install paddleocr paddlepaddle")


if __name__ == "__main__":
    main()
