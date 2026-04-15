"""CLI inference pipeline for Receipt Digitizer.

Supports single images, directories of images, and batch processing.

Usage::

    python infer.py --source receipt.jpg
    python infer.py --source receipts/ --export-json results.json
    python infer.py --source receipt.jpg --export-csv out.csv --no-display
    python infer.py --source receipts/ --save-annotated --output-dir output
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ReceiptConfig, load_config
from export import ReceiptExporter
from ocr_engine import OCREngine
from parser import ReceiptParser
from preprocess import preprocess_receipt
from validator import ReceiptValidator
from visualize import draw_overlay

log = logging.getLogger("receipt_digitizer.infer")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Receipt Digitizer -- Inference")
    p.add_argument("--source", required=True, help="Image path or directory")
    p.add_argument("--config", default=None, help="Path to YAML/JSON config")
    p.add_argument("--lang", default=None, help="OCR language (default: en)")
    p.add_argument("--gpu", action="store_true", help="Enable GPU for OCR")
    p.add_argument("--no-display", action="store_true", help="Disable GUI window")
    p.add_argument("--no-preprocess", action="store_true", help="Skip preprocessing")
    p.add_argument("--export-json", default=None, help="JSON export path")
    p.add_argument("--export-csv", default=None, help="CSV export path")
    p.add_argument("--save-annotated", action="store_true", help="Save annotated images")
    p.add_argument("--output-dir", default="output", help="Output directory")
    p.add_argument("--force-download", action="store_true", help="Force dataset re-download")
    return p.parse_args(argv)


def _collect_sources(source: str) -> list[Path]:
    """Resolve *source* to a list of image paths."""
    p = Path(source)
    if p.is_dir():
        files = []
        for ext in IMAGE_EXTS:
            files.extend(p.glob(f"*{ext}"))
        files.sort()
        return files
    if p.is_file():
        return [p]
    log.error("Source not found: %s", source)
    return []


def run(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    cfg = load_config(args.config) if args.config else ReceiptConfig()

    # CLI overrides
    if args.lang:
        cfg.ocr_lang = args.lang
    if args.gpu:
        cfg.use_gpu = True
    if args.no_display:
        cfg.show_display = False
    if args.no_preprocess:
        cfg.denoise = False
        cfg.deskew = False
        cfg.sharpen = False
        cfg.binarize = False
    if args.export_json:
        cfg.export_json = args.export_json
    if args.export_csv:
        cfg.export_csv = args.export_csv
    if args.save_annotated:
        cfg.save_annotated = True
    if args.output_dir:
        cfg.output_dir = args.output_dir

    if args.force_download:
        from data_bootstrap import ensure_receipt_dataset
        ensure_receipt_dataset(force=True)

    engine = OCREngine(cfg)
    parser = ReceiptParser()
    validator = ReceiptValidator(cfg)

    sources = _collect_sources(args.source)
    if not sources:
        log.error("No images found at: %s", args.source)
        return

    with ReceiptExporter(cfg) as exporter:
        for src_path in sources:
            img = cv2.imread(str(src_path))
            if img is None:
                log.warning("Cannot read: %s", src_path)
                continue
            _process_image(
                img, src_path.name, engine, parser, validator, exporter, cfg,
            )

    log.info("Done -- processed %d receipt(s)", len(sources))


def _process_image(
    image,
    label: str,
    engine: OCREngine,
    parser: ReceiptParser,
    validator: ReceiptValidator,
    exporter: ReceiptExporter,
    cfg: ReceiptConfig,
) -> None:
    """Process a single receipt image through the full pipeline."""
    # Preprocess
    cleaned = preprocess_receipt(image, cfg)

    # OCR
    blocks = engine.run(cleaned)

    # Parse + validate
    result = parser.parse(blocks)
    report = validator.validate(result)

    # Export
    exporter.write(result, report=report, source=label)

    # Log summary
    found = [f"{n}={ef.value}" for n, ef in result.fields.items()]
    log.info("%s: %d blocks, fields=[%s]", label, len(blocks), ", ".join(found))
    if result.line_items:
        log.info("  %d line items extracted", len(result.line_items))
    if report.warnings:
        for w in report.warnings:
            log.warning("  %s: %s", w.field_name, w.message)

    # Visualize
    vis = draw_overlay(image, result, cfg, ocr_blocks=blocks)

    if cfg.save_annotated:
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"annotated_{label}"
        cv2.imwrite(str(out_path), vis)
        log.info("  Saved -> %s", out_path)

    if cfg.show_display:
        cv2.imshow(f"Receipt: {label}", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
