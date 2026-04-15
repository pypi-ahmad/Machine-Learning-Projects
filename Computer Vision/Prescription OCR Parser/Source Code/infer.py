"""CLI inference pipeline for Prescription OCR Parser.

Supports single images and directories for batch processing.

**DISCLAIMER:** This tool is for informational and educational
purposes only.  It does not provide medical advice.

Usage::

    python infer.py --source prescription.jpg
    python infer.py --source rxs/ --export-json results.json
    python infer.py --source rx.jpg --no-display --save-annotated
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PrescriptionConfig, load_config
from export import PrescriptionExporter
from parser import PrescriptionParser
from validator import PrescriptionValidator
from visualize import draw_overlay

log = logging.getLogger("prescription_ocr.infer")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

_DISCLAIMER = (
    "DISCLAIMER: This tool is for informational and educational purposes "
    "only. It does NOT provide medical advice and must NOT be used for "
    "diagnosis, treatment, or clinical decision-making. Always consult "
    "a licensed healthcare professional."
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prescription OCR Parser -- Inference",
        epilog=_DISCLAIMER,
    )
    p.add_argument("--source", required=True, help="Image path or directory")
    p.add_argument("--config", default=None, help="Path to YAML/JSON config")
    p.add_argument("--lang", default=None, help="OCR language (default: en)")
    p.add_argument("--gpu", action="store_true", help="Enable GPU for OCR")
    p.add_argument("--no-display", action="store_true", help="Disable GUI window")
    p.add_argument("--show-boxes", action="store_true",
                   help="Show OCR bounding boxes")
    p.add_argument("--export-json", default=None, help="JSON export path")
    p.add_argument("--export-csv", default=None, help="CSV export path")
    p.add_argument("--save-annotated", action="store_true",
                   help="Save annotated images")
    p.add_argument("--output-dir", default="output", help="Output directory")
    p.add_argument("--force-download", action="store_true",
                   help="Force dataset re-download")
    return p.parse_args(argv)


def _collect_sources(source: str) -> list[Path]:
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    print(f"\n{_DISCLAIMER}\n")

    cfg = load_config(args.config) if args.config else PrescriptionConfig()

    # CLI overrides
    if args.lang:
        cfg.ocr_lang = args.lang
    if args.gpu:
        cfg.use_gpu = True
    if args.no_display:
        cfg.show_display = False
    if args.show_boxes:
        cfg.show_ocr_boxes = True
    if args.export_json:
        cfg.export_json = args.export_json
    if args.export_csv:
        cfg.export_csv = args.export_csv
    if args.save_annotated:
        cfg.save_annotated = True
    if args.output_dir:
        cfg.output_dir = args.output_dir

    if args.force_download:
        from data_bootstrap import ensure_prescription_dataset
        ensure_prescription_dataset(force=True)

    parser = PrescriptionParser(cfg)
    validator = PrescriptionValidator(cfg)

    sources = _collect_sources(args.source)
    if not sources:
        log.error("No images found at: %s", args.source)
        return

    with PrescriptionExporter(cfg) as exporter:
        for src_path in sources:
            img = cv2.imread(str(src_path))
            if img is None:
                log.warning("Cannot read: %s", src_path)
                continue
            _process_image(
                img, src_path.name, parser, validator, exporter, cfg,
            )

    log.info("Done -- processed %d prescription(s)", len(sources))


def _process_image(
    image,
    label: str,
    parser: PrescriptionParser,
    validator: PrescriptionValidator,
    exporter: PrescriptionExporter,
    cfg: PrescriptionConfig,
) -> None:
    """Process a single prescription image."""
    # 1. Parse
    result, blocks = parser.parse_with_blocks(image)

    # 2. Validate
    report = validator.validate(result)

    # 3. Export
    exporter.write(result, report=report, source=label)

    # 4. Log
    med_names = [m.medicine_name[:25] for m in result.medicines]
    log.info(
        "%s: %d blocks, %d medicines=%s, conf=%.2f",
        label,
        result.num_blocks,
        result.num_medicines,
        med_names,
        result.mean_confidence,
    )
    if report.warnings:
        for w in report.warnings:
            log.warning("  %s: %s", w.field_name, w.message)

    # 5. Visualize
    vis = draw_overlay(image, result, cfg, ocr_blocks=blocks)

    if cfg.save_annotated:
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"annotated_{label}"
        cv2.imwrite(str(out_path), vis)
        log.info("  Annotated -> %s", out_path)

    if cfg.show_display:
        cv2.imshow(f"Rx: {label}", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
