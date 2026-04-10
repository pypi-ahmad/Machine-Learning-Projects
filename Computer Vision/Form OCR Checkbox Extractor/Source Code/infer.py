"""CLI inference pipeline for Form OCR Checkbox Extractor.

Supports single images and directories for batch processing.

Usage::

    python infer.py --source form.jpg
    python infer.py --source forms/ --export-json out.json
    python infer.py --source form.jpg --no-display --save-annotated
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import FormCheckboxConfig, load_config
from export import FormExporter
from parser import FormParser
from validator import FormValidator
from visualize import draw_overlay

log = logging.getLogger("form_checkbox.infer")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Form OCR Checkbox Extractor — Inference")
    p.add_argument("--source", required=True, help="Image path or directory")
    p.add_argument("--config", default=None, help="Path to YAML/JSON config")
    p.add_argument("--lang", default=None, help="OCR language (default: en)")
    p.add_argument("--gpu", action="store_true", help="Enable GPU for OCR")
    p.add_argument("--no-display", action="store_true", help="Disable GUI window")
    p.add_argument("--show-ocr", action="store_true", help="Show OCR bounding boxes")
    p.add_argument("--export-json", default=None, help="JSON export path")
    p.add_argument("--export-csv", default=None, help="CSV export path")
    p.add_argument("--save-annotated", action="store_true", help="Save annotated images")
    p.add_argument("--output-dir", default="output", help="Output directory")
    p.add_argument("--fill-threshold", type=float, default=None,
                   help="Override fill-ratio threshold for checked state")
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

    cfg = load_config(args.config) if args.config else FormCheckboxConfig()

    # CLI overrides
    if args.lang:
        cfg.ocr_lang = args.lang
    if args.gpu:
        cfg.use_gpu = True
    if args.no_display:
        cfg.show_display = False
    if args.show_ocr:
        cfg.show_ocr_boxes = True
    if args.export_json:
        cfg.export_json = args.export_json
    if args.export_csv:
        cfg.export_csv = args.export_csv
    if args.save_annotated:
        cfg.save_annotated = True
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.fill_threshold is not None:
        cfg.fill_threshold = args.fill_threshold

    if args.force_download:
        from data_bootstrap import ensure_form_dataset
        ensure_form_dataset(force=True)

    parser = FormParser(cfg)
    validator = FormValidator(cfg)

    sources = _collect_sources(args.source)
    if not sources:
        log.error("No images found at: %s", args.source)
        return

    with FormExporter(cfg) as exporter:
        for src_path in sources:
            img = cv2.imread(str(src_path))
            if img is None:
                log.warning("Cannot read: %s", src_path)
                continue
            _process_image(img, src_path.name, parser, validator, exporter, cfg)

    log.info("Done — processed %d form(s)", len(sources))


def _process_image(
    image,
    label: str,
    parser: FormParser,
    validator: FormValidator,
    exporter: FormExporter,
    cfg: FormCheckboxConfig,
) -> None:
    """Process a single form image through the full pipeline."""
    # 1. Parse (OCR + checkbox detection)
    result, blocks, controls = parser.parse_with_details(image)

    # 2. Validate
    report = validator.validate(result)

    # 3. Export
    exporter.write(result, report=report, source=label)

    # 4. Log
    checked = [
        cb.label[:20] for cb in result.checkbox_fields if cb.state == "checked"
    ]
    text_found = list(result.text_fields.keys())
    log.info(
        "%s: %d blocks, %d checkboxes (%d checked), text=%s, checked=%s",
        label,
        result.num_ocr_blocks,
        result.num_checkboxes,
        result.num_checked,
        text_found,
        checked,
    )
    if report.warnings:
        for w in report.warnings:
            log.warning("  %s: %s", w.field_name, w.message)

    # 5. Visualize
    vis = draw_overlay(image, result, cfg, ocr_blocks=blocks, controls=controls)

    if cfg.save_annotated:
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"annotated_{label}"
        cv2.imwrite(str(out_path), vis)
        log.info("  Annotated → %s", out_path)

    if cfg.show_display:
        cv2.imshow(f"Form: {label}", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
