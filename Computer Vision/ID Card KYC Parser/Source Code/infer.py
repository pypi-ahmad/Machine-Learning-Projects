"""CLI inference pipeline for ID Card KYC Parser.

Supports single images and directories for batch processing.

Usage::

    python infer.py --source id_card.jpg
    python infer.py --source cards/ --template passport --export-json out.json
    python infer.py --source card.jpg --no-rectify --no-display
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from card_detector import CardDetector
from config import IDCardConfig, load_config
from export import IDCardExporter
from ocr_engine import OCREngine
from parser import IDCardParser
from templates import list_templates
from validator import IDCardValidator
from visualize import draw_overlay

log = logging.getLogger("id_card_kyc.infer")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ID Card KYC Parser -- Inference")
    p.add_argument("--source", required=True, help="Image path or directory")
    p.add_argument("--config", default=None, help="Path to YAML/JSON config")
    p.add_argument("--template", default=None,
                   help=f"Template: {', '.join(list_templates())}")
    p.add_argument("--lang", default=None, help="OCR language (default: en)")
    p.add_argument("--gpu", action="store_true", help="Enable GPU for OCR")
    p.add_argument("--no-rectify", action="store_true", help="Skip perspective correction")
    p.add_argument("--no-detect", action="store_true", help="Skip card boundary detection")
    p.add_argument("--no-display", action="store_true", help="Disable GUI window")
    p.add_argument("--export-json", default=None, help="JSON export path")
    p.add_argument("--export-csv", default=None, help="CSV export path")
    p.add_argument("--save-annotated", action="store_true", help="Save annotated images")
    p.add_argument("--save-rectified", action="store_true", help="Save rectified cards")
    p.add_argument("--output-dir", default="output", help="Output directory")
    p.add_argument("--force-download", action="store_true", help="Force dataset re-download")
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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    cfg = load_config(args.config) if args.config else IDCardConfig()

    # CLI overrides
    if args.template:
        cfg.template = args.template
    if args.lang:
        cfg.ocr_lang = args.lang
    if args.gpu:
        cfg.use_gpu = True
    if args.no_rectify:
        cfg.rectify = False
    if args.no_detect:
        cfg.detect_card = False
    if args.no_display:
        cfg.show_display = False
    if args.export_json:
        cfg.export_json = args.export_json
    if args.export_csv:
        cfg.export_csv = args.export_csv
    if args.save_annotated:
        cfg.save_annotated = True
    if args.save_rectified:
        cfg.save_rectified = True
    if args.output_dir:
        cfg.output_dir = args.output_dir

    if args.force_download:
        from data_bootstrap import ensure_idcard_dataset
        ensure_idcard_dataset(force=True)

    detector = CardDetector(cfg)
    engine = OCREngine(cfg)
    parser = IDCardParser(template=cfg.template)
    validator = IDCardValidator(cfg)

    sources = _collect_sources(args.source)
    if not sources:
        log.error("No images found at: %s", args.source)
        return

    with IDCardExporter(cfg) as exporter:
        for src_path in sources:
            img = cv2.imread(str(src_path))
            if img is None:
                log.warning("Cannot read: %s", src_path)
                continue
            _process_image(
                img, src_path.name, detector, engine, parser,
                validator, exporter, cfg,
            )

    log.info("Done -- processed %d card(s)", len(sources))


def _process_image(
    image,
    label: str,
    detector: CardDetector,
    engine: OCREngine,
    parser: IDCardParser,
    validator: IDCardValidator,
    exporter: IDCardExporter,
    cfg: IDCardConfig,
) -> None:
    """Process a single ID card image through the full pipeline."""
    # 1. Detect + rectify
    det = detector.detect_and_rectify(image)
    ocr_input = det.rectified if det.rectified is not None else image

    # 2. OCR
    blocks = engine.run(ocr_input)

    # 3. Parse + validate
    result = parser.parse(blocks)
    report = validator.validate(result, card_detected=det.found)

    # 4. Export
    exporter.write(result, report=report, source=label, card_detected=det.found)

    # Log
    found = [f"{n}={ef.value}" for n, ef in result.fields.items()]
    card_str = "detected" if det.found else "not detected"
    log.info(
        "%s: card=%s, template=%s, %d blocks, fields=[%s]",
        label, card_str, result.template_used, len(blocks), ", ".join(found),
    )
    if report.warnings:
        for w in report.warnings:
            log.warning("  %s: %s", w.field_name, w.message)

    # Save rectified
    if cfg.save_rectified and det.rectified is not None:
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rect_path = out_dir / f"rectified_{label}"
        cv2.imwrite(str(rect_path), det.rectified)
        log.info("  Rectified -> %s", rect_path)

    # Visualize (on original image for boundary overlay)
    vis = draw_overlay(image, result, cfg, ocr_blocks=blocks, detection=det)

    if cfg.save_annotated:
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"annotated_{label}"
        cv2.imwrite(str(out_path), vis)
        log.info("  Annotated -> %s", out_path)

    if cfg.show_display:
        cv2.imshow(f"ID Card: {label}", vis)
        if det.rectified is not None and det.found:
            cv2.imshow(f"Rectified: {label}", det.rectified)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
