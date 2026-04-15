"""CLI inference pipeline for Handwritten Note to Markdown.

Supports single images and directories for batch processing.

Usage::

    python infer.py --source note.jpg
    python infer.py --source notes/ --export-md output.md
    python infer.py --source note.jpg --export-txt note.txt --no-display
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import NoteConfig, load_config
from export import NoteExporter
from parser import NoteParser
from validator import NoteValidator
from visualize import draw_overlay

log = logging.getLogger("handwritten_note.infer")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Handwritten Note to Markdown -- Inference",
    )
    p.add_argument("--source", required=True, help="Image path or directory")
    p.add_argument("--config", default=None, help="Path to YAML/JSON config")
    p.add_argument("--model", default=None,
                   help="TrOCR model name (default: microsoft/trocr-base-handwritten)")
    p.add_argument("--gpu", action="store_true", help="Enable GPU")
    p.add_argument("--no-segment", action="store_true",
                   help="Disable line segmentation (single-line images)")
    p.add_argument("--no-display", action="store_true", help="Disable GUI window")
    p.add_argument("--export-md", default=None, help="Markdown export path")
    p.add_argument("--export-txt", default=None, help="Plain text export path")
    p.add_argument("--export-json", default=None, help="JSON export path")
    p.add_argument("--save-annotated", action="store_true",
                   help="Save annotated images")
    p.add_argument("--output-dir", default="output", help="Output directory")
    p.add_argument("--confidence", action="store_true",
                   help="Include confidence annotations in Markdown")
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

    cfg = load_config(args.config) if args.config else NoteConfig()

    # CLI overrides
    if args.model:
        cfg.model_name = args.model
    if args.gpu:
        cfg.use_gpu = True
    if args.no_segment:
        cfg.enable_segmentation = False
    if args.no_display:
        cfg.show_display = False
    if args.export_md:
        cfg.export_md = args.export_md
    if args.export_txt:
        cfg.export_txt = args.export_txt
    if args.export_json:
        cfg.export_json = args.export_json
    if args.save_annotated:
        cfg.save_annotated = True
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.confidence:
        cfg.show_confidence = True

    if args.force_download:
        from data_bootstrap import ensure_handwriting_dataset
        ensure_handwriting_dataset(force=True)

    parser = NoteParser(cfg)
    validator = NoteValidator(cfg)

    sources = _collect_sources(args.source)
    if not sources:
        log.error("No images found at: %s", args.source)
        return

    with NoteExporter(cfg) as exporter:
        for src_path in sources:
            img = cv2.imread(str(src_path))
            if img is None:
                log.warning("Cannot read: %s", src_path)
                continue
            _process_image(
                img, src_path.name, parser, validator, exporter, cfg,
            )

    log.info("Done -- processed %d note(s)", len(sources))


def _process_image(
    image,
    label: str,
    parser: NoteParser,
    validator: NoteValidator,
    exporter: NoteExporter,
    cfg: NoteConfig,
) -> None:
    """Process a single handwritten note image."""
    # 1. Parse (segment + recognise + format)
    result = parser.parse(image)

    # 2. Validate
    report = validator.validate(result)

    # 3. Export
    exporter.write(result, report=report, source=label)

    # 4. Log
    nonempty = sum(1 for ln in result.lines if ln.text.strip())
    log.info(
        "%s: %d lines (%d non-empty), mean_conf=%.2f",
        label, result.num_lines, nonempty, result.mean_confidence,
    )
    if report.warnings:
        for w in report.warnings:
            log.warning("  %s: %s", w.field_name, w.message)

    # 5. Print recognised text
    print(f"\n{'='*60}")
    print(f"Source: {label}")
    print(f"{'='*60}")
    print(result.markdown)

    # 6. Visualize
    vis = draw_overlay(image, result, cfg)

    if cfg.save_annotated:
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"annotated_{label}"
        cv2.imwrite(str(out_path), vis)
        log.info("  Annotated -> %s", out_path)

    if cfg.show_display:
        cv2.imshow(f"Note: {label}", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
