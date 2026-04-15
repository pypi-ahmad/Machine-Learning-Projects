"""CLI inference pipeline for Exam Sheet Parser.

Supports single images and directories of exam sheet scans.

Usage::

    python infer.py --source exam.jpg
    python infer.py --source scans/ --export-json results.json
    python infer.py --source test.png --no-display --save-annotated
    python infer.py --source exams/ --export-csv questions.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ExamSheetConfig, load_config
from export import ExamSheetExporter
from parser import ExamSheetPipeline
from validator import ExamSheetValidator
from visualize import draw_overlay

log = logging.getLogger("exam_sheet.infer")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exam Sheet Parser -- Inference",
    )
    p.add_argument("--source", required=True,
                   help="Image path or directory of exam sheet scans")
    p.add_argument("--config", default=None, help="Path to YAML/JSON config")
    p.add_argument("--lang", default=None, help="OCR language (default: en)")
    p.add_argument("--gpu", action="store_true", help="Enable GPU for OCR")
    p.add_argument("--no-display", action="store_true", help="Disable GUI window")
    p.add_argument("--export-json", default=None, help="JSON export path")
    p.add_argument("--export-csv", default=None, help="CSV export path")
    p.add_argument("--save-annotated", action="store_true",
                   help="Save annotated images")
    p.add_argument("--output-dir", default="output", help="Output directory")
    p.add_argument("--force-download", action="store_true",
                   help="Force dataset re-download")
    return p.parse_args(argv)


def _collect_images(source: str) -> list[Path]:
    p = Path(source)
    if p.is_dir():
        files = []
        for ext in IMAGE_EXTS:
            files.extend(p.glob(f"*{ext}"))
        files.sort()
        return files
    if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
        return [p]
    return []


def _apply_cli_overrides(
    cfg: ExamSheetConfig, args: argparse.Namespace,
) -> None:
    if args.lang:
        cfg.ocr_lang = args.lang
    if args.gpu:
        cfg.use_gpu = True
    if args.no_display:
        cfg.show_display = False
    if args.export_json:
        cfg.export_json = args.export_json
    if args.export_csv:
        cfg.export_csv = args.export_csv
    if args.save_annotated:
        cfg.save_annotated = True
    if args.output_dir:
        cfg.output_dir = args.output_dir


def run(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    cfg = load_config(args.config) if args.config else ExamSheetConfig()
    _apply_cli_overrides(cfg, args)

    if args.force_download:
        from data_bootstrap import ensure_exam_sheet_dataset
        ensure_exam_sheet_dataset(force=True)

    pipeline = ExamSheetPipeline(cfg)
    validator = ExamSheetValidator(cfg)

    images = _collect_images(args.source)
    if not images:
        log.error("No images found at: %s", args.source)
        return

    out_dir = Path(cfg.output_dir)

    with ExamSheetExporter(cfg) as exporter:
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                log.warning("Cannot read: %s", img_path)
                continue

            _process_image(
                img, img_path.name, pipeline, validator, exporter, cfg,
                out_dir,
            )

    log.info("Done -- processed %d exam sheet(s)", len(images))


def _process_image(
    image: np.ndarray,
    label: str,
    pipeline: ExamSheetPipeline,
    validator: ExamSheetValidator,
    exporter: ExamSheetExporter,
    cfg: ExamSheetConfig,
    out_dir: Path,
) -> None:
    """Process a single exam sheet image."""
    # 1. Parse
    result, blocks, elements = pipeline.process_with_details(image)

    # 2. Validate
    report = validator.validate(result)

    # 3. Export
    exporter.write(result, report=report, source=label)

    # 4. Log
    q_summary = [f"Q{q.number}" for q in result.questions[:5]]
    marks_str = f", total={result.total_marks}m" if result.total_marks else ""
    log.info(
        "%s: %d blocks, %d questions%s, conf=%.2f -- %s",
        label,
        result.num_blocks,
        result.num_questions,
        marks_str,
        result.mean_confidence,
        q_summary or "(no questions)",
    )
    if report.warnings:
        for w in report.warnings:
            log.warning("  %s: %s", w.field_name, w.message)

    # 5. Visualize
    vis = draw_overlay(image, result, cfg, elements=elements)

    if cfg.save_annotated:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"annotated_{label}"
        cv2.imwrite(str(out_path), vis)
        log.info("  Annotated -> %s", out_path)

    if cfg.show_display:
        cv2.imshow(f"Exam Sheet: {label}", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
