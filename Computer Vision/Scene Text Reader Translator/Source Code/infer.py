"""CLI inference pipeline for Scene Text Reader Translator.

Supports images, directories, video files, and live webcam.

Usage::

    python infer.py --source street_sign.jpg
    python infer.py --source images/ --export-json results.json
    python infer.py --source city.mp4 --export-csv texts.csv
    python infer.py --source 0                         # webcam
    python infer.py --source sign.jpg --translate --target-lang es
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import SceneTextConfig, load_config
from export import SceneTextExporter
from parser import SceneTextPipeline
from validator import SceneTextValidator
from visualize import draw_overlay

log = logging.getLogger("scene_text.infer")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scene Text Reader Translator — Inference",
    )
    p.add_argument("--source", required=True,
                   help="Image path, directory, video file, or webcam index (integer)")
    p.add_argument("--config", default=None, help="Path to YAML/JSON config")
    p.add_argument("--lang", default=None, help="OCR language (default: en)")
    p.add_argument("--gpu", action="store_true", help="Enable GPU for OCR")
    p.add_argument("--no-display", action="store_true", help="Disable GUI window")
    p.add_argument("--show-boxes", action="store_true",
                   help="Show OCR bounding boxes (default: on)")
    p.add_argument("--export-json", default=None, help="JSON export path")
    p.add_argument("--export-csv", default=None, help="CSV export path")
    p.add_argument("--save-annotated", action="store_true",
                   help="Save annotated images/frames")
    p.add_argument("--output-dir", default="output", help="Output directory")
    p.add_argument("--translate", action="store_true",
                   help="Enable translation")
    p.add_argument("--target-lang", default=None,
                   help="Translation target language (e.g. es, fr, de)")
    p.add_argument("--translate-provider", default=None,
                   help="Translation provider (e.g. googletrans)")
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


def _is_video(source: str) -> bool:
    return Path(source).suffix.lower() in VIDEO_EXTS


def _is_webcam(source: str) -> bool:
    return source.isdigit()


def _apply_cli_overrides(cfg: SceneTextConfig, args: argparse.Namespace) -> None:
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
    if args.translate:
        cfg.translate_enabled = True
    if args.target_lang:
        cfg.translate_target_lang = args.target_lang
        cfg.translate_enabled = True
    if args.translate_provider:
        cfg.translate_provider = args.translate_provider
        cfg.translate_enabled = True


def run(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    cfg = load_config(args.config) if args.config else SceneTextConfig()
    _apply_cli_overrides(cfg, args)

    if args.force_download:
        from data_bootstrap import ensure_scene_text_dataset
        ensure_scene_text_dataset(force=True)

    pipeline = SceneTextPipeline(cfg)
    validator = SceneTextValidator(cfg)

    source = args.source

    if _is_webcam(source):
        _run_video(int(source), pipeline, validator, cfg, is_webcam=True)
    elif _is_video(source):
        _run_video(source, pipeline, validator, cfg, is_webcam=False)
    else:
        _run_images(source, pipeline, validator, cfg)


# ------------------------------------------------------------------
# Image mode
# ------------------------------------------------------------------


def _run_images(
    source: str,
    pipeline: SceneTextPipeline,
    validator: SceneTextValidator,
    cfg: SceneTextConfig,
) -> None:
    images = _collect_images(source)
    if not images:
        log.error("No images found at: %s", source)
        return

    out_dir = Path(cfg.output_dir)

    with SceneTextExporter(cfg) as exporter:
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                log.warning("Cannot read: %s", img_path)
                continue

            result, blocks = pipeline.process_with_blocks(img)
            report = validator.validate(result)
            exporter.write(result, report=report, source=img_path.name)

            _log_result(img_path.name, result, report)
            _save_outputs(img, img_path.name, result, cfg, out_dir, blocks)

            if cfg.show_display:
                vis = draw_overlay(img, result, cfg, ocr_blocks=blocks)
                cv2.imshow(f"Scene Text: {img_path.name}", vis)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    log.info("Done — processed %d image(s)", len(images))


# ------------------------------------------------------------------
# Video / webcam mode
# ------------------------------------------------------------------


def _run_video(
    source,
    pipeline: SceneTextPipeline,
    validator: SceneTextValidator,
    cfg: SceneTextConfig,
    *,
    is_webcam: bool = False,
) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error("Cannot open video source: %s", source)
        return

    src_label = f"webcam:{source}" if is_webcam else Path(str(source)).name
    log.info("Processing video: %s (press 'q' to quit)", src_label)

    frame_idx = 0

    with SceneTextExporter(cfg) as exporter:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            result, blocks = pipeline.process_with_blocks(frame)
            report = validator.validate(result)
            exporter.write(result, report=report, source=src_label)

            if result.num_blocks > 0:
                _log_result(f"{src_label}:f{frame_idx}", result, report)

            if cfg.show_display:
                vis = draw_overlay(frame, result, cfg, ocr_blocks=blocks)
                cv2.imshow("Scene Text Reader", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log.info("Quit signal received")
                    break

    cap.release()
    if cfg.show_display:
        cv2.destroyAllWindows()
    log.info("Done — processed %d frames from %s", frame_idx, src_label)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _log_result(label: str, result, report) -> None:
    texts = [r.text[:25] for r in result.reads[:5]]
    log.info(
        "%s: %d blocks, conf=%.2f — %s",
        label, result.num_blocks, result.mean_confidence,
        texts or "(no text)",
    )
    if report.warnings:
        for w in report.warnings:
            log.warning("  %s: %s", w.field_name, w.message)


def _save_outputs(
    image: np.ndarray,
    label: str,
    result,
    cfg: SceneTextConfig,
    out_dir: Path,
    blocks=None,
) -> None:
    if cfg.save_annotated:
        out_dir.mkdir(parents=True, exist_ok=True)
        vis = draw_overlay(image, result, cfg, ocr_blocks=blocks)
        out_path = out_dir / f"annotated_{label}"
        cv2.imwrite(str(out_path), vis)
        log.info("  Annotated → %s", out_path)


if __name__ == "__main__":
    run()
