"""CLI inference pipeline for Number Plate Reader Pro.

Supports images, directories, video files, and live webcam.

Usage::

    python infer.py --source photo.jpg
    python infer.py --source images/ --export-json results.json
    python infer.py --source traffic.mp4 --export-csv plates.csv
    python infer.py --source 0 --no-display   # webcam (index 0)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PlateConfig, load_config
from export import PlateExporter
from parser import PlateReaderPipeline
from validator import PlateValidator
from visualize import draw_overlay

log = logging.getLogger("plate_reader.infer")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Number Plate Reader Pro -- Inference",
    )
    p.add_argument("--source", required=True,
                   help="Image path, directory, video file, or webcam index (integer)")
    p.add_argument("--config", default=None, help="Path to YAML/JSON config")
    p.add_argument("--model", default=None, help="Optional detector weights override")
    p.add_argument("--gpu", action="store_true", help="Enable GPU for OCR")
    p.add_argument("--no-display", action="store_true", help="Disable GUI window")
    p.add_argument("--export-json", default=None, help="JSON export path")
    p.add_argument("--export-csv", default=None, help="CSV export path")
    p.add_argument("--save-annotated", action="store_true",
                   help="Save annotated images/frames")
    p.add_argument("--save-crops", action="store_true",
                   help="Save detected plate crops")
    p.add_argument("--output-dir", default="output", help="Output directory")
    p.add_argument("--confidence", type=float, default=None,
                   help="Override detection confidence threshold")
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


def _apply_cli_overrides(cfg: PlateConfig, args: argparse.Namespace) -> None:
    if args.model:
        cfg.det_model = args.model
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
    if args.save_crops:
        cfg.save_crops = True
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.confidence is not None:
        cfg.det_confidence = args.confidence


def run(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    cfg = load_config(args.config) if args.config else PlateConfig()
    _apply_cli_overrides(cfg, args)

    if args.force_download:
        from data_bootstrap import ensure_plate_dataset
        ensure_plate_dataset(force=True)

    pipeline = PlateReaderPipeline(cfg)
    validator = PlateValidator(cfg)

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
    pipeline: PlateReaderPipeline,
    validator: PlateValidator,
    cfg: PlateConfig,
) -> None:
    images = _collect_images(source)
    if not images:
        log.error("No images found at: %s", source)
        return

    out_dir = Path(cfg.output_dir)

    with PlateExporter(cfg) as exporter:
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                log.warning("Cannot read: %s", img_path)
                continue

            result = pipeline.process_frame(img)
            report = validator.validate(result)
            exporter.write(result, report=report, source=img_path.name)

            _log_result(img_path.name, result, report)
            _save_outputs(img, img_path.name, result, cfg, out_dir)

            if cfg.show_display:
                vis = draw_overlay(img, result, cfg)
                cv2.imshow(f"Plate Reader: {img_path.name}", vis)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    log.info("Done -- processed %d image(s)", len(images))


# ------------------------------------------------------------------
# Video / webcam mode
# ------------------------------------------------------------------


def _run_video(
    source,
    pipeline: PlateReaderPipeline,
    validator: PlateValidator,
    cfg: PlateConfig,
    *,
    is_webcam: bool = False,
) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error("Cannot open video source: %s", source)
        return

    src_label = f"webcam:{source}" if is_webcam else Path(str(source)).name
    log.info("Processing video: %s (press 'q' to quit)", src_label)

    out_dir = Path(cfg.output_dir)
    frame_idx = 0

    with PlateExporter(cfg) as exporter:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            result = pipeline.process_frame(frame)
            report = validator.validate(result)
            exporter.write(result, report=report, source=src_label)

            # Log only frames with new plates
            if result.num_new > 0:
                _log_result(f"{src_label}:f{frame_idx}", result, report)

            # Save crops for new plates
            if cfg.save_crops:
                _save_crops(result, out_dir, frame_idx)

            # Display
            if cfg.show_display:
                vis = draw_overlay(frame, result, cfg)
                cv2.imshow("Number Plate Reader Pro", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log.info("Quit signal received")
                    break

    cap.release()
    if cfg.show_display:
        cv2.destroyAllWindows()
    log.info("Done -- processed %d frames from %s", frame_idx, src_label)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _log_result(label: str, result, report) -> None:
    plates = [r.plate_text for r in result.reads if r.is_new and r.plate_text]
    log.info(
        "%s: %d detected, %d valid, %d new -- %s",
        label, result.num_detections, result.num_valid, result.num_new,
        plates or "(none)",
    )
    if report.warnings:
        for w in report.warnings:
            log.warning("  %s: %s", w.field_name, w.message)


def _save_outputs(
    image: np.ndarray,
    label: str,
    result,
    cfg: PlateConfig,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.save_annotated:
        vis = draw_overlay(image, result, cfg)
        out_path = out_dir / f"annotated_{label}"
        cv2.imwrite(str(out_path), vis)
        log.info("  Annotated -> %s", out_path)

    if cfg.save_crops:
        _save_crops(result, out_dir, result.frame_index)


def _save_crops(result, out_dir: Path, frame_idx: int) -> None:
    crop_dir = out_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    for i, read in enumerate(result.reads):
        if read.is_new and read.plate_text:
            safe_text = read.plate_text.replace(" ", "_")[:20]
            crop_path = crop_dir / f"f{frame_idx}_{i}_{safe_text}.jpg"
            cv2.imwrite(str(crop_path), read.crop)


if __name__ == "__main__":
    run()
