"""CLI inference for Blink Headpose Analyzer.

Supports webcam, video, and single-image inputs with optional
per-frame CSV/JSON export.

Usage::

    python infer.py --source 0                    # webcam
    python infer.py --source video.mp4            # video file
    python infer.py --source face.jpg             # single image
    python infer.py --source 0 --export-csv log.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyzer import AnalyzerPipeline
from config import AnalyzerConfig, load_config
from export import FrameExporter
from validator import AnalyzerValidator
from visualize import draw_overlay

log = logging.getLogger("blink_headpose.infer")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Blink Headpose Analyzer -- Inference",
    )
    p.add_argument("--source", required=True,
                   help="'0' for webcam, or path to video/image")
    p.add_argument("--config", default=None,
                   help="Path to YAML/JSON config file")
    p.add_argument("--ear-threshold", type=float, default=None,
                   help="Override EAR threshold")
    p.add_argument("--yaw-threshold", type=float, default=None,
                   help="Override yaw threshold (degrees)")
    p.add_argument("--no-display", action="store_true",
                   help="Disable GUI windows")
    p.add_argument("--export-csv", default=None,
                   help="CSV export path for per-frame metrics")
    p.add_argument("--export-json", default=None,
                   help="JSON export path for per-frame metrics")
    p.add_argument("--save-annotated", action="store_true",
                   help="Save annotated frames to output dir")
    p.add_argument("--output-dir", default="output",
                   help="Output directory (default: output/)")
    p.add_argument("--force-download", action="store_true",
                   help="Force dataset re-download")
    return p.parse_args(argv)


def _apply_overrides(cfg: AnalyzerConfig, args: argparse.Namespace) -> None:
    if args.ear_threshold is not None:
        cfg.ear_threshold = args.ear_threshold
    if args.yaw_threshold is not None:
        cfg.yaw_threshold = args.yaw_threshold
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

    cfg = load_config(args.config) if args.config else AnalyzerConfig()
    _apply_overrides(cfg, args)

    if args.force_download:
        from data_bootstrap import ensure_blink_headpose_dataset
        ensure_blink_headpose_dataset(force=True)

    pipeline = AnalyzerPipeline(cfg)
    pipeline.load()
    validator = AnalyzerValidator(cfg)

    source = args.source
    if source.isdigit():
        _run_stream(pipeline, validator, cfg, int(source), "Webcam")
    elif Path(source).suffix.lower() in VIDEO_EXTS:
        _run_stream(pipeline, validator, cfg, source, Path(source).name)
    elif Path(source).suffix.lower() in IMAGE_EXTS:
        _run_image(pipeline, validator, cfg, source)
    else:
        log.error("Unsupported source: %s", source)


def _run_stream(
    pipeline: AnalyzerPipeline,
    validator: AnalyzerValidator,
    cfg: AnalyzerConfig,
    source: int | str,
    label: str,
) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error("Cannot open source: %s", source)
        return

    log.info("Processing stream: %s -- press 'q' to quit", label)
    frame_idx = 0
    out_dir = Path(cfg.output_dir)

    with FrameExporter(cfg) as exporter:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = pipeline.process(frame)
            validator.validate(result)
            exporter.write(result, frame_idx=frame_idx)

            if cfg.show_display:
                vis = draw_overlay(frame, result, cfg)
                cv2.imshow(f"Blink Headpose -- {label}", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if cfg.save_annotated and frame_idx % 30 == 0:
                out_dir.mkdir(parents=True, exist_ok=True)
                vis = draw_overlay(frame, result, cfg)
                cv2.imwrite(
                    str(out_dir / f"frame_{frame_idx:06d}.jpg"), vis,
                )

            frame_idx += 1

    cap.release()
    if cfg.show_display:
        cv2.destroyAllWindows()

    log.info(
        "Done — %d frames, %d blinks detected",
        frame_idx, pipeline.blink_counter._total_blinks,
    )


def _run_image(
    pipeline: AnalyzerPipeline,
    validator: AnalyzerValidator,
    cfg: AnalyzerConfig,
    source: str,
) -> None:
    img = cv2.imread(source)
    if img is None:
        log.error("Cannot read: %s", source)
        return

    result = pipeline.process(img)
    validator.validate(result)

    log.info(
        "%s: face=%s, EAR=%.2f, yaw=%.1f, pitch=%.1f",
        source, result.face_detected, result.blink.ear,
        result.head_pose.yaw, result.head_pose.pitch,
    )

    vis = draw_overlay(img, result, cfg)

    if cfg.save_annotated:
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"annotated_{Path(source).name}"), vis)

    if cfg.show_display:
        cv2.imshow("Blink Headpose Analyzer", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
