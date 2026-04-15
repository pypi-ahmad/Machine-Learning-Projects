"""CLI inference pipeline for Driver Drowsiness Monitor.

Supports webcam and video file inputs for real-time drowsiness
detection with EAR, MAR, head pose, and alert logging.

Usage::

    python infer.py --source 0                      # webcam
    python infer.py --source driving_video.mp4      # video
    python infer.py --source 0 --export-csv log.csv
    python infer.py --source video.mp4 --no-display --export-json metrics.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DrowsinessConfig, load_config
from export import DrowsinessExporter
from parser import DrowsinessPipeline
from validator import DrowsinessValidator
from visualize import draw_overlay

log = logging.getLogger("drowsiness.infer")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Driver Drowsiness Monitor -- Inference",
    )
    p.add_argument("--source", required=True,
                   help="'0' for webcam, path to video file, or image")
    p.add_argument("--config", default=None,
                   help="Path to YAML/JSON config")
    p.add_argument("--ear-threshold", type=float, default=None,
                   help="EAR threshold for blink detection")
    p.add_argument("--mar-threshold", type=float, default=None,
                   help="MAR threshold for yawn detection")
    p.add_argument("--yaw-threshold", type=float, default=None,
                   help="Yaw threshold for distraction")
    p.add_argument("--no-display", action="store_true",
                   help="Disable GUI windows")
    p.add_argument("--export-json", default=None,
                   help="JSON export path")
    p.add_argument("--export-csv", default=None,
                   help="CSV export path")
    p.add_argument("--save-annotated", action="store_true",
                   help="Save annotated frames")
    p.add_argument("--output-dir", default="output",
                   help="Output directory")
    p.add_argument("--force-download", action="store_true",
                   help="Force dataset re-download")
    return p.parse_args(argv)


def _apply_cli_overrides(
    cfg: DrowsinessConfig, args: argparse.Namespace,
) -> None:
    if args.ear_threshold is not None:
        cfg.ear_threshold = args.ear_threshold
    if args.mar_threshold is not None:
        cfg.mar_threshold = args.mar_threshold
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


def _is_webcam(source: str) -> bool:
    return source.isdigit()


def _is_video(source: str) -> bool:
    return Path(source).suffix.lower() in VIDEO_EXTS


def _is_image(source: str) -> bool:
    return Path(source).suffix.lower() in IMAGE_EXTS


def run(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    cfg = load_config(args.config) if args.config else DrowsinessConfig()
    _apply_cli_overrides(cfg, args)

    if args.force_download:
        from data_bootstrap import ensure_drowsiness_dataset
        ensure_drowsiness_dataset(force=True)

    pipeline = DrowsinessPipeline(cfg)
    pipeline.load()
    if not pipeline.detector.ready:
        log.error("MediaPipe Face Landmarker unavailable")
        return
    validator = DrowsinessValidator(cfg)

    source = args.source

    if _is_webcam(source):
        _run_stream(pipeline, validator, cfg, int(source), "Webcam")
    elif _is_video(source):
        _run_stream(pipeline, validator, cfg, source, Path(source).name)
    elif _is_image(source):
        _run_image(pipeline, validator, cfg, source)
    else:
        log.error("Unsupported source: %s", source)


def _run_stream(
    pipeline: DrowsinessPipeline,
    validator: DrowsinessValidator,
    cfg: DrowsinessConfig,
    source: int | str,
    label: str,
) -> None:
    """Process webcam or video stream."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error("Cannot open source: %s", source)
        return

    log.info("Processing stream: %s -- press 'q' to quit", label)
    frame_idx = 0
    out_dir = Path(cfg.output_dir)

    with DrowsinessExporter(cfg) as exporter:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = pipeline.process(frame)
            report = validator.validate(result)
            exporter.write(result, frame_idx=frame_idx)

            if cfg.show_display:
                vis = draw_overlay(frame, result, cfg)
                cv2.imshow(f"Drowsiness Monitor -- {label}", vis)
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

    # Save alert logs
    if pipeline.alert_manager.count > 0:
        pipeline.alert_manager.save_csv()
        pipeline.alert_manager.save_json()

    log.info(
        "Done -- %d frames, %d blinks, %d yawns, %d alerts",
        frame_idx,
        pipeline.blink_tracker._total_blinks,
        pipeline.yawn_tracker._total_yawns,
        pipeline.alert_manager.count,
    )


def _run_image(
    pipeline: DrowsinessPipeline,
    validator: DrowsinessValidator,
    cfg: DrowsinessConfig,
    source: str,
) -> None:
    """Process a single image."""
    img = cv2.imread(source)
    if img is None:
        log.error("Cannot read: %s", source)
        return

    result = pipeline.process(img)
    report = validator.validate(result)

    log.info(
        "%s: face=%s, EAR=%.2f, MAR=%.2f, yaw=%.0f, pitch=%.0f",
        source,
        result.face_detected,
        result.blink.ear,
        result.yawn.mar,
        result.head_pose.yaw,
        result.head_pose.pitch,
    )

    vis = draw_overlay(img, result, cfg)

    if cfg.save_annotated:
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"annotated_{Path(source).name}"), vis)

    if cfg.show_display:
        cv2.imshow("Drowsiness Monitor", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
