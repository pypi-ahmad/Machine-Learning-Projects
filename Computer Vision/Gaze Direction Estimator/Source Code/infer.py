"""CLI inference for Gaze Direction Estimator.

Supports webcam, video, and single-image inputs with optional
calibration, smoothing, and per-frame CSV/JSON export.

Usage::

    python infer.py --source 0                      # webcam
    python infer.py --source video.mp4              # video
    python infer.py --source face.jpg               # image
    python infer.py --source 0 --calibrate          # with calibration
    python infer.py --source 0 --export-csv gaze.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyzer import GazePipeline
from calibrator import CALIBRATION_POSITIONS, GazeCalibrator
from config import GazeConfig, load_config
from export import GazeExporter
from iris_locator import locate_iris
from validator import GazeValidator
from visualize import draw_overlay

log = logging.getLogger("gaze.infer")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gaze Direction Estimator -- Inference",
    )
    p.add_argument("--source", required=True,
                   help="'0' for webcam, or path to video/image")
    p.add_argument("--config", default=None,
                   help="Path to YAML/JSON config file")
    p.add_argument("--calibrate", action="store_true",
                   help="Run interactive calibration before inference")
    p.add_argument("--calibration-file", default=None,
                   help="Path to save/load calibration offsets")
    p.add_argument("--no-smoothing", action="store_true",
                   help="Disable temporal smoothing")
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


def _apply_overrides(cfg: GazeConfig, args: argparse.Namespace) -> None:
    if args.no_smoothing:
        cfg.enable_smoothing = False
    if args.calibration_file:
        cfg.calibration_file = args.calibration_file
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

    cfg = load_config(args.config) if args.config else GazeConfig()
    _apply_overrides(cfg, args)

    if args.calibrate and not cfg.show_display:
        log.error("Calibration requires display output; remove --no-display")
        return

    if args.force_download:
        from data_bootstrap import ensure_gaze_dataset
        ensure_gaze_dataset(force=True)

    pipeline = GazePipeline(cfg)
    pipeline.load()
    validator = GazeValidator(cfg)

    # Interactive calibration
    if args.calibrate and args.source.isdigit():
        _run_calibration(pipeline, cfg, int(args.source))

    source = args.source
    if source.isdigit():
        _run_stream(pipeline, validator, cfg, int(source), "Webcam")
    elif Path(source).suffix.lower() in VIDEO_EXTS:
        _run_stream(pipeline, validator, cfg, source, Path(source).name)
    elif Path(source).suffix.lower() in IMAGE_EXTS:
        _run_image(pipeline, validator, cfg, source)
    else:
        log.error("Unsupported source: %s", source)


def _run_calibration(
    pipeline: GazePipeline,
    cfg: GazeConfig,
    source: int,
) -> None:
    """Run interactive calibration via webcam."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error("Cannot open webcam for calibration")
        return

    calibrator = pipeline.calibrator
    calibrator.reset()

    log.info("Starting calibration -- follow on-screen instructions")

    for position in CALIBRATION_POSITIONS:
        log.info("Look %s and press SPACE", position)
        collecting = False
        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            lm = pipeline.detector.detect(frame)
            iris = locate_iris(lm)

            # Overlay instruction
            vis = frame.copy()
            status = f"Calibration: Look {position}"
            if collecting:
                status += f" -- collecting ({count}/{cfg.calibration_frames})"
            else:
                status += " -- press SPACE to start"
            cv2.putText(
                vis, status, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )
            cv2.imshow("Gaze Calibration", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return
            if key == ord(" ") and not collecting:
                collecting = True

            if collecting and iris.detected:
                calibrator.record(iris, position)
                count += 1
                if count >= cfg.calibration_frames:
                    break

    cap.release()
    cv2.destroyAllWindows()

    offsets = calibrator.compute_offsets()
    pipeline.set_offsets(offsets)

    if cfg.calibration_file:
        calibrator.save(cfg.calibration_file, offsets)

    log.info("Calibration complete")


def _run_stream(
    pipeline: GazePipeline,
    validator: GazeValidator,
    cfg: GazeConfig,
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

    with GazeExporter(cfg) as exporter:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = pipeline.process(frame)
            validator.validate(result)
            exporter.write(result, frame_idx=frame_idx)

            if cfg.show_display:
                vis = draw_overlay(frame, result, cfg)
                cv2.imshow(f"Gaze Direction -- {label}", vis)
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
    log.info("Done -- %d frames processed", frame_idx)


def _run_image(
    pipeline: GazePipeline,
    validator: GazeValidator,
    cfg: GazeConfig,
    source: str,
) -> None:
    img = cv2.imread(source)
    if img is None:
        log.error("Cannot read: %s", source)
        return

    result = pipeline.process(img)
    validator.validate(result)

    log.info(
        "%s: face=%s, direction=%s, h=%.2f, v=%.2f, conf=%s",
        source, result.face_detected, result.direction,
        result.raw_gaze.h_ratio, result.raw_gaze.v_ratio,
        result.raw_gaze.confidence,
    )

    vis = draw_overlay(img, result, cfg)

    if cfg.save_annotated:
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"annotated_{Path(source).name}"), vis)

    if cfg.show_display:
        cv2.imshow("Gaze Direction Estimator", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
