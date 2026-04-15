"""CLI and demo app for Gesture Controlled Slideshow.
"""CLI and demo app for Gesture Controlled Slideshow.

Supports two display modes:
    - **cam-only**: webcam with gesture overlay (no slides needed)
    - **slideshow**: slide image with camera PiP (default)

Usage::

    python infer.py --source 0                         # demo slides
    python infer.py --source 0 --slides ./my_slides/   # custom slides
    python infer.py --source 0 --cam-only              # camera only
    python infer.py --source video.mp4 --cam-only      # from video
    python infer.py --source 0 --export-csv gestures.csv
"""
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import GestureConfig, load_config
from controller import SlideshowController
from export import GestureExporter
from validator import GestureValidator
from visualize import draw_overlay, draw_slide_with_cam

log = logging.getLogger("gesture.infer")

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gesture Controlled Slideshow -- Demo",
    )
    p.add_argument("--source", required=True,
                   help="'0' for webcam, or path to video")
    p.add_argument("--slides", default=None,
                   help="Path to folder of slide images")
    p.add_argument("--cam-only", action="store_true",
                   help="Camera-only mode (no slide display)")
    p.add_argument("--config", default=None,
                   help="Path to YAML/JSON config file")
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


def _apply_overrides(cfg: GestureConfig, args: argparse.Namespace) -> None:
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
    if args.slides:
        cfg.slide_dir = args.slides


def run(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    cfg = load_config(args.config) if args.config else GestureConfig()
    _apply_overrides(cfg, args)

    if args.force_download:
        from data_bootstrap import ensure_gesture_dataset
        ensure_gesture_dataset(force=True)

    controller = SlideshowController(cfg)
    controller.load(slide_dir=args.slides)
    validator = GestureValidator(cfg)

    source = args.source
    cam_only = args.cam_only

    if source.isdigit():
        _run_stream(controller, validator, cfg, int(source), "Webcam", cam_only)
    elif Path(source).suffix.lower() in VIDEO_EXTS:
        _run_stream(controller, validator, cfg, source, Path(source).name, cam_only)
    elif Path(source).suffix.lower() in IMAGE_EXTS:
        _run_image(controller, validator, cfg, source)
    else:
        log.error("Unsupported source: %s", source)


def _run_stream(
    controller: SlideshowController,
    validator: GestureValidator,
    cfg: GestureConfig,
    source: int | str,
    label: str,
    cam_only: bool,
) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error("Cannot open source: %s", source)
        return

    log.info(
        "Gesture slideshow running: %s -- press 'q' to quit", label,
    )
    log.info(
        "Keyboard: n=next, p=prev, SPACE=pause, t=pointer",
    )

    frame_idx = 0
    out_dir = Path(cfg.output_dir)

    with GestureExporter(cfg) as exporter:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip for mirror effect (webcam)
            if isinstance(source, int):
                frame = cv2.flip(frame, 1)

            result = controller.process(frame)
            validator.validate(result)
            exporter.write(result, frame_idx=frame_idx)

            if cfg.show_display:
                if cam_only:
                    vis = draw_overlay(
                        frame, result, cfg,
                        detector=controller.detector,
                    )
                    cv2.imshow(f"Gesture Control -- {label}", vis)
                else:
                    slide_img = controller.slideshow.current_slide()
                    vis = draw_slide_with_cam(
                        slide_img, frame, result, cfg,
                        detector=controller.detector,
                    )
                    cv2.imshow("Gesture Slideshow", vis)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                controller.handle_key(key)

            if cfg.save_annotated and frame_idx % 30 == 0:
                out_dir.mkdir(parents=True, exist_ok=True)
                vis = draw_overlay(
                    frame, result, cfg,
                    detector=controller.detector,
                )
                cv2.imwrite(
                    str(out_dir / f"frame_{frame_idx:06d}.jpg"), vis,
                )

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    log.info("Done -- %d frames processed", frame_idx)


def _run_image(
    controller: SlideshowController,
    validator: GestureValidator,
    cfg: GestureConfig,
    source: str,
) -> None:
    """Process a single image (gesture detection only)."""
    img = cv2.imread(source)
    if img is None:
        log.error("Cannot read: %s", source)
        return

    result = controller.process(img)
    validator.validate(result)

    log.info(
        "%s: hand=%s, gesture=%s, fingers=%d",
        source, result.hand_detected, result.gesture.gesture,
        result.gesture.finger_count,
    )

    vis = draw_overlay(img, result, cfg, detector=controller.detector)

    if cfg.save_annotated:
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"annotated_{Path(source).name}"), vis)

    if cfg.show_display:
        cv2.imshow("Gesture Detection", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
