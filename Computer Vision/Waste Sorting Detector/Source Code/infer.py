"""CLI inference pipeline for Waste Sorting Detector.

Supports image, video, and webcam sources.

Usage::

    python infer.py --source video.mp4
    python infer.py --source 0                     # webcam
    python infer.py --source img.jpg --config waste.yaml
    python infer.py --source video.mp4 --save-video out.mp4 --no-display
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import WasteConfig, load_config
from export import WasteExporter
from sorter import WasteSorter
from visualize import draw_overlay

log = logging.getLogger("waste_sorting.infer")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Waste Sorting Detector — Inference")
    p.add_argument("--source", default="0", help="Image/video path or camera index (default: 0)")
    p.add_argument("--config", default=None, help="Path to YAML/JSON config")
    p.add_argument("--model", default=None, help="Override model weights path")
    p.add_argument("--conf", type=float, default=None, help="Override confidence threshold")
    p.add_argument("--no-display", action="store_true", help="Disable GUI window")
    p.add_argument("--save-video", default=None, help="Save annotated output to video file")
    p.add_argument("--export-csv", default=None, help="CSV export path")
    p.add_argument("--export-json", default=None, help="JSON export path")
    p.add_argument("--force-download", action="store_true", help="Force dataset re-download")
    return p.parse_args(argv)


def _is_image(path: str) -> bool:
    return Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def run(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    # Load / build configuration
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = WasteConfig()

    # CLI overrides
    if args.model:
        cfg.model = args.model
    if args.conf is not None:
        cfg.conf_threshold = args.conf
    if args.export_csv:
        cfg.export_csv = args.export_csv
    if args.export_json:
        cfg.export_json = args.export_json

    # Dataset bootstrap (optional)
    if args.force_download:
        from data_bootstrap import ensure_waste_dataset
        ensure_waste_dataset(force=True)

    sorter = WasteSorter(cfg)
    exporter = WasteExporter(cfg)

    source = args.source
    is_cam = source.isdigit()
    is_img = not is_cam and _is_image(source)

    if is_img:
        _run_image(source, sorter, exporter, cfg, args)
    else:
        _run_video(int(source) if is_cam else source, sorter, exporter, cfg, args)

    exporter.close()
    log.info("Done.")


# ------------------------------------------------------------------
# Image mode
# ------------------------------------------------------------------

def _run_image(path: str, sorter: WasteSorter, exporter: WasteExporter,
               cfg: WasteConfig, args: argparse.Namespace) -> None:
    frame = cv2.imread(path)
    if frame is None:
        log.error("Cannot read image: %s", path)
        return

    result = sorter.process(frame, frame_idx=0)
    exporter.write(result)
    vis = draw_overlay(frame, result, cfg)

    if not args.no_display:
        cv2.imshow("Waste Sorting Detector", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if args.save_video:
        out_path = Path(args.save_video).with_suffix(".jpg")
        cv2.imwrite(str(out_path), vis)
        log.info("Saved annotated image -> %s", out_path)


# ------------------------------------------------------------------
# Video / webcam mode
# ------------------------------------------------------------------

def _run_video(source: int | str, sorter: WasteSorter, exporter: WasteExporter,
               cfg: WasteConfig, args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error("Cannot open video source: %s", source)
        return

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer: cv2.VideoWriter | None = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_video, fourcc, fps, (fw, fh))
        log.info("Recording -> %s", args.save_video)

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = sorter.process(frame, frame_idx=frame_idx)
            exporter.write(result)
            vis = draw_overlay(frame, result, cfg)

            if writer is not None:
                writer.write(vis)

            if not args.no_display:
                cv2.imshow("Waste Sorting Detector", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        log.info("Processed %d frames", frame_idx)


if __name__ == "__main__":
    run()
