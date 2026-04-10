"""CLI inference pipeline for Crowd Zone Counter.

Supports image, video, and webcam sources.

Usage::

    python infer.py --source crowd.mp4 --config crowd_config.yaml
    python infer.py --source 0
    python infer.py --source stadium.jpg --no-display --export-json out.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CrowdConfig, load_config
from detector import PersonDetector
from export import CrowdExporter
from visualize import draw_overlay
from zone_counter import ZoneCounter

log = logging.getLogger("crowd_zone.infer")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Crowd Zone Counter — Inference")
    p.add_argument("--source", default="0", help="Image/video path or camera index")
    p.add_argument("--config", default=None, help="Path to YAML/JSON config")
    p.add_argument("--model", default=None, help="Override model weights")
    p.add_argument("--conf", type=float, default=None, help="Override confidence threshold")
    p.add_argument("--no-display", action="store_true", help="Disable GUI window")
    p.add_argument("--save-video", default=None, help="Save annotated video path")
    p.add_argument("--export-json", default=None, help="JSON export path")
    p.add_argument("--export-csv", default=None, help="CSV export path")
    p.add_argument("--force-download", action="store_true", help="Force dataset re-download")
    return p.parse_args(argv)


def _is_image(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def run(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    cfg = load_config(args.config) if args.config else CrowdConfig()

    # CLI overrides
    if args.model:
        cfg.model = args.model
    if args.conf is not None:
        cfg.conf_threshold = args.conf
    if args.no_display:
        cfg.show_display = False
    if args.save_video:
        cfg.save_video = True
        cfg.output_path = args.save_video
    if args.export_json:
        cfg.export_json = args.export_json
    if args.export_csv:
        cfg.export_csv = args.export_csv

    if args.force_download:
        from data_bootstrap import ensure_crowd_dataset
        ensure_crowd_dataset(force=True)

    detector = PersonDetector(cfg)
    counter = ZoneCounter(cfg)

    source = args.source
    is_cam = source.isdigit()
    is_img = not is_cam and _is_image(source)

    if is_img:
        _run_image(source, detector, counter, cfg, args)
    else:
        _run_video(int(source) if is_cam else source, detector, counter, cfg, args)

    log.info("Done.")


def _run_image(path: str, detector: PersonDetector, counter: ZoneCounter,
               cfg: CrowdConfig, args: argparse.Namespace) -> None:
    frame = cv2.imread(path)
    if frame is None:
        log.error("Cannot read image: %s", path)
        return

    dets = detector.detect(frame, frame_idx=0)
    result = counter.update(dets)

    with CrowdExporter(cfg) as exporter:
        exporter.write(result)

    vis = draw_overlay(frame, dets, result, cfg)

    if cfg.show_display:
        cv2.imshow("Crowd Zone Counter", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if args.save_video:
        out_path = Path(args.save_video).with_suffix(".jpg")
        cv2.imwrite(str(out_path), vis)
        log.info("Saved annotated image → %s", out_path)

    for zs in result.zone_states:
        cap = f"/{zs.max_capacity}" if zs.max_capacity > 0 else ""
        alert = " ⚠ OVERCROWDED" if zs.overcrowded else ""
        log.info("  %s: %d%s%s", zs.name, zs.count, cap, alert)
    log.info("Total: %d  Unzoned: %d", result.total_persons, result.unzoned_count)


def _run_video(source: int | str, detector: PersonDetector, counter: ZoneCounter,
               cfg: CrowdConfig, args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error("Cannot open video source: %s", source)
        return

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer: cv2.VideoWriter | None = None
    if cfg.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        Path(cfg.output_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(cfg.output_path, fourcc, fps, (fw, fh))
        log.info("Recording → %s", cfg.output_path)

    exporter = CrowdExporter(cfg)
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            dets = detector.detect(frame, frame_idx=frame_idx)
            result = counter.update(dets)
            exporter.write(result)

            vis = draw_overlay(frame, dets, result, cfg)

            if writer is not None:
                writer.write(vis)

            if cfg.show_display:
                cv2.imshow("Crowd Zone Counter", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if cfg.show_display:
            cv2.destroyAllWindows()
        exporter.close()
        log.info("Processed %d frames", frame_idx)


if __name__ == "__main__":
    run()
