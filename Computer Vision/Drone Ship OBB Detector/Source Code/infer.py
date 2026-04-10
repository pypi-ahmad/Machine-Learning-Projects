"""CLI inference pipeline for Drone Ship OBB Detector.

Supports aerial images, video, and webcam.

Usage::

    python infer.py --source aerial.jpg
    python infer.py --source satellite_video.mp4 --config obb_config.yaml
    python infer.py --source 0 --no-display --export-json results.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OBBConfig, load_config
from detector import OBBDetector
from export import OBBExporter
from visualize import draw_overlay

log = logging.getLogger("drone_ship_obb.infer")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Drone Ship OBB Detector — Inference")
    p.add_argument("--source", default="0", help="Image/video path or camera index")
    p.add_argument("--config", default=None, help="Path to YAML/JSON config")
    p.add_argument("--model", default=None, help="Override model weights")
    p.add_argument("--conf", type=float, default=None, help="Override confidence threshold")
    p.add_argument("--imgsz", type=int, default=None, help="Override image size")
    p.add_argument("--no-display", action="store_true", help="Disable GUI window")
    p.add_argument("--save-video", default=None, help="Save annotated video")
    p.add_argument("--export-json", default=None, help="JSON export path")
    p.add_argument("--export-txt", default=None, help="TXT export directory")
    p.add_argument("--force-download", action="store_true", help="Force dataset re-download")
    return p.parse_args(argv)


def _is_image(path: str) -> bool:
    return Path(path).suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
    }


def run(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    cfg = load_config(args.config) if args.config else OBBConfig()

    # CLI overrides
    if args.model:
        cfg.model = args.model
    if args.conf is not None:
        cfg.conf_threshold = args.conf
    if args.imgsz is not None:
        cfg.imgsz = args.imgsz
    if args.export_json:
        cfg.export_json = args.export_json
    if args.export_txt:
        cfg.export_txt = args.export_txt
    if args.no_display:
        cfg.show_display = False

    if args.force_download:
        from data_bootstrap import ensure_obb_dataset
        ensure_obb_dataset(force=True)

    detector = OBBDetector(cfg)

    source = args.source
    is_cam = source.isdigit()
    is_img = not is_cam and _is_image(source)

    if is_img:
        _run_image(source, detector, cfg, args)
    else:
        _run_video(int(source) if is_cam else source, detector, cfg, args)

    log.info("Done.")


def _run_image(path: str, detector: OBBDetector, cfg: OBBConfig,
               args: argparse.Namespace) -> None:
    frame = cv2.imread(path)
    if frame is None:
        log.error("Cannot read image: %s", path)
        return

    h, w = frame.shape[:2]
    result = detector.process(frame, frame_idx=0)

    with OBBExporter(cfg, image_shape=(h, w)) as exporter:
        exporter.write(result, image_name=Path(path).stem, image_shape=(h, w))

    vis = draw_overlay(frame, result, cfg)

    if cfg.show_display:
        cv2.imshow("Drone Ship OBB Detector", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if args.save_video:
        out_path = Path(args.save_video).with_suffix(".jpg")
        cv2.imwrite(str(out_path), vis)
        log.info("Saved annotated image → %s", out_path)

    log.info("Detections: %d  |  %s", result.total, result.class_counts)


def _run_video(source: int | str, detector: OBBDetector, cfg: OBBConfig,
               args: argparse.Namespace) -> None:
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
        log.info("Recording → %s", args.save_video)

    frame_idx = 0
    with OBBExporter(cfg, image_shape=(fh, fw)) as exporter:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                result = detector.process(frame, frame_idx=frame_idx)
                exporter.write(result, image_shape=(fh, fw))
                vis = draw_overlay(frame, result, cfg)

                if writer is not None:
                    writer.write(vis)

                if cfg.show_display:
                    cv2.imshow("Drone Ship OBB Detector", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_idx += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if cfg.show_display:
                cv2.destroyAllWindows()
            log.info("Processed %d frames", frame_idx)


if __name__ == "__main__":
    run()
