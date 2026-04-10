"""Inference entry point — image, video, and webcam.

Orchestrates detection → zone counting → visualization → export.

Usage::

    # Image
    python infer.py --source shelf.jpg

    # Video
    python infer.py --source store_cam.mp4

    # Webcam
    python infer.py --source 0

    # With config
    python infer.py --source shelf.jpg --config zones.yaml

    # Force dataset download (only matters if no custom weights)
    python infer.py --source shelf.jpg --force-download
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
# Project root (for local imports)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from config import StockoutConfig, load_config, default_sample_config
from data_bootstrap import ensure_retail_dataset
from export import EventExporter
from visualize import OverlayRenderer
from zones import Detection, ZoneCounter

from utils.yolo import load_yolo


def _parse_detections(results, model) -> list[Detection]:
    """Convert YOLO results into Detection objects."""
    detections: list[Detection] = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cls_id = int(box.cls[0])
        cls_name = model.names.get(cls_id, str(cls_id))
        detections.append(Detection(
            box=(x1, y1, x2, y2),
            center=(cx, cy),
            class_name=cls_name,
            confidence=float(box.conf[0]),
            class_id=cls_id,
        ))
    return detections


def run_inference(cfg: StockoutConfig, source: str) -> None:
    """Run detection + counting + export pipeline on a source.

    Parameters
    ----------
    cfg : StockoutConfig
        Full project configuration.
    source : str
        Image path, video path, or ``"0"`` for webcam.
    """
    # ── Model ──
    model = load_yolo(cfg.model, device=cfg.device)
    print(f"[INFO] Loaded model: {cfg.model}")

    # ── Zone counter ──
    counter = ZoneCounter(cfg.zones, default_threshold=cfg.default_low_stock_threshold)

    # ── Renderer ──
    renderer = OverlayRenderer(counter)

    # ── Exporter ──
    export_dir = Path(cfg.export_dir)
    exporter = EventExporter(
        output_dir=export_dir,
        save_csv=cfg.save_events_csv,
        save_json=cfg.save_events_json,
        save_snapshots=cfg.save_alert_snapshots,
        snapshot_cooldown=cfg.snapshot_cooldown_sec,
    )

    # ── Determine source type ──
    is_image = False
    src = source
    if source.isdigit():
        src = int(source)
    elif Path(source).suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"):
        is_image = True

    if is_image:
        _run_image(model, cfg, counter, renderer, exporter, source)
    else:
        _run_video(model, cfg, counter, renderer, exporter, src)

    # ── Flush ──
    written = exporter.flush()
    print(f"\n[INFO] Events: {exporter.event_count}")
    for k, v in written.items():
        print(f"  {k}: {v}")


def _run_image(model, cfg, counter, renderer, exporter, path: str) -> None:
    """Single-image inference."""
    frame = cv2.imread(path)
    if frame is None:
        print(f"[ERROR] Cannot read image: {path}")
        return

    results = model(frame, verbose=False, conf=cfg.conf_threshold, iou=cfg.iou_threshold)
    detections = _parse_detections(results, model)
    frame_result = counter.update(detections)

    annotated = renderer.draw(frame, frame_result)
    exporter.log_frame(frame, frame_result, annotated)

    # Save output image
    out_path = Path(cfg.export_dir) / f"result_{Path(path).stem}.jpg"
    cv2.imwrite(str(out_path), annotated)
    print(f"[INFO] Saved result to {out_path}")

    if cfg.show_display:
        cv2.imshow("Retail Shelf Stockout Detector", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Print per-zone summary
    for zs in frame_result.zone_statuses:
        status = "LOW" if zs.is_low_stock else "OK"
        print(f"  Zone '{zs.name}': {zs.count}/{zs.threshold} [{status}]")


def _run_video(model, cfg, counter, renderer, exporter, source) -> None:
    """Video / webcam inference loop."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or cfg.output_fps

    writer = None
    if cfg.save_video:
        out_path = Path(cfg.export_dir) / "output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    frame_count = 0
    t0 = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model(frame, verbose=False, conf=cfg.conf_threshold, iou=cfg.iou_threshold)
            detections = _parse_detections(results, model)
            frame_result = counter.update(detections)
            annotated = renderer.draw(frame, frame_result)

            exporter.log_frame(frame, frame_result, annotated)

            if writer:
                writer.write(annotated)

            frame_count += 1

            if cfg.show_display:
                cv2.imshow("Retail Shelf Stockout Detector", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")
    finally:
        elapsed = time.time() - t0
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"[INFO] Processed {frame_count} frames in {elapsed:.1f}s ({avg_fps:.1f} FPS)")
        cap.release()
        if writer:
            writer.release()
        if cfg.show_display:
            cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retail Shelf Stockout Detector — Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python infer.py --source shelf.jpg\n"
            "  python infer.py --source store.mp4 --config zones.yaml\n"
            "  python infer.py --source 0 --no-display\n"
        ),
    )
    parser.add_argument("--source", required=True, help="Image, video path, or 0 for webcam")
    parser.add_argument("--config", type=str, default=None, help="Zone config YAML/JSON")
    parser.add_argument("--model", type=str, default=None, help="Model weights (overrides config)")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, 0, cuda:0)")
    parser.add_argument("--export-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--no-display", action="store_true", help="Disable GUI window")
    parser.add_argument("--save-video", action="store_true", help="Save annotated video")
    parser.add_argument("--force-download", action="store_true", help="Force dataset re-download")
    args = parser.parse_args()

    # Ensure dataset is available
    ensure_retail_dataset(force=args.force_download)

    # Load config
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = default_sample_config()

    # CLI overrides
    if args.model:
        cfg.model = args.model
    if args.conf:
        cfg.conf_threshold = args.conf
    if args.device:
        cfg.device = args.device
    if args.export_dir:
        cfg.export_dir = args.export_dir
    if args.no_display:
        cfg.show_display = False
    if args.save_video:
        cfg.save_video = True

    run_inference(cfg, args.source)


if __name__ == "__main__":
    main()
