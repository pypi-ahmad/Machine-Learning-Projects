"""Traffic Violation Analyzer — inference pipeline.

Runs YOLO detection + tracking on video files or a webcam stream, feeds
tracked detections through the rule engine → visualise → export pipeline.
Also supports single-image mode (no tracking).

Usage (command-line)::

    python infer.py --source traffic.mp4 --config traffic.yaml
    python infer.py --source 0 --config traffic.yaml
    python infer.py --source frame.jpg --no-display
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.yolo import load_yolo  # noqa: E402

from config import TrafficConfig, load_config  # noqa: E402
from detector import Detection                 # noqa: E402
from export import EventExporter               # noqa: E402
from rules import RuleEngine                   # noqa: E402
from tracker import TrackManager               # noqa: E402
from visualize import OverlayRenderer          # noqa: E402

log = logging.getLogger("traffic_violation.infer")


# ---------------------------------------------------------------------------
# Detection parser (for image-only / non-tracking fallback)
# ---------------------------------------------------------------------------

def _parse_detections(results, conf_threshold: float = 0.30) -> list[Detection]:
    """Convert plain YOLO results (no tracking) to Detection list."""
    detections: list[Detection] = []
    for result in results:
        names = result.names
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            cls_name = names.get(cls_id, str(cls_id))
            detections.append(Detection(
                box=(x1, y1, x2, y2),
                center=((x1 + x2) // 2, (y1 + y2) // 2),
                class_name=cls_name,
                confidence=conf,
                class_id=cls_id,
            ))
    return detections


# ---------------------------------------------------------------------------
# Core runners
# ---------------------------------------------------------------------------

def run_inference(
    source: str | int,
    cfg: TrafficConfig,
    *,
    force_download: bool = False,
) -> None:
    """Run the full inference + tracking + rule-engine pipeline."""
    if force_download:
        from data_bootstrap import ensure_traffic_dataset
        ensure_traffic_dataset(force=True)

    model = load_yolo(cfg.model, device=cfg.device)

    tm = TrackManager(max_history=60)
    engine = RuleEngine(cfg)
    renderer = OverlayRenderer()
    exporter = EventExporter(cfg)

    # Determine source type
    try:
        src = int(source)
    except (ValueError, TypeError):
        src = str(source)

    if isinstance(src, str) and Path(src).suffix.lower() in (
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
    ):
        _run_image(src, model, cfg, engine, renderer, exporter)
    else:
        _run_video(src, model, cfg, tm, engine, renderer, exporter)

    exporter.flush()
    log.info("Total events logged: %d", exporter.event_count)


def _run_image(
    path: str,
    model,
    cfg: TrafficConfig,
    engine: RuleEngine,
    renderer: OverlayRenderer,
    exporter: EventExporter,
) -> None:
    """Single-image mode — detection only (no tracking)."""
    frame = cv2.imread(path)
    if frame is None:
        log.error("Cannot read image: %s", path)
        return

    results = model.predict(frame, conf=cfg.conf_threshold, iou=cfg.iou_threshold,
                            imgsz=cfg.imgsz, verbose=False)
    dets = _parse_detections(results, cfg.conf_threshold)

    from rules import FrameEvents
    fe = FrameEvents(line_counts=engine.line_counts)

    annotated = renderer.draw(frame, dets, fe, {}, cfg)

    if cfg.show_display:
        cv2.imshow("Traffic Violation Analyzer", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        out = Path(cfg.export_dir) / "result.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), annotated)
        log.info("Saved -> %s", out)


def _run_video(
    source,
    model,
    cfg: TrafficConfig,
    tm: TrackManager,
    engine: RuleEngine,
    renderer: OverlayRenderer,
    exporter: EventExporter,
) -> None:
    """Video / webcam mode — detection + tracking + rules."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error("Cannot open video source: %s", source)
        return

    writer: cv2.VideoWriter | None = None
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run tracking
            try:
                results = model.track(
                    frame,
                    persist=True,
                    conf=cfg.conf_threshold,
                    iou=cfg.iou_threshold,
                    imgsz=cfg.imgsz,
                    tracker=cfg.tracker,
                    verbose=False,
                )
            except Exception:
                # Fallback to plain detection if tracker unavailable
                results = model.predict(
                    frame,
                    conf=cfg.conf_threshold,
                    iou=cfg.iou_threshold,
                    imgsz=cfg.imgsz,
                    verbose=False,
                )

            dets = tm.update(results, cfg.conf_threshold)
            fe = engine.evaluate(dets, tm, frame_idx)
            trails = tm.get_all_trails()
            annotated = renderer.draw(frame, dets, fe, trails, cfg)

            exporter.log_events(fe)

            if cfg.save_video and writer is None:
                h, w = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_path = Path(cfg.export_dir) / "output.mp4"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(out_path), fourcc, cfg.output_fps, (w, h))

            if writer is not None:
                writer.write(annotated)

            if cfg.show_display:
                cv2.imshow("Traffic Violation Analyzer", annotated)
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Traffic Violation Analyzer — Inference")
    parser.add_argument("--source", required=True,
                        help="Video path or webcam index (0, 1, …)")
    parser.add_argument("--config", default=None,
                        help="Path to YAML / JSON config file")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable live window")
    parser.add_argument("--save-video", action="store_true",
                        help="Save annotated video to export dir")
    parser.add_argument("--force-download", action="store_true",
                        help="Force dataset re-download")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    cfg = load_config(args.config) if args.config else TrafficConfig()
    if args.no_display:
        cfg.show_display = False
    if args.save_video:
        cfg.save_video = True

    run_inference(args.source, cfg, force_download=args.force_download)


if __name__ == "__main__":
    main()
