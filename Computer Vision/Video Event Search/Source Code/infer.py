"""Video Event Search — inference pipeline.
"""Video Event Search — inference pipeline.

Runs YOLO detection + tracking on video, generates structured events,
stores them, and provides a query interface.

Usage (command-line)::

    python infer.py --source crosswalk.avi
    python infer.py --source crosswalk.avi --config event_search.yaml
    python infer.py --query --event-type zone_enter
    python infer.py --query --summary
"""
"""

from __future__ import annotations

import argparse
import json
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

from config import EventSearchConfig, load_config  # noqa: E402
from event_generator import EventGenerator          # noqa: E402
from event_store import EventStore                   # noqa: E402
from query import EventQuery                         # noqa: E402
from tracker import TrackManager                     # noqa: E402
from visualize import OverlayRenderer                # noqa: E402

log = logging.getLogger("video_event_search.infer")


# ---------------------------------------------------------------------------
# Video processing pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    source: str | int,
    cfg: EventSearchConfig,
    *,
    force_download: bool = False,
) -> Path:
    """Run detection + tracking + event generation on a video source.
    """Run detection + tracking + event generation on a video source.

    Returns the path to the generated events JSON.
    """
    """
    if force_download:
        from data_bootstrap import ensure_video_event_dataset
        ensure_video_event_dataset(force=True)

    model = load_yolo(cfg.model, device=cfg.device)

    # Open video
    try:
        src = int(source)
    except (ValueError, TypeError):
        src = str(source)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        log.error("Cannot open video source: %s", source)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    log.info("Video: %dx%d @ %.1f FPS (%d frames)", w, h, fps, total_frames)

    tm = TrackManager(max_history=120)
    gen = EventGenerator(cfg, fps=fps)
    out_dir = Path(cfg.export_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    store = EventStore(out_dir / "events.json", out_dir / "events.csv")
    renderer = OverlayRenderer()

    writer = None
    if cfg.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(out_dir / "annotated.mp4"), fourcc, fps, (w, h),
        )

    frame_idx = 0
    recent_events = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO tracking
        try:
            results = model.track(
                frame, persist=True, verbose=False,
                conf=cfg.conf_threshold,
                iou=cfg.iou_threshold,
                tracker=cfg.tracker,
            )
        except Exception:
            results = model(
                frame, verbose=False,
                conf=cfg.conf_threshold,
                iou=cfg.iou_threshold,
            )

        dets = tm.update(results, cfg.conf_threshold)
        events = gen.process(dets, tm, frame_idx)
        store.add_batch(events)
        recent_events = (recent_events + events)[-10:]

        if cfg.show_display or writer:
            annotated = renderer.draw(frame, dets, recent_events,
                                      tm.get_all_trails(), cfg)
            if cfg.show_display:
                cv2.imshow("Video Event Search", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if writer:
                writer.write(annotated)

        frame_idx += 1
        if frame_idx % 100 == 0:
            log.info("Frame %d/%d -- %d events so far", frame_idx,
                     total_frames, store.count)

    cap.release()
    if writer:
        writer.release()
    if cfg.show_display:
        cv2.destroyAllWindows()

    store.flush()
    log.info("Pipeline done: %d frames, %d events", frame_idx, store.count)
    return store.json_path


# ---------------------------------------------------------------------------
# Query mode
# ---------------------------------------------------------------------------

def run_query(
    events_path: str | Path,
    *,
    event_type: str | None = None,
    track_id: int | None = None,
    class_name: str | None = None,
    zone: str | None = None,
    time_range: tuple[float, float] | None = None,
    show_summary: bool = False,
    limit: int = 20,
) -> None:
    """Query stored events and print results."""
    q = EventQuery(events_path)

    if show_summary:
        summary = q.summary()
        print(json.dumps(summary, indent=2))
        return

    results = q.search(
        event_type=event_type,
        track_id=track_id,
        class_name=class_name,
        zone=zone,
        time_range=time_range,
        limit=limit,
    )
    print(f"Found {len(results)} events:")
    for evt in results:
        ts = evt.get("timestamp_sec", 0)
        print(f"  [{ts:7.2f}s] {evt['event_type']:12s} "
              f"T{evt.get('track_id', '?'):>3} {evt.get('class_name', '')}"
              f"  {evt.get('zone_or_line', '')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Video Event Search -- detect, track, and search video events",
    )
    sub = parser.add_subparsers(dest="command")

    # -- process sub-command --
    p_proc = sub.add_parser("process", help="Process video and generate events")
    p_proc.add_argument("--source", required=True, help="Video file or camera index")
    p_proc.add_argument("--config", default=None, help="YAML/JSON config file")
    p_proc.add_argument("--output-dir", default="outputs", help="Output directory")
    p_proc.add_argument("--no-display", action="store_true", help="Disable GUI")
    p_proc.add_argument("--save-video", action="store_true", help="Save annotated video")
    p_proc.add_argument("--download", action="store_true", help="Force dataset download")

    # -- query sub-command --
    p_query = sub.add_parser("query", help="Search stored events")
    p_query.add_argument("--events", default="outputs/events.json",
                         help="Path to events JSON")
    p_query.add_argument("--event-type", default=None, help="Filter by event type")
    p_query.add_argument("--track-id", type=int, default=None, help="Filter by track ID")
    p_query.add_argument("--class-name", default=None, help="Filter by class")
    p_query.add_argument("--zone", default=None, help="Filter by zone/line name")
    p_query.add_argument("--time-start", type=float, default=None)
    p_query.add_argument("--time-end", type=float, default=None)
    p_query.add_argument("--summary", action="store_true", help="Show event summary")
    p_query.add_argument("--limit", type=int, default=20)

    args = parser.parse_args()

    if args.command == "process":
        cfg = load_config(args.config)
        cfg.export_dir = args.output_dir
        cfg.show_display = not args.no_display
        cfg.save_video = args.save_video
        events_path = run_pipeline(args.source, cfg, force_download=args.download)
        print(f"\nEvents saved to: {events_path}")

    elif args.command == "query":
        time_range = None
        if args.time_start is not None or args.time_end is not None:
            time_range = (args.time_start or 0.0, args.time_end or 999999.0)

        run_query(
            args.events,
            event_type=args.event_type,
            track_id=args.track_id,
            class_name=args.class_name,
            zone=args.zone,
            time_range=time_range,
            show_summary=args.summary,
            limit=args.limit,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
