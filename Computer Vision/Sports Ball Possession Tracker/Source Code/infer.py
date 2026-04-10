"""CLI inference pipeline for Sports Ball Possession Tracker.

Main input mode is video.  Also supports webcam.

Usage::

    python infer.py --source match.mp4
    python infer.py --source match.mp4 --config possession_config.yaml
    python infer.py --source 0 --no-display --export-json results.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PossessionConfig, load_config
from export import PossessionExporter
from possession import PossessionEstimator
from tracker import SportTracker
from visualize import Visualizer

log = logging.getLogger("sports_possession.infer")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sports Ball Possession Tracker — Inference")
    p.add_argument("--source", default="0", help="Video path or camera index")
    p.add_argument("--config", default=None, help="Path to YAML/JSON config")
    p.add_argument("--model", default=None, help="Override model weights")
    p.add_argument("--conf", type=float, default=None, help="Override confidence threshold")
    p.add_argument("--imgsz", type=int, default=None, help="Override image size")
    p.add_argument("--no-display", action="store_true", help="Disable GUI window")
    p.add_argument("--save-video", default=None, help="Save annotated video path")
    p.add_argument("--export-json", default=None, help="JSON export path")
    p.add_argument("--export-csv", default=None, help="CSV export path")
    p.add_argument("--force-download", action="store_true", help="Force dataset re-download")
    return p.parse_args(argv)


def run(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    cfg = load_config(args.config) if args.config else PossessionConfig()

    # CLI overrides
    if args.model:
        cfg.model = args.model
    if args.conf is not None:
        cfg.conf_threshold = args.conf
    if args.imgsz is not None:
        cfg.imgsz = args.imgsz
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
        from data_bootstrap import ensure_sports_dataset
        ensure_sports_dataset(force=True)

    tracker = SportTracker(cfg)
    estimator = PossessionEstimator(cfg)
    visualizer = Visualizer(cfg)
    exporter = PossessionExporter(cfg)

    source = args.source
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
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

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            dets = tracker.update(frame, frame_idx=frame_idx)
            state = estimator.update(dets)
            exporter.write_frame(state, num_players=len(dets.players))

            vis = visualizer.draw(frame, dets, state)

            if writer is not None:
                writer.write(vis)

            if cfg.show_display:
                cv2.imshow("Sports Ball Possession Tracker", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if cfg.show_display:
            cv2.destroyAllWindows()

    # Final summary
    exporter.close(estimator)
    pcts = estimator.possession_percentages()
    log.info("Processed %d frames", frame_idx)
    log.info("Possession summary: %s", pcts)
    log.info("Done.")


if __name__ == "__main__":
    run()
