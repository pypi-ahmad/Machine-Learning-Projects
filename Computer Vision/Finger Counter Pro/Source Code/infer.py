"""Finger Counter Pro -- CLI entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Finger Counter Pro -- robust multi-hand finger counting",
    )
    p.add_argument(
        "--source",
        default="0",
        help="'0' for webcam, or path to video/image (default: 0)",
    )
    p.add_argument("--config", default=None, help="YAML/JSON config path")
    p.add_argument(
        "--no-smoothing",
        action="store_true",
        help="Disable EMA + majority-vote smoothing",
    )
    p.add_argument(
        "--no-display",
        action="store_true",
        help="Disable GUI windows (headless mode)",
    )
    p.add_argument("--export-csv", default=None, help="CSV export path")
    p.add_argument("--export-json", default=None, help="JSON export path")
    p.add_argument(
        "--save-annotated",
        action="store_true",
        help="Save annotated frames to output dir",
    )
    p.add_argument(
        "--output-dir",
        default="output",
        help="Output directory (default: output/)",
    )
    p.add_argument(
        "--force-download",
        action="store_true",
        help="Force dataset re-download",
    )
    return p


def _is_image(path: str) -> bool:
    return Path(path).suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp",
    }


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    # --- lazy imports (keep CLI fast) ---
    import cv2

    from config import FingerCounterConfig, load_config
    from controller import CountingController
    from export import FrameExporter
    from visualize import draw_overlay

    # Bootstrap dataset if requested
    if args.force_download:
        from data_bootstrap import ensure_finger_counter_dataset
        ensure_finger_counter_dataset(force=True)

    # Config
    cfg = load_config(args.config) if args.config else FingerCounterConfig()
    if args.no_smoothing:
        cfg.enable_smoothing = False

    ctrl = CountingController(cfg)
    ctrl.load()

    # Source
    source = args.source
    if source.isdigit():
        source = int(source)

    is_img = isinstance(source, str) and _is_image(source)

    out_dir = Path(args.output_dir)
    if args.save_annotated:
        out_dir.mkdir(parents=True, exist_ok=True)

    with FrameExporter(args.export_csv, args.export_json) as exporter:
        if is_img:
            frame = cv2.imread(source)
            if frame is None:
                print(f"Cannot read image: {source}", file=sys.stderr)
                sys.exit(1)
            _process_frame(frame, ctrl, cfg, exporter, draw_overlay, cv2, args, out_dir, 0)
            if not args.no_display:
                cv2.waitKey(0)
        else:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print(f"Cannot open source: {source}", file=sys.stderr)
                sys.exit(1)
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                # Mirror webcam for natural interaction
                if isinstance(source, int):
                    frame = cv2.flip(frame, 1)
                _process_frame(frame, ctrl, cfg, exporter, draw_overlay, cv2, args, out_dir, idx)
                idx += 1
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    ctrl.reset()
            cap.release()

    ctrl.close()
    if not args.no_display:
        cv2.destroyAllWindows()


def _process_frame(frame, ctrl, cfg, exporter, draw_overlay, cv2, args, out_dir, idx):
    """Process a single frame through the pipeline."""
    result = ctrl.process(frame)

    # Export
    exporter.record(
        per_hand=result.frame_count.per_hand,
        total_raw=result.frame_count.total,
        smoothed_per_hand=result.smoothed_per_hand,
        smoothed_total=result.smoothed_total,
    )

    # Visualise
    if not args.no_display or args.save_annotated:
        vis = draw_overlay(
            frame,
            result.multi,
            result.frame_count.per_hand,
            result.smoothed_per_hand,
            result.smoothed_total,
            ctrl.detector,
            show_landmarks=cfg.show_landmarks,
            show_finger_state=cfg.show_finger_state,
            show_count=cfg.show_count,
            show_handedness=cfg.show_handedness,
        )
        if not args.no_display:
            cv2.imshow("Finger Counter Pro", vis)
        if args.save_annotated:
            cv2.imwrite(str(out_dir / f"frame_{idx:06d}.jpg"), vis)


if __name__ == "__main__":
    main()
