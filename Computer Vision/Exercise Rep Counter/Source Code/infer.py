"""Exercise Rep Counter — CLI entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Exercise Rep Counter — pose-based rep counting",
    )
    p.add_argument(
        "--source", default="0",
        help="'0' for webcam, or path to video/image (default: 0)",
    )
    p.add_argument(
        "--exercise", default="squat",
        choices=["squat", "pushup", "bicep_curl"],
        help="Exercise type (default: squat)",
    )
    p.add_argument(
        "--side", default="left", choices=["left", "right"],
        help="Body side to track (default: left)",
    )
    p.add_argument("--config", default=None, help="YAML/JSON config path")
    p.add_argument("--no-smoothing", action="store_true", help="Disable EMA smoothing")
    p.add_argument("--no-display", action="store_true", help="Headless mode")
    p.add_argument("--export-csv", default=None, help="CSV export path")
    p.add_argument("--export-json", default=None, help="JSON export path")
    p.add_argument("--save-annotated", action="store_true", help="Save annotated frames")
    p.add_argument("--output-dir", default="output", help="Output directory")
    p.add_argument("--force-download", action="store_true", help="Re-download dataset")
    return p


def _is_image(path: str) -> bool:
    return Path(path).suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp",
    }


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    import cv2

    from config import ExerciseConfig, load_config
    from controller import ExerciseController
    from export import FrameExporter
    from visualize import draw_overlay

    if args.force_download:
        from data_bootstrap import ensure_exercise_dataset
        ensure_exercise_dataset(force=True)

    cfg = load_config(args.config) if args.config else ExerciseConfig()
    cfg.exercise = args.exercise
    if args.no_smoothing:
        cfg.enable_smoothing = False

    ctrl = ExerciseController(cfg)
    ctrl.load()

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
    print(f"\nTotal reps: {ctrl.reps}")


def _process_frame(frame, ctrl, cfg, exporter, draw_overlay, cv2, args, out_dir, idx):
    result = ctrl.process(frame, side=args.side)
    exporter.record(result)

    if not args.no_display or args.save_annotated:
        vis = draw_overlay(
            frame, result, ctrl.detector,
            show_skeleton=cfg.show_skeleton,
            show_angles=cfg.show_angles,
            show_rep_count=cfg.show_rep_count,
            show_stage=cfg.show_stage,
            show_exercise=cfg.show_exercise,
        )
        if not args.no_display:
            cv2.imshow("Exercise Rep Counter", vis)
        if args.save_annotated:
            cv2.imwrite(str(out_dir / f"frame_{idx:06d}.jpg"), vis)


if __name__ == "__main__":
    main()
