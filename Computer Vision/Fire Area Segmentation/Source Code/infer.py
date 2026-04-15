"""Fire Area Segmentation — CLI entry point.
"""Fire Area Segmentation — CLI entry point.

Usage::

    # Single image
    python infer.py --source fire.jpg

    # Video
    python infer.py --source wildfire.mp4 --save-annotated

    # Webcam
    python infer.py --source 0

    # Directory of images
    python infer.py --source images/ --save-annotated --save-masks

    # Full export
    python infer.py --source fire.jpg \
        --export-json report.json --export-csv stats.csv \
        --save-annotated --save-masks
"""
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fire Area Segmentation -- fire/smoke instance segmentation",
    )
    p.add_argument(
        "--source", default="0",
        help="'0' for webcam, or path to image/video/directory (default: 0)",
    )
    p.add_argument("--config", default=None, help="YAML/JSON config path")
    p.add_argument("--no-display", action="store_true", help="Headless mode")
    p.add_argument("--output-dir", default="output", help="Output directory")
    p.add_argument("--export-json", default=None, help="JSON export path")
    p.add_argument("--export-csv", default=None, help="CSV export path")
    p.add_argument("--save-annotated", action="store_true",
                    help="Save annotated images/video")
    p.add_argument("--save-masks", action="store_true",
                    help="Save binary mask PNGs")
    p.add_argument("--no-smoke", action="store_true",
                    help="Disable smoke segmentation (fire only)")
    p.add_argument("--force-download", action="store_true",
                    help="Re-download dataset")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    import cv2

    from config import FireConfig, load_config
    from controller import FireController
    from export import CSVExporter, export_frame_json, save_mask
    from validator import is_image, is_video, validate_source
    from visualize import compose_report, draw_fire_overlay

    if args.force_download:
        from data_bootstrap import ensure_fire_dataset
        ensure_fire_dataset(force=True)

    cfg = load_config(args.config) if args.config else FireConfig()
    cfg.output_dir = args.output_dir
    if args.no_smoke:
        cfg.enable_smoke = False
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vr = validate_source(args.source)
    if not vr.ok:
        for w in vr.warnings:
            print(f"ERROR: {w}", file=sys.stderr)
        sys.exit(1)

    ctrl = FireController(cfg)
    ctrl.load()

    csv_exp = CSVExporter(args.export_csv) if args.export_csv else None

    try:
        source = args.source
        if source.isdigit():
            _run_video(ctrl, int(source), cfg, args, out_dir, csv_exp,
                       cv2, compose_report, export_frame_json, save_mask,
                       is_webcam=True)
        elif Path(source).is_dir():
            _run_directory(ctrl, Path(source), cfg, args, out_dir, csv_exp,
                           cv2, compose_report, export_frame_json, save_mask)
        elif is_video(source):
            _run_video(ctrl, source, cfg, args, out_dir, csv_exp,
                       cv2, compose_report, export_frame_json, save_mask,
                       is_webcam=False)
        elif is_image(source):
            _run_image(ctrl, Path(source), cfg, args, out_dir, csv_exp,
                       cv2, compose_report, export_frame_json, save_mask)
        else:
            print(f"Unsupported source: {source}", file=sys.stderr)
            sys.exit(1)
    finally:
        if csv_exp:
            csv_exp.close()
        ctrl.close()
        if not args.no_display:
            cv2.destroyAllWindows()


# ── source handlers ────────────────────────────────────────


def _run_image(ctrl, path, cfg, args, out_dir, csv_exp,
               cv2, compose_report, export_frame_json, save_mask_fn):
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"Cannot read image: {path}", file=sys.stderr)
        return

    result = ctrl.process(frame)
    m = result.metrics
    alert = result.alert
    print(f"{path.name}: fire={m.fire_coverage:.2%} ({m.fire_count} rgn)  "
          f"smoke={m.smoke_coverage:.2%}  alert={alert.level.value}")

    vis = compose_report(frame, result.segmentation, m, result.trend, cfg)

    if not args.no_display:
        cv2.imshow("Fire Segmentation", vis)
        cv2.waitKey(0)

    if args.save_annotated:
        cv2.imwrite(str(out_dir / f"{path.stem}_annotated.jpg"), vis)

    if args.save_masks:
        save_mask_fn(out_dir / f"{path.stem}_fire_mask.png",
                     result.segmentation.fire_mask)
        if cfg.enable_smoke:
            save_mask_fn(out_dir / f"{path.stem}_smoke_mask.png",
                         result.segmentation.smoke_mask)

    if args.export_json:
        jp = args.export_json
        if Path(jp).is_dir() or jp.endswith("/") or jp.endswith("\\"):
            jp = str(Path(jp) / f"{path.stem}.json")
        export_frame_json(jp, result.segmentation, m, result.trend,
                          source=str(path))

    if csv_exp:
        csv_exp.write_row(_csv_row(result, str(path)))


def _run_directory(ctrl, dir_path, cfg, args, out_dir, csv_exp,
                   cv2, compose_report, export_frame_json, save_mask_fn):
    from validator import _IMAGE_EXTS
    images = sorted(f for f in dir_path.iterdir() if f.suffix.lower() in _IMAGE_EXTS)
    print(f"Processing {len(images)} image(s) from {dir_path}")

    ctrl.reset_trend()

    for idx, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  [SKIP] Cannot read: {img_path.name}")
            continue

        result = ctrl.process(frame)
        m = result.metrics
        print(f"  [{idx + 1}/{len(images)}] {img_path.name}: "
              f"fire={m.fire_coverage:.2%}  alert={result.alert.level.value}")

        if args.save_annotated:
            vis = compose_report(frame, result.segmentation, m, result.trend, cfg)
            cv2.imwrite(str(out_dir / f"{img_path.stem}_annotated.jpg"), vis)

        if args.save_masks:
            save_mask_fn(out_dir / f"{img_path.stem}_fire_mask.png",
                         result.segmentation.fire_mask)

        if args.export_json:
            jp = args.export_json
            if Path(jp).is_dir() or jp.endswith("/") or jp.endswith("\\"):
                Path(jp).mkdir(parents=True, exist_ok=True)
                jp = str(Path(jp) / f"{img_path.stem}.json")
            else:
                jp = str(out_dir / f"{img_path.stem}.json")
            export_frame_json(jp, result.segmentation, m, result.trend,
                              source=str(img_path))

        if csv_exp:
            csv_exp.write_row(_csv_row(result, str(img_path), frame_idx=idx))

    print(f"\nDone -- processed {len(images)} image(s)")
    _print_trend_summary(ctrl)


def _run_video(ctrl, source, cfg, args, out_dir, csv_exp,
               cv2, compose_report, export_frame_json, save_mask_fn,
               is_webcam):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot open source: {source}", file=sys.stderr)
        return

    ctrl.reset_trend()

    writer = None
    if args.save_annotated:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vid_name = "webcam" if is_webcam else Path(str(source)).stem
        writer = cv2.VideoWriter(
            str(out_dir / f"{vid_name}_annotated.mp4"), fourcc, fps, (w, h),
        )

    idx = 0
    last_result = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if is_webcam:
                frame = cv2.flip(frame, 1)

            result = ctrl.process(frame)
            last_result = result
            m = result.metrics

            if idx % 30 == 0:
                print(f"  [frame {idx}] fire={m.fire_coverage:.2%} "
                      f"smoke={m.smoke_coverage:.2%}  "
                      f"alert={result.alert.level.value}")

            vis = compose_report(frame, result.segmentation, m, result.trend, cfg)

            if not args.no_display:
                # Use only the overlay portion for display (not trend panel)
                from visualize import draw_fire_overlay
                disp = draw_fire_overlay(frame, result.segmentation, m, cfg)
                cv2.imshow("Fire Segmentation", disp)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            if writer:
                # Write the overlay-only frame to keep original resolution
                from visualize import draw_fire_overlay
                writer.write(draw_fire_overlay(frame, result.segmentation, m, cfg))

            if csv_exp:
                csv_exp.write_row(_csv_row(result, str(source), frame_idx=idx))

            idx += 1
    finally:
        cap.release()
        if writer:
            writer.release()

    if args.export_json and last_result is not None:
        src_name = "webcam" if is_webcam else Path(str(source)).stem
        export_frame_json(
            args.export_json, last_result.segmentation, last_result.metrics,
            last_result.trend, source=str(source), frame_idx=idx - 1,
        )

    print(f"\nDone -- processed {idx} frame(s)")
    _print_trend_summary(ctrl)


# ── helpers ────────────────────────────────────────────────


def _csv_row(result, source, frame_idx=None):
    m = result.metrics
    row = {
        "source": source,
        "fire_area_px": m.fire_area_px,
        "smoke_area_px": m.smoke_area_px,
        "fire_coverage": round(m.fire_coverage, 6),
        "smoke_coverage": round(m.smoke_coverage, 6),
        "fire_count": m.fire_count,
        "smoke_count": m.smoke_count,
        "alert": result.alert.level.value,
    }
    if frame_idx is not None:
        row["frame_idx"] = frame_idx
    return row


def _print_trend_summary(ctrl):
    """Print final trend summary if available."""
    # Access the latest trend from the tracker
    if ctrl._trend._history:
        t = ctrl._trend._compute()
        print(f"\nTrend summary ({t.frames_seen} frames):")
        print(f"  Avg fire coverage:  {t.avg_fire_coverage:.2%}")
        print(f"  Peak fire coverage: {t.peak_fire_coverage:.2%}")
        print(f"  Fire growth rate:   {t.fire_growth_rate:+.4%}/frame")


if __name__ == "__main__":
    main()
