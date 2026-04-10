"""Wound Area Measurement — CLI entry point.

Usage::

    # Single image
    python infer.py --source wound.jpg

    # Directory of images (with change tracking)
    python infer.py --source images/ --track-changes \
        --save-annotated --save-masks

    # Video / webcam
    python infer.py --source 0
    python infer.py --source clip.mp4 --save-annotated

    # Full export
    python infer.py --source wound.jpg \
        --export-json report.json --export-csv stats.csv \
        --save-annotated --save-masks

DISCLAIMER: This tool produces relative pixel-based estimates only.
It is NOT a medical device and must NOT be used for clinical diagnosis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Wound Area Measurement — wound segmentation and relative area estimation",
        epilog="DISCLAIMER: Informational only. NOT for clinical diagnosis.",
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
    p.add_argument("--track-changes", action="store_true",
                    help="Track wound area changes across images in a directory")
    p.add_argument("--force-download", action="store_true",
                    help="Re-download dataset")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    print("DISCLAIMER: This tool produces relative pixel-based estimates only.")
    print("It is NOT a medical device and must NOT be used for clinical diagnosis.\n")

    import cv2

    from config import WoundConfig, load_config
    from controller import WoundController
    from export import CSVExporter, export_frame_json, export_series_json, save_mask
    from validator import is_image, is_video, validate_source
    from visualize import compose_report, draw_wound_overlay

    if args.force_download:
        from data_bootstrap import ensure_wound_dataset
        ensure_wound_dataset(force=True)

    cfg = load_config(args.config) if args.config else WoundConfig()
    cfg.output_dir = args.output_dir
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vr = validate_source(args.source)
    if not vr.ok:
        for w in vr.warnings:
            print(f"ERROR: {w}", file=sys.stderr)
        sys.exit(1)

    ctrl = WoundController(cfg)
    ctrl.load()

    csv_exp = CSVExporter(args.export_csv) if args.export_csv else None

    try:
        source = args.source
        if source.isdigit():
            _run_video(ctrl, int(source), cfg, args, out_dir, csv_exp,
                       cv2, draw_wound_overlay, export_frame_json, save_mask,
                       is_webcam=True)
        elif Path(source).is_dir():
            _run_directory(ctrl, Path(source), cfg, args, out_dir, csv_exp,
                           cv2, compose_report, export_frame_json,
                           export_series_json, save_mask)
        elif is_video(source):
            _run_video(ctrl, source, cfg, args, out_dir, csv_exp,
                       cv2, draw_wound_overlay, export_frame_json, save_mask,
                       is_webcam=False)
        elif is_image(source):
            _run_image(ctrl, Path(source), cfg, args, out_dir, csv_exp,
                       cv2, draw_wound_overlay, export_frame_json, save_mask)
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
               cv2, draw_wound_overlay, export_frame_json, save_mask_fn):
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"Cannot read image: {path}", file=sys.stderr)
        return

    result = ctrl.process(frame, source=str(path))
    m = result.metrics
    print(f"{path.name}: wound={m.wound_coverage:.2%} ({m.wound_count} region(s), "
          f"{m.wound_area_px:,}px)")

    vis = draw_wound_overlay(frame, result.segmentation, m, cfg)

    if not args.no_display:
        cv2.imshow("Wound Segmentation", vis)
        cv2.waitKey(0)

    if args.save_annotated:
        cv2.imwrite(str(out_dir / f"{path.stem}_annotated.jpg"), vis)

    if args.save_masks:
        save_mask_fn(out_dir / f"{path.stem}_mask.png",
                     result.segmentation.combined_mask)

    if args.export_json:
        jp = args.export_json
        if Path(jp).is_dir() or jp.endswith("/") or jp.endswith("\\"):
            jp = str(Path(jp) / f"{path.stem}.json")
        export_frame_json(jp, result.segmentation, m, source=str(path))

    if csv_exp:
        csv_exp.write_row(_csv_row(result, str(path)))


def _run_directory(ctrl, dir_path, cfg, args, out_dir, csv_exp,
                   cv2, compose_report, export_frame_json,
                   export_series_json, save_mask_fn):
    from validator import _IMAGE_EXTS
    images = sorted(f for f in dir_path.iterdir() if f.suffix.lower() in _IMAGE_EXTS)
    print(f"Processing {len(images)} image(s) from {dir_path}")

    track = args.track_changes
    if track:
        ctrl.reset_tracker()

    for idx, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  [SKIP] Cannot read: {img_path.name}")
            continue

        result = ctrl.process(frame, source=str(img_path), track=track)
        m = result.metrics
        delta_str = ""
        if result.change_entry and result.change_entry.index > 0:
            delta_str = f"  Δ={result.change_entry.delta_px:+,}px"
        print(f"  [{idx + 1}/{len(images)}] {img_path.name}: "
              f"wound={m.wound_coverage:.2%} ({m.wound_area_px:,}px){delta_str}")

        summary = ctrl.change_summary() if track else None

        if args.save_annotated:
            vis = compose_report(frame, result.segmentation, m, summary, cfg)
            cv2.imwrite(str(out_dir / f"{img_path.stem}_annotated.jpg"), vis)

        if args.save_masks:
            save_mask_fn(out_dir / f"{img_path.stem}_mask.png",
                         result.segmentation.combined_mask)

        if args.export_json:
            jp = args.export_json
            if Path(jp).is_dir() or jp.endswith("/") or jp.endswith("\\"):
                Path(jp).mkdir(parents=True, exist_ok=True)
                jp = str(Path(jp) / f"{img_path.stem}.json")
            else:
                jp = str(out_dir / f"{img_path.stem}.json")
            export_frame_json(jp, result.segmentation, m, source=str(img_path))

        if csv_exp:
            csv_exp.write_row(_csv_row(result, str(img_path), frame_idx=idx))

    print(f"\nDone — processed {len(images)} image(s)")

    if track:
        summary = ctrl.change_summary()
        _print_change_summary(summary)
        if args.export_json:
            series_path = str(out_dir / "series_summary.json")
            export_series_json(series_path, summary)
            print(f"Series summary saved to {series_path}")


def _run_video(ctrl, source, cfg, args, out_dir, csv_exp,
               cv2, draw_wound_overlay, export_frame_json, save_mask_fn,
               is_webcam):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot open source: {source}", file=sys.stderr)
        return

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

            result = ctrl.process(frame, source=str(source))
            last_result = result
            m = result.metrics

            if idx % 30 == 0:
                print(f"  [frame {idx}] wound={m.wound_coverage:.2%} "
                      f"({m.wound_count} region(s))")

            vis = draw_wound_overlay(frame, result.segmentation, m, cfg)

            if not args.no_display:
                cv2.imshow("Wound Segmentation", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            if writer:
                writer.write(vis)

            if csv_exp:
                csv_exp.write_row(_csv_row(result, str(source), frame_idx=idx))

            idx += 1
    finally:
        cap.release()
        if writer:
            writer.release()

    if args.export_json and last_result is not None:
        export_frame_json(
            args.export_json, last_result.segmentation, last_result.metrics,
            source=str(source), frame_idx=idx - 1,
        )

    print(f"\nDone — processed {idx} frame(s)")


# ── helpers ────────────────────────────────────────────────


def _csv_row(result, source, frame_idx=None):
    m = result.metrics
    row = {
        "source": source,
        "wound_area_px": m.wound_area_px,
        "wound_coverage": round(m.wound_coverage, 6),
        "wound_count": m.wound_count,
        "mean_confidence": round(m.mean_confidence, 4),
        "largest_wound_px": m.largest_wound_px,
    }
    if frame_idx is not None:
        row["frame_idx"] = frame_idx
    if result.change_entry is not None:
        row["delta_px"] = result.change_entry.delta_px
        row["delta_ratio"] = result.change_entry.delta_ratio
    return row


def _print_change_summary(summary):
    print(f"\nChange summary ({summary.total_images} images):")
    print(f"  Initial area: {summary.initial_area_px:,}px")
    print(f"  Final area:   {summary.final_area_px:,}px")
    print(f"  Net change:   {summary.net_change_px:+,}px "
          f"({summary.net_change_ratio:+.2%})")
    print(f"  Peak area:    {summary.peak_area_px:,}px")


if __name__ == "__main__":
    main()
