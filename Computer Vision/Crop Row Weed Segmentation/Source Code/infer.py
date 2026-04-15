"""Crop Row & Weed Segmentation — CLI entry point.
"""Crop Row & Weed Segmentation — CLI entry point.

Usage::

    # Single image
    python infer.py --source field.jpg

    # Video
    python infer.py --source drone_clip.mp4 --save-annotated

    # Webcam
    python infer.py --source 0

    # Directory of images
    python infer.py --source images/ --save-annotated --save-masks

    # Full export
    python infer.py --source field.jpg \
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
        description="Crop Row & Weed Segmentation -- multi-class agricultural segmentation",
    )
    p.add_argument(
        "--source", default="0",
        help="'0' for webcam, or path to image/video/directory (default: 0)",
    )
    p.add_argument("--config", default=None, help="YAML/JSON config path")
    p.add_argument("--no-display", action="store_true", help="Headless mode")
    p.add_argument("--output-dir", default="output", help="Output directory")
    p.add_argument("--export-json", default=None,
                    help="JSON export path (or directory for batch)")
    p.add_argument("--export-csv", default=None, help="CSV export path")
    p.add_argument("--save-annotated", action="store_true",
                    help="Save annotated images/video")
    p.add_argument("--save-masks", action="store_true",
                    help="Save per-class binary mask PNGs")
    p.add_argument("--force-download", action="store_true",
                    help="Re-download dataset")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    import cv2

    from config import CropWeedConfig, load_config
    from controller import CropWeedController
    from export import CSVExporter, export_json, save_masks
    from validator import is_image, is_video, validate_source
    from visualize import draw_overlay, render_class_masks

    if args.force_download:
        from data_bootstrap import ensure_cropweed_dataset
        ensure_cropweed_dataset(force=True)

    cfg = load_config(args.config) if args.config else CropWeedConfig()
    cfg.output_dir = args.output_dir
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vr = validate_source(args.source)
    if not vr.ok:
        for w in vr.warnings:
            print(f"ERROR: {w}", file=sys.stderr)
        sys.exit(1)

    ctrl = CropWeedController(cfg)
    ctrl.load()

    csv_exp = CSVExporter(args.export_csv, cfg.class_names) if args.export_csv else None

    try:
        source = args.source
        if source.isdigit():
            _run_video(ctrl, int(source), cfg, args, out_dir, csv_exp,
                       cv2, draw_overlay, export_json, save_masks, is_webcam=True)
        elif Path(source).is_dir():
            _run_directory(ctrl, Path(source), cfg, args, out_dir, csv_exp,
                           cv2, draw_overlay, export_json, save_masks)
        elif is_video(source):
            _run_video(ctrl, source, cfg, args, out_dir, csv_exp,
                       cv2, draw_overlay, export_json, save_masks, is_webcam=False)
        elif is_image(source):
            _run_image(ctrl, Path(source), cfg, args, out_dir, csv_exp,
                       cv2, draw_overlay, export_json, save_masks)
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
               cv2, draw_overlay, export_json, save_masks_fn):
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"Cannot read image: {path}", file=sys.stderr)
        return

    result = ctrl.process(frame)
    _print_summary(result, str(path.name))

    vis = draw_overlay(frame, result.segmentation, result.area_report, cfg)

    if not args.no_display:
        cv2.imshow("Crop/Weed Segmentation", vis)
        cv2.waitKey(0)

    if args.save_annotated:
        cv2.imwrite(str(out_dir / f"{path.stem}_annotated.jpg"), vis)

    if args.save_masks:
        save_masks_fn(out_dir, result.segmentation, path.stem)

    if args.export_json:
        jp = args.export_json
        if Path(jp).is_dir() or jp.endswith("/") or jp.endswith("\\"):
            jp = str(Path(jp) / f"{path.stem}.json")
        export_json(jp, result.segmentation, result.area_report, source=str(path))

    if csv_exp:
        csv_exp.write_row(result.area_report, source=str(path))


def _run_directory(ctrl, dir_path, cfg, args, out_dir, csv_exp,
                   cv2, draw_overlay, export_json, save_masks_fn):
    from validator import _IMAGE_EXTS
    images = sorted(f for f in dir_path.iterdir() if f.suffix.lower() in _IMAGE_EXTS)
    print(f"Processing {len(images)} image(s) from {dir_path}")

    for idx, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  [SKIP] Cannot read: {img_path.name}")
            continue

        result = ctrl.process(frame)
        _print_summary(result, img_path.name, prefix=f"  [{idx + 1}/{len(images)}] ")

        if args.save_annotated:
            vis = draw_overlay(frame, result.segmentation, result.area_report, cfg)
            cv2.imwrite(str(out_dir / f"{img_path.stem}_annotated.jpg"), vis)

        if args.save_masks:
            save_masks_fn(out_dir, result.segmentation, img_path.stem)

        if args.export_json:
            jp = args.export_json
            if Path(jp).is_dir() or jp.endswith("/") or jp.endswith("\\"):
                Path(jp).mkdir(parents=True, exist_ok=True)
                jp = str(Path(jp) / f"{img_path.stem}.json")
            else:
                jp = str(out_dir / f"{img_path.stem}.json")
            export_json(jp, result.segmentation, result.area_report, source=str(img_path))

        if csv_exp:
            csv_exp.write_row(result.area_report, source=str(img_path), frame_idx=idx)

    print(f"\nDone -- processed {len(images)} image(s)")


def _run_video(ctrl, source, cfg, args, out_dir, csv_exp,
               cv2, draw_overlay, export_json, save_masks_fn, is_webcam):
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
        writer = cv2.VideoWriter(str(out_dir / f"{vid_name}_annotated.mp4"), fourcc, fps, (w, h))

    idx = 0
    result = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if is_webcam:
                frame = cv2.flip(frame, 1)

            result = ctrl.process(frame)

            if idx % 30 == 0:
                _print_summary(result, str(source), prefix=f"  [frame {idx}] ")

            vis = draw_overlay(frame, result.segmentation, result.area_report, cfg)

            if not args.no_display:
                cv2.imshow("Crop/Weed Segmentation", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            if writer:
                writer.write(vis)

            if csv_exp:
                csv_exp.write_row(result.area_report, source=str(source), frame_idx=idx)

            idx += 1
    finally:
        cap.release()
        if writer:
            writer.release()

    if args.export_json and result is not None:
        src_name = "webcam" if is_webcam else Path(str(source)).stem
        export_json(
            args.export_json, result.segmentation, result.area_report,
            source=str(source), frame_idx=idx - 1,
        )

    print(f"\nDone -- processed {idx} frame(s)")


def _print_summary(result, name, prefix=""):
    ar = result.area_report
    parts = [f"{name}: {ar.total_instances} instance(s)"]
    for cn, cs in ar.per_class.items():
        if cs.instance_count > 0:
            parts.append(f"{cn}={cs.instance_count}({cs.coverage_ratio:.1%})")
    parts.append(f"bg={ar.background_ratio:.1%}")
    print(f"{prefix}{' | '.join(parts)}")


if __name__ == "__main__":
    main()
