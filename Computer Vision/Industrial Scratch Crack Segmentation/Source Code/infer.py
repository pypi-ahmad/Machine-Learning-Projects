"""Industrial Scratch / Crack Segmentation — CLI entry point.
"""Industrial Scratch / Crack Segmentation — CLI entry point.

Usage::

    # Single image
    python infer.py --source surface.jpg

    # Directory of images
    python infer.py --source images/ --save-annotated --save-masks

    # Video / webcam
    python infer.py --source 0
    python infer.py --source inspection.mp4 --save-annotated

    # Full export
    python infer.py --source surface.jpg \\
        --export-json report.json --export-csv stats.csv \\
        --save-annotated --save-masks
"""
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Industrial Scratch / Crack Segmentation -- surface defect detection",
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
    p.add_argument("--force-download", action="store_true",
                    help="Re-download dataset")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    import cv2

    from config import DefectConfig, load_config
    from controller import DefectController
    from export import CSVExporter, export_frame_json, save_mask
    from validator import is_image, is_video, validate_source
    from visualize import draw_defect_overlay

    if args.force_download:
        from data_bootstrap import ensure_defect_dataset
        ensure_defect_dataset(force=True)

    cfg = load_config(args.config) if args.config else DefectConfig()
    cfg.output_dir = args.output_dir
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vr = validate_source(args.source)
    if not vr.ok:
        for w in vr.warnings:
            print(f"ERROR: {w}", file=sys.stderr)
        sys.exit(1)

    ctrl = DefectController(cfg)
    ctrl.load()

    csv_exp = CSVExporter(args.export_csv) if args.export_csv else None

    try:
        source = args.source
        if source.isdigit():
            _run_video(ctrl, int(source), cfg, args, out_dir, csv_exp,
                       cv2, draw_defect_overlay, export_frame_json, save_mask,
                       is_webcam=True)
        elif Path(source).is_dir():
            _run_directory(ctrl, Path(source), cfg, args, out_dir, csv_exp,
                           cv2, draw_defect_overlay, export_frame_json,
                           save_mask)
        elif is_video(source):
            _run_video(ctrl, source, cfg, args, out_dir, csv_exp,
                       cv2, draw_defect_overlay, export_frame_json, save_mask,
                       is_webcam=False)
        elif is_image(source):
            _run_image(ctrl, Path(source), cfg, args, out_dir, csv_exp,
                       cv2, draw_defect_overlay, export_frame_json, save_mask)
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
               cv2, draw_defect_overlay, export_frame_json, save_mask_fn):
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"Cannot read image: {path}", file=sys.stderr)
        return

    result = ctrl.process(frame, source=str(path))
    m = result.metrics
    print(f"{path.name}: {m.defect_count} defect(s) "
          f"severity={m.severity.upper()} coverage={m.defect_coverage:.2%} "
          f"max_len={m.max_length_px:.0f}px")

    vis = draw_defect_overlay(frame, result.segmentation, m, cfg)

    if not args.no_display:
        cv2.imshow("Defect Segmentation", vis)
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
                   cv2, draw_defect_overlay, export_frame_json,
                   save_mask_fn):
    from validator import _IMAGE_EXTS
    images = sorted(f for f in dir_path.iterdir() if f.suffix.lower() in _IMAGE_EXTS)
    print(f"Processing {len(images)} image(s) from {dir_path}")

    severity_counts = {"none": 0, "low": 0, "medium": 0, "high": 0}
    for idx, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  [SKIP] Cannot read: {img_path.name}")
            continue

        result = ctrl.process(frame, source=str(img_path))
        m = result.metrics
        severity_counts[m.severity] = severity_counts.get(m.severity, 0) + 1

        print(f"  [{idx + 1}/{len(images)}] {img_path.name}: "
              f"{m.defect_count} defect(s) severity={m.severity.upper()} "
              f"coverage={m.defect_coverage:.2%}")

        if args.save_annotated:
            vis = draw_defect_overlay(frame, result.segmentation, m, cfg)
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
            csv_exp.write_row(_csv_row(result, str(img_path)))

    print(f"\nDone -- processed {len(images)} image(s)")
    print(f"Severity distribution: {severity_counts}")


def _run_video(ctrl, source, cfg, args, out_dir, csv_exp,
               cv2, draw_defect_overlay, export_frame_json, save_mask_fn,
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
                print(f"  [frame {idx}] defects={m.defect_count} "
                      f"severity={m.severity.upper()}")

            vis = draw_defect_overlay(frame, result.segmentation, m, cfg)

            if not args.no_display:
                cv2.imshow("Defect Segmentation", vis)
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

    print(f"\nDone -- processed {idx} frame(s)")


# ── helpers ────────────────────────────────────────────────


def _csv_row(result, source, frame_idx=None):
    m = result.metrics
    row = {
        "source": source,
        "defect_count": m.defect_count,
        "total_defect_area_px": m.total_defect_area_px,
        "defect_coverage": round(m.defect_coverage, 6),
        "severity": m.severity,
        "max_length_px": m.max_length_px,
        "mean_length_px": m.mean_length_px,
        "max_aspect_ratio": m.max_aspect_ratio,
        "mean_confidence": m.mean_confidence,
    }
    if frame_idx is not None:
        row["frame_idx"] = frame_idx
    return row


if __name__ == "__main__":
    main()
