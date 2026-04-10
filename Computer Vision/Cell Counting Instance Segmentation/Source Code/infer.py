"""Cell Counting Instance Segmentation — CLI entry point.

Usage::

    # Single image
    python infer.py --source cells.png

    # Directory of images
    python infer.py --source images/ --save-annotated --save-masks

    # Video / webcam
    python infer.py --source 0
    python infer.py --source microscopy.mp4 --save-annotated

    # Full export
    python infer.py --source cells.png \\
        --export-json report.json --export-csv stats.csv \\
        --save-annotated --save-masks

    # Disable watershed splitting
    python infer.py --source cells.png --no-watershed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cell Counting Instance Segmentation — segment and count cells/nuclei",
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
    p.add_argument("--no-watershed", action="store_true",
                    help="Disable watershed splitting of touching cells")
    p.add_argument("--force-download", action="store_true",
                    help="Re-download dataset")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    import cv2

    from config import CellConfig, load_config
    from controller import CellController
    from export import CSVExporter, export_frame_json, save_mask
    from validator import is_image, is_video, validate_source
    from visualize import draw_cell_overlay, draw_count_badge

    if args.force_download:
        from data_bootstrap import ensure_cell_dataset
        ensure_cell_dataset(force=True)

    cfg = load_config(args.config) if args.config else CellConfig()
    cfg.output_dir = args.output_dir
    if args.no_watershed:
        cfg.watershed_split = False
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vr = validate_source(args.source)
    if not vr.ok:
        for w in vr.warnings:
            print(f"ERROR: {w}", file=sys.stderr)
        sys.exit(1)

    ctrl = CellController(cfg)
    ctrl.load()

    csv_exp = CSVExporter(args.export_csv) if args.export_csv else None

    try:
        source = args.source
        if source.isdigit():
            _run_video(ctrl, int(source), cfg, args, out_dir, csv_exp,
                       cv2, draw_cell_overlay, draw_count_badge,
                       export_frame_json, save_mask, is_webcam=True)
        elif Path(source).is_dir():
            _run_directory(ctrl, Path(source), cfg, args, out_dir, csv_exp,
                           cv2, draw_cell_overlay, draw_count_badge,
                           export_frame_json, save_mask)
        elif is_video(source):
            _run_video(ctrl, source, cfg, args, out_dir, csv_exp,
                       cv2, draw_cell_overlay, draw_count_badge,
                       export_frame_json, save_mask, is_webcam=False)
        elif is_image(source):
            _run_image(ctrl, Path(source), cfg, args, out_dir, csv_exp,
                       cv2, draw_cell_overlay, draw_count_badge,
                       export_frame_json, save_mask)
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
               cv2, draw_cell_overlay, draw_count_badge,
               export_frame_json, save_mask_fn):
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"Cannot read image: {path}", file=sys.stderr)
        return

    result = ctrl.process(frame, source=str(path))
    m = result.metrics
    raw_n = result.raw_segmentation.count
    print(f"{path.name}: {m.cell_count} cell(s) "
          f"(raw={raw_n}, coverage={m.cell_coverage:.2%}, "
          f"mean_area={m.mean_cell_area_px:.0f}px)")

    vis = draw_cell_overlay(frame, result.segmentation, m, cfg)
    vis = draw_count_badge(vis, m.cell_count)

    if not args.no_display:
        cv2.imshow("Cell Counting", vis)
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
                   cv2, draw_cell_overlay, draw_count_badge,
                   export_frame_json, save_mask_fn):
    from validator import _IMAGE_EXTS
    images = sorted(f for f in dir_path.iterdir() if f.suffix.lower() in _IMAGE_EXTS)
    print(f"Processing {len(images)} image(s) from {dir_path}")

    total_cells = 0
    for idx, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  [SKIP] Cannot read: {img_path.name}")
            continue

        result = ctrl.process(frame, source=str(img_path))
        m = result.metrics
        total_cells += m.cell_count

        print(f"  [{idx + 1}/{len(images)}] {img_path.name}: "
              f"{m.cell_count} cell(s) (coverage={m.cell_coverage:.2%})")

        if args.save_annotated:
            vis = draw_cell_overlay(frame, result.segmentation, m, cfg)
            vis = draw_count_badge(vis, m.cell_count)
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

    print(f"\nDone — processed {len(images)} image(s), "
          f"total cells counted: {total_cells}")


def _run_video(ctrl, source, cfg, args, out_dir, csv_exp,
               cv2, draw_cell_overlay, draw_count_badge,
               export_frame_json, save_mask_fn, is_webcam):
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
                print(f"  [frame {idx}] cells={m.cell_count} "
                      f"(coverage={m.cell_coverage:.2%})")

            vis = draw_cell_overlay(frame, result.segmentation, m, cfg)
            vis = draw_count_badge(vis, m.cell_count)

            if not args.no_display:
                cv2.imshow("Cell Counting", vis)
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
        "cell_count": m.cell_count,
        "total_cell_area_px": m.total_cell_area_px,
        "cell_coverage": round(m.cell_coverage, 6),
        "mean_cell_area_px": m.mean_cell_area_px,
        "median_cell_area_px": m.median_cell_area_px,
        "min_cell_area_px": m.min_cell_area_px,
        "max_cell_area_px": m.max_cell_area_px,
        "mean_confidence": m.mean_confidence,
        "raw_count": result.raw_segmentation.count,
    }
    if frame_idx is not None:
        row["frame_idx"] = frame_idx
    return row


if __name__ == "__main__":
    main()
