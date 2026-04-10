"""Waterbody & Flood Extent Segmentation — CLI entry point.

Usage::

    # Single image — detect water bodies
    python infer.py --source satellite.jpg

    # Video / webcam
    python infer.py --source 0
    python infer.py --source drone_clip.mp4 --save-annotated

    # Directory of images
    python infer.py --source images/ --save-annotated --save-masks

    # Before / after flood comparison (single pair)
    python infer.py --before pre_flood.jpg --after post_flood.jpg

    # Before / after flood comparison (batch — matched filenames)
    python infer.py --before-dir images/pre --after-dir images/post

    # Full export
    python infer.py --source satellite.jpg \
        --export-json report.json --export-csv stats.csv \
        --save-annotated --save-masks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Waterbody & Flood Extent Segmentation — water detection and flood comparison",
    )
    # ── single-image / video / webcam ─────────────────────
    p.add_argument(
        "--source", default=None,
        help="'0' for webcam, or path to image/video/directory",
    )

    # ── before/after comparison ───────────────────────────
    g = p.add_argument_group("comparison mode (single pair)")
    g.add_argument("--before", default=None, help="Path to before image")
    g.add_argument("--after", default=None, help="Path to after image")

    g2 = p.add_argument_group("comparison mode (batch)")
    g2.add_argument("--before-dir", default=None, help="Directory of before images")
    g2.add_argument("--after-dir", default=None, help="Directory of after images")

    # ── options ────────────────────────────────────────────
    p.add_argument("--config", default=None, help="YAML/JSON config path")
    p.add_argument("--no-display", action="store_true", help="Headless mode")
    p.add_argument("--output-dir", default="output", help="Output directory")

    # ── export ─────────────────────────────────────────────
    p.add_argument("--export-json", default=None, help="JSON export path")
    p.add_argument("--export-csv", default=None, help="CSV export path")
    p.add_argument("--save-annotated", action="store_true",
                    help="Save annotated images/video")
    p.add_argument("--save-masks", action="store_true",
                    help="Save binary mask PNGs")

    # ── data ───────────────────────────────────────────────
    p.add_argument("--force-download", action="store_true",
                    help="Re-download dataset")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    import cv2

    from config import FloodConfig, load_config
    from controller import WaterFloodController
    from export import CSVExporter, export_comparison_json, export_single_json, save_mask
    from validator import is_image, is_video, validate_directory_pair, validate_pair, validate_source
    from visualize import compose_comparison_report, draw_water_overlay

    if args.force_download:
        from data_bootstrap import ensure_flood_dataset
        ensure_flood_dataset(force=True)

    cfg = load_config(args.config) if args.config else FloodConfig()
    cfg.output_dir = args.output_dir
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ctrl = WaterFloodController(cfg)
    ctrl.load()

    csv_exp = CSVExporter(args.export_csv) if args.export_csv else None

    try:
        # ── decide mode ────────────────────────────────────
        if args.before and args.after:
            _run_pair(ctrl, Path(args.before), Path(args.after),
                      cfg, args, out_dir, csv_exp, cv2,
                      draw_water_overlay, compose_comparison_report,
                      export_comparison_json, save_mask)
        elif args.before_dir and args.after_dir:
            _run_batch_pairs(ctrl, args.before_dir, args.after_dir,
                             cfg, args, out_dir, csv_exp, cv2,
                             draw_water_overlay, compose_comparison_report,
                             export_comparison_json, save_mask,
                             validate_directory_pair)
        elif args.source is not None:
            vr = validate_source(args.source)
            if not vr.ok:
                for w in vr.warnings:
                    print(f"ERROR: {w}", file=sys.stderr)
                sys.exit(1)
            source = args.source
            if source.isdigit():
                _run_video(ctrl, int(source), cfg, args, out_dir, csv_exp,
                           cv2, draw_water_overlay, export_single_json,
                           save_mask, is_webcam=True)
            elif Path(source).is_dir():
                _run_directory(ctrl, Path(source), cfg, args, out_dir, csv_exp,
                               cv2, draw_water_overlay, export_single_json,
                               save_mask)
            elif is_video(source):
                _run_video(ctrl, source, cfg, args, out_dir, csv_exp,
                           cv2, draw_water_overlay, export_single_json,
                           save_mask, is_webcam=False)
            elif is_image(source):
                _run_image(ctrl, Path(source), cfg, args, out_dir, csv_exp,
                           cv2, draw_water_overlay, export_single_json,
                           save_mask)
            else:
                print(f"Unsupported source: {source}", file=sys.stderr)
                sys.exit(1)
        else:
            print(
                "Provide --source OR --before/--after OR --before-dir/--after-dir",
                file=sys.stderr,
            )
            sys.exit(1)
    finally:
        if csv_exp:
            csv_exp.close()
        ctrl.close()
        if not args.no_display:
            cv2.destroyAllWindows()


# ── single-image mode ───────────────────────────────────────


def _run_image(ctrl, path, cfg, args, out_dir, csv_exp,
               cv2, draw_water_overlay, export_single_json, save_mask_fn):
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"Cannot read image: {path}", file=sys.stderr)
        return

    result = ctrl.process(frame)
    cov = result.coverage
    print(f"{path.name}: {cov.instance_count} water region(s), "
          f"coverage={cov.coverage_ratio:.2%}")

    vis = draw_water_overlay(frame, result.segmentation, cov, cfg)

    if not args.no_display:
        cv2.imshow("Water Segmentation", vis)
        cv2.waitKey(0)

    if args.save_annotated:
        cv2.imwrite(str(out_dir / f"{path.stem}_annotated.jpg"), vis)

    if args.save_masks:
        save_mask_fn(out_dir / f"{path.stem}_mask.png", result.segmentation.combined_mask)

    if args.export_json:
        jp = args.export_json
        if Path(jp).is_dir() or jp.endswith("/") or jp.endswith("\\"):
            jp = str(Path(jp) / f"{path.stem}.json")
        export_single_json(jp, result.segmentation, cov, source=str(path))

    if csv_exp:
        csv_exp.write_row({
            "source": str(path),
            "water_regions": cov.instance_count,
            "water_area_px": cov.water_area_px,
            "coverage_ratio": round(cov.coverage_ratio, 6),
            "mean_confidence": round(cov.mean_confidence, 4),
        })


def _run_directory(ctrl, dir_path, cfg, args, out_dir, csv_exp,
                   cv2, draw_water_overlay, export_single_json, save_mask_fn):
    from validator import _IMAGE_EXTS
    images = sorted(f for f in dir_path.iterdir() if f.suffix.lower() in _IMAGE_EXTS)
    print(f"Processing {len(images)} image(s) from {dir_path}")

    for idx, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  [SKIP] Cannot read: {img_path.name}")
            continue

        result = ctrl.process(frame)
        cov = result.coverage
        print(f"  [{idx + 1}/{len(images)}] {img_path.name}: "
              f"{cov.instance_count} region(s), coverage={cov.coverage_ratio:.2%}")

        if args.save_annotated:
            vis = draw_water_overlay(frame, result.segmentation, cov, cfg)
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
            export_single_json(jp, result.segmentation, cov, source=str(img_path))

        if csv_exp:
            csv_exp.write_row({
                "source": str(img_path),
                "frame_idx": idx,
                "water_regions": cov.instance_count,
                "water_area_px": cov.water_area_px,
                "coverage_ratio": round(cov.coverage_ratio, 6),
                "mean_confidence": round(cov.mean_confidence, 4),
            })

    print(f"\nDone — processed {len(images)} image(s)")


def _run_video(ctrl, source, cfg, args, out_dir, csv_exp,
               cv2, draw_water_overlay, export_single_json,
               save_mask_fn, is_webcam):
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

            result = ctrl.process(frame)
            last_result = result
            cov = result.coverage

            if idx % 30 == 0:
                print(f"  [frame {idx}] {cov.instance_count} region(s), "
                      f"coverage={cov.coverage_ratio:.2%}")

            vis = draw_water_overlay(frame, result.segmentation, cov, cfg)

            if not args.no_display:
                cv2.imshow("Water Segmentation", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            if writer:
                writer.write(vis)

            if csv_exp:
                csv_exp.write_row({
                    "source": str(source),
                    "frame_idx": idx,
                    "water_regions": cov.instance_count,
                    "water_area_px": cov.water_area_px,
                    "coverage_ratio": round(cov.coverage_ratio, 6),
                    "mean_confidence": round(cov.mean_confidence, 4),
                })

            idx += 1
    finally:
        cap.release()
        if writer:
            writer.release()

    print(f"\nDone — processed {idx} frame(s)")


# ── comparison mode ──────────────────────────────────────────


def _run_pair(ctrl, bp, ap, cfg, args, out_dir, csv_exp, cv2,
              draw_water_overlay, compose_comparison_report,
              export_comparison_json, save_mask_fn):
    from validator import validate_pair

    report = validate_pair(bp, ap)
    if not report.ok:
        for w in report.warnings:
            print(f"ERROR: {w}", file=sys.stderr)
        sys.exit(1)

    print(f"Comparing: {bp.name}  ↔  {ap.name}")
    result = ctrl.process_pair(bp, ap)

    fm = result.flood_metrics
    print(f"  Before: {fm.before_water_px:,}px ({fm.before_coverage:.2%})")
    print(f"  After:  {fm.after_water_px:,}px ({fm.after_coverage:.2%})")
    print(f"  New flooding:   {fm.flooded_new_px:,}px ({fm.flood_expansion_ratio:.2%})")
    print(f"  Water receded:  {fm.receded_px:,}px ({fm.recession_ratio:.2%})")
    print(f"  IoU={fm.iou:.4f}  net_change={fm.net_change_ratio:+.4f}")

    if not args.no_display or args.save_annotated:
        report_img = compose_comparison_report(
            result.before_image, result.after_image,
            result.before_seg, result.after_seg,
            result.comparison, result.flood_metrics, cfg,
        )
        if not args.no_display:
            cv2.imshow(f"Flood Comparison — {bp.name}", report_img)
            cv2.waitKey(0)
        if args.save_annotated:
            cv2.imwrite(str(out_dir / f"{bp.stem}_flood_report.jpg"), report_img)

    if args.save_masks:
        save_mask_fn(out_dir / f"{bp.stem}_before_mask.png",
                     result.before_seg.combined_mask)
        save_mask_fn(out_dir / f"{bp.stem}_after_mask.png",
                     result.after_seg.combined_mask)
        save_mask_fn(out_dir / f"{bp.stem}_change_map.png",
                     result.comparison.change_map)

    if args.export_json:
        export_comparison_json(
            args.export_json, result.flood_metrics,
            before_path=str(bp), after_path=str(ap),
        )

    if csv_exp:
        csv_exp.write_row({
            "before": str(bp), "after": str(ap),
            "before_water_px": fm.before_water_px,
            "after_water_px": fm.after_water_px,
            "flooded_new_px": fm.flooded_new_px,
            "receded_px": fm.receded_px,
            "iou": round(fm.iou, 6),
            "net_change": round(fm.net_change_ratio, 6),
        })


def _run_batch_pairs(ctrl, before_dir, after_dir, cfg, args, out_dir, csv_exp,
                     cv2, draw_water_overlay, compose_comparison_report,
                     export_comparison_json, save_mask_fn,
                     validate_directory_pair):
    report, pairs = validate_directory_pair(before_dir, after_dir)
    if not report.ok:
        for w in report.warnings:
            print(f"ERROR: {w}", file=sys.stderr)
        sys.exit(1)
    for w in report.warnings:
        print(f"WARN: {w}")
    print(f"Found {len(pairs)} matched image pair(s)")

    for idx, (bp, ap) in enumerate(pairs):
        print(f"\n[{idx + 1}/{len(pairs)}] {bp.name}  ↔  {ap.name}")

        result = ctrl.process_pair(bp, ap)
        fm = result.flood_metrics

        print(f"  Before={fm.before_water_px:,}px  After={fm.after_water_px:,}px  "
              f"IoU={fm.iou:.4f}  net_change={fm.net_change_ratio:+.4f}")

        if args.save_annotated:
            report_img = compose_comparison_report(
                result.before_image, result.after_image,
                result.before_seg, result.after_seg,
                result.comparison, result.flood_metrics, cfg,
            )
            cv2.imwrite(str(out_dir / f"{bp.stem}_flood_report.jpg"), report_img)

        if args.save_masks:
            save_mask_fn(out_dir / f"{bp.stem}_before_mask.png",
                         result.before_seg.combined_mask)
            save_mask_fn(out_dir / f"{bp.stem}_after_mask.png",
                         result.after_seg.combined_mask)

        if args.export_json:
            json_path = str(out_dir / f"{bp.stem}_result.json")
            export_comparison_json(
                json_path, result.flood_metrics,
                before_path=str(bp), after_path=str(ap),
            )

        if csv_exp:
            csv_exp.write_row({
                "before": str(bp), "after": str(ap),
                "before_water_px": fm.before_water_px,
                "after_water_px": fm.after_water_px,
                "flooded_new_px": fm.flooded_new_px,
                "receded_px": fm.receded_px,
                "iou": round(fm.iou, 6),
                "net_change": round(fm.net_change_ratio, 6),
            })

    print(f"\nDone — processed {len(pairs)} pair(s)")


if __name__ == "__main__":
    main()
