"""Building Footprint Change Detector — CLI entry point.

Usage::

    # Single pair
    python infer.py --before before.png --after after.png

    # Batch (matched filenames)
    python infer.py --before-dir images/A --after-dir images/B

    # Export
    python infer.py --before a.png --after b.png --export-json results.json
    python infer.py --before-dir A --after-dir B --export-csv metrics.csv

    # Histogram matching for different illumination
    python infer.py --before a.png --after b.png --match-histograms
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Building Footprint Change Detector — before/after aerial analysis",
    )
    # ── input ──────────────────────────────────────────────
    g = p.add_argument_group("input (single pair)")
    g.add_argument("--before", default=None, help="Path to before image")
    g.add_argument("--after", default=None, help="Path to after image")

    g2 = p.add_argument_group("input (batch)")
    g2.add_argument("--before-dir", default=None, help="Directory of before images")
    g2.add_argument("--after-dir", default=None, help="Directory of after images")

    # ── options ────────────────────────────────────────────
    p.add_argument("--config", default=None, help="YAML/JSON config path")
    p.add_argument("--match-histograms", action="store_true",
                    help="Apply histogram matching to reduce illumination differences")
    p.add_argument("--no-display", action="store_true", help="Headless mode")
    p.add_argument("--output-dir", default="output", help="Output directory")

    # ── export ─────────────────────────────────────────────
    p.add_argument("--export-json", default=None, help="JSON export path (per pair)")
    p.add_argument("--export-csv", default=None, help="CSV export path (appended)")
    p.add_argument("--save-visuals", action="store_true",
                    help="Save side-by-side and change overlay images")

    # ── data ───────────────────────────────────────────────
    p.add_argument("--force-download", action="store_true",
                    help="Re-download dataset")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    import cv2

    from config import ChangeConfig, load_config
    from controller import ChangeDetectorController
    from export import export_csv_row, export_json
    from validator import validate_directory_pair
    from visualize import compose_report, draw_change_overlay, draw_side_by_side

    if args.force_download:
        from data_bootstrap import ensure_change_dataset
        ensure_change_dataset(force=True)

    cfg = load_config(args.config) if args.config else ChangeConfig()
    cfg.output_dir = args.output_dir
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ctrl = ChangeDetectorController(cfg)
    ctrl.load()

    # ── resolve pairs ──────────────────────────────────────
    pairs: list[tuple[Path, Path]] = []

    if args.before and args.after:
        pairs.append((Path(args.before), Path(args.after)))
    elif args.before_dir and args.after_dir:
        report, matched = validate_directory_pair(args.before_dir, args.after_dir)
        if not report.ok:
            for w in report.warnings:
                print(f"ERROR: {w}", file=sys.stderr)
            sys.exit(1)
        for w in report.warnings:
            print(f"WARN: {w}")
        pairs = matched
        print(f"Found {len(pairs)} matched image pair(s)")
    else:
        print("Provide --before/--after or --before-dir/--after-dir", file=sys.stderr)
        sys.exit(1)

    # ── process each pair ──────────────────────────────────
    for idx, (bp, ap) in enumerate(pairs):
        print(f"\n[{idx + 1}/{len(pairs)}] {bp.name}  ↔  {ap.name}")

        result = ctrl.process_pair(
            bp, ap,
            match_histograms=args.match_histograms,
        )

        m = result.metrics
        print(f"  Buildings: before={m.before_area_px:,}px  after={m.after_area_px:,}px")
        print(f"  New: {m.new_area_px:,}px ({m.num_new_regions} regions)  "
              f"Demolished: {m.demolished_area_px:,}px ({m.num_demolished_regions} regions)")
        print(f"  IoU={m.iou:.4f}  change_ratio={m.change_ratio:.4f}  "
              f"growth={m.growth_ratio:.4f}")

        if not result.report.ok:
            for w in result.report.warnings:
                print(f"  WARN: {w}")

        # ── visuals ────────────────────────────────────────
        if not args.no_display or args.save_visuals:
            report_img = compose_report(
                result.pair.before, result.pair.after,
                result.before_seg.mask, result.after_seg.mask,
                result.diff, result.metrics,
                alpha=cfg.mask_alpha,
            )
            if not args.no_display:
                title = f"Change Detection — {bp.name}"
                cv2.imshow(title, report_img)
                key = cv2.waitKey(0 if len(pairs) == 1 else 1500) & 0xFF
                if key == ord("q"):
                    break

            if args.save_visuals:
                stem = bp.stem
                cv2.imwrite(str(out_dir / f"{stem}_report.jpg"), report_img)
                sbs = draw_side_by_side(
                    result.pair.before, result.pair.after,
                    result.before_seg.mask, result.after_seg.mask,
                )
                cv2.imwrite(str(out_dir / f"{stem}_side_by_side.jpg"), sbs)
                overlay = draw_change_overlay(result.pair.after, result.diff)
                cv2.imwrite(str(out_dir / f"{stem}_change_overlay.jpg"), overlay)

        # ── export ─────────────────────────────────────────
        if args.export_json:
            json_path = args.export_json
            if len(pairs) > 1:
                json_path = str(out_dir / f"{bp.stem}_result.json")
            export_json(json_path, m, result.diff,
                        before_path=str(bp), after_path=str(ap))

        if args.export_csv:
            export_csv_row(args.export_csv, m,
                           before_path=str(bp), after_path=str(ap),
                           write_header=(idx == 0))

    ctrl.close()
    if not args.no_display:
        cv2.destroyAllWindows()
    print(f"\nDone — processed {len(pairs)} pair(s)")


if __name__ == "__main__":
    main()
