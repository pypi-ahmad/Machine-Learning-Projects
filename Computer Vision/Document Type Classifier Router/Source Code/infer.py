"""Document Type Classifier Router — CLI inference.

Usage::

    # Single image
    python infer.py --source document.jpg

    # Batch directory
    python infer.py --source docs/ --batch

    # With exports and saved visuals
    python infer.py --source docs/ --batch --save --save-grid \\
        --export-json results.json --export-csv results.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classifier import DocumentClassifier
from config import RouterConfig, load_config
from export import export_csv, export_json
from router import DocumentRouter
from validator import collect_images, validate_image
from visualize import save_annotated, save_grid


def _print_result(path: Path | str, cr, rd) -> None:
    status = "ROUTED" if rd.routed else "REVIEW"
    print(
        f"  {Path(path).name:40s} "
        f"{cr.display_label:25s} "
        f"conf={cr.confidence:.2%}  "
        f"-> {rd.pipeline:30s} [{status}]"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Document Type Classifier Router — inference"
    )
    ap.add_argument("--source", required=True,
                    help="Image path or directory")
    ap.add_argument("--batch", action="store_true",
                    help="Process all images in source directory")
    ap.add_argument("--weights", type=str, default=None,
                    help="Path to model weights")
    ap.add_argument("--config", type=str, default=None,
                    help="Path to config file (JSON/YAML)")
    ap.add_argument("--model", type=str, default=None,
                    help="Model architecture override")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--threshold", type=float, default=None,
                    help="Confidence threshold for routing")
    ap.add_argument("--save", action="store_true",
                    help="Save annotated images")
    ap.add_argument("--save-grid", action="store_true",
                    help="Save batch thumbnail grid")
    ap.add_argument("--export-json", type=str, default=None,
                    help="Export results to JSON file")
    ap.add_argument("--export-csv", type=str, default=None,
                    help="Export results to CSV file")
    ap.add_argument("--output-dir", type=str, default="output",
                    help="Output directory for saved files")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg.model_name = args.model
    if args.device:
        cfg.device = args.device
    if args.threshold is not None:
        cfg.confidence_threshold = args.threshold

    clf = DocumentClassifier(cfg)
    clf.load(args.weights)
    router = DocumentRouter(cfg)

    paths = collect_images(args.source) if args.batch else [Path(args.source)]
    images = [validate_image(p) for p in paths]
    cls_results = clf.classify_batch(images)
    routes = router.route_batch(cls_results)

    print(f"\n{'-' * 90}")
    print(f"  Document Type Classifier Router -- {len(cls_results)} document(s)")
    print(f"{'-' * 90}")
    for p, cr, rd in zip(paths, cls_results, routes):
        _print_result(p, cr, rd)

    routed = sum(1 for rd in routes if rd.routed)
    review = len(routes) - routed
    print(f"{'-' * 90}")
    print(f"  Routed: {routed}  |  Manual review: {review}")
    print(f"{'-' * 90}\n")

    # ── Optional outputs ──────────────────────────────────
    if args.save:
        for p, img, cr, rd in zip(paths, images, cls_results, routes):
            out = save_annotated(img, cr, rd, args.output_dir,
                                 f"annotated_{p.stem}.jpg", cfg)
            print(f"  Saved: {out}")

    if args.save_grid and len(images) > 1:
        out = save_grid(images, cls_results, routes, args.output_dir,
                        "grid.jpg", cfg)
        print(f"  Grid:  {out}")

    if args.export_json:
        out = export_json(cls_results, routes, args.export_json,
                          [str(p) for p in paths])
        print(f"  JSON:  {out}")

    if args.export_csv:
        out = export_csv(cls_results, routes, args.export_csv,
                         [str(p) for p in paths])
        print(f"  CSV:   {out}")

    clf.close()


if __name__ == "__main__":
    main()
