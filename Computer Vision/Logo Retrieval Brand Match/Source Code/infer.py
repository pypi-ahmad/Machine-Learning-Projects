"""Logo Retrieval Brand Match — CLI entry point.
"""Logo Retrieval Brand Match — CLI entry point.

Query the logo index with an image and retrieve the most similar brands.

Usage::

    # Single image query
    python infer.py --source query_logo.png

    # Directory of queries
    python infer.py --source test_logos/ --export-csv results.csv

    # Custom top-k and index
    python infer.py --source logo.jpg --top-k 10 --index path/to/index.npz

    # With optional logo detection
    python infer.py --source scene.jpg --use-detector
"""
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _run_image(args: argparse.Namespace) -> None:
    import cv2
    from config import LogoConfig, load_config
    from controller import LogoController
    from export import export_results_json
    from visualize import draw_query_overlay, make_result_grid

    cfg = load_config(args.config) if args.config else LogoConfig()
    if args.use_detector:
        cfg.use_detector = True
    if args.top_k:
        cfg.top_k = args.top_k
    cfg.output_dir = args.output_dir

    ctrl = LogoController(cfg)
    ctrl.load(args.index)

    image = cv2.imread(args.source)
    if image is None:
        print(f"[ERROR] Cannot read image: {args.source}")
        return

    result = ctrl.query(image, source=args.source)
    r = result.retrieval

    if not r.hits:
        print("No matches found.")
        ctrl.close()
        return

    # Print results
    print(f"Query: {args.source}")
    if result.detection_used:
        print("  (logo detected and cropped)")
    print(f"  Top brand: {r.top_brand} (score={r.top_score:.4f})")
    print(f"  Brand votes: {r.brand_votes}")
    for h in r.hits:
        print(f"    [{h.rank}] {h.brand:20s}  score={h.score:.4f}  {h.path}")

    out_dir = Path(cfg.output_dir)

    # Result grid
    if args.save_grid:
        grid = make_result_grid(
            result.image, r.hits,
            thumb_size=cfg.grid_thumb_size, cols=cfg.grid_cols,
        )
        grid_path = out_dir / f"{Path(args.source).stem}_grid.jpg"
        grid_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(grid_path), grid)
        print(f"  Grid saved: {grid_path}")

    # Export JSON
    if args.export_json:
        jp = Path(args.export_json)
        jp.parent.mkdir(parents=True, exist_ok=True)
        export_results_json(jp, args.source, r.hits, r.brand_votes)
        print(f"  JSON saved: {jp}")

    # Display
    if not args.no_display:
        overlay = draw_query_overlay(result.image, r.top_brand, r.top_score)
        cv2.imshow("Logo Match", overlay)
        grid = make_result_grid(result.image, r.hits,
                                thumb_size=cfg.grid_thumb_size, cols=cfg.grid_cols)
        cv2.imshow("Top Matches", grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    ctrl.close()


def _run_directory(args: argparse.Namespace) -> None:
    import cv2
    from config import LogoConfig, load_config
    from controller import LogoController
    from export import CSVExporter, export_results_json
    from validator import is_image
    from visualize import make_result_grid

    cfg = load_config(args.config) if args.config else LogoConfig()
    if args.use_detector:
        cfg.use_detector = True
    if args.top_k:
        cfg.top_k = args.top_k
    cfg.output_dir = args.output_dir

    ctrl = LogoController(cfg)
    ctrl.load(args.index)

    out_dir = Path(cfg.output_dir)

    images = sorted(f for f in Path(args.source).iterdir() if is_image(f))
    print(f"Querying {len(images)} image(s) ...\n")

    csv_exp = CSVExporter(args.export_csv) if args.export_csv else None

    for idx, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        result = ctrl.query(frame, source=str(img_path))
        r = result.retrieval

        brand_str = r.top_brand or "?"
        score_str = f"{r.top_score:.4f}" if r.hits else "--"
        print(f"  [{idx + 1}/{len(images)}] {img_path.name:25s} -> "
              f"{brand_str:20s} ({score_str})")

        if csv_exp and r.hits:
            csv_exp.write_hits(str(img_path), r.hits)

        if args.save_grid and r.hits:
            grid = make_result_grid(result.image, r.hits,
                                    thumb_size=cfg.grid_thumb_size, cols=cfg.grid_cols)
            grid_path = out_dir / "grids" / f"{img_path.stem}_grid.jpg"
            grid_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(grid_path), grid)

        if args.export_json and r.hits:
            jp = out_dir / "json" / f"{img_path.stem}.json"
            jp.parent.mkdir(parents=True, exist_ok=True)
            export_results_json(jp, str(img_path), r.hits, r.brand_votes)

    if csv_exp:
        csv_exp.close()
    ctrl.close()
    print(f"\n[DONE] Results written to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Logo Retrieval Brand Match -- query logo images",
    )
    parser.add_argument("--source", type=str, required=True,
                        help="Image or directory of query images")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--index", type=str, default=None,
                        help="Path to logo index (.npz)")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--use-detector", action="store_true",
                        help="Enable YOLO logo detection before retrieval")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--save-grid", action="store_true",
                        help="Save preview grids")
    parser.add_argument("--export-json", type=str, default=None)
    parser.add_argument("--export-csv", type=str, default=None)
    parser.add_argument("--force-download", action="store_true",
                        help="Force re-download dataset")
    args = parser.parse_args()

    if args.force_download:
        from data_bootstrap import ensure_logo_dataset
        ensure_logo_dataset(force=True)

    from validator import validate_source
    report = validate_source(args.source)
    if not report.ok:
        for w in report.warnings:
            print(f"[ERROR] {w}")
        sys.exit(1)

    if report.source_type == "image":
        _run_image(args)
    elif report.source_type == "directory":
        _run_directory(args)
    else:
        print(f"[ERROR] Unsupported source type: {report.source_type}")
        sys.exit(1)


if __name__ == "__main__":
    main()
