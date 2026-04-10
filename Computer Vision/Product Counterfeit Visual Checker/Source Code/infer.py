"""Product Counterfeit Visual Checker — CLI entry point.

Screen suspect product images against approved references.

Usage:
    python infer.py --source suspect.jpg
    python infer.py --source suspect.jpg --save-grid output/grid.jpg
    python infer.py --source suspect.jpg --save-heatmap output/heatmap.jpg
    python infer.py --source suspect.jpg --export-json output/result.json
    python infer.py --source suspect.jpg --product "Granny-Smith"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CounterfeitConfig, load_config
from controller import CounterfeitController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_DISCLAIMER = (
    "NOTE: This is a visual screening tool. Results indicate visual "
    "similarity to approved references and do NOT constitute proof of "
    "counterfeit status. Further investigation is always required."
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Screen a product image for visual mismatch risk",
        epilog=_DISCLAIMER,
    )
    ap.add_argument("--source", required=True, help="Suspect image path")
    ap.add_argument("--ref-path", default=None, help="Override reference store path")
    ap.add_argument("--product", default=None,
                    help="Filter references to a specific product label")
    ap.add_argument("--top-k", type=int, default=None, help="Number of references")
    ap.add_argument("--backbone", default=None, help="Backbone model name")
    ap.add_argument("--save-grid", default=None, help="Save comparison grid to path")
    ap.add_argument("--save-heatmap", default=None,
                    help="Save region heatmap to path")
    ap.add_argument("--show", action="store_true", help="Display comparison grid")
    ap.add_argument("--export-json", default=None, help="Export results to JSON")
    ap.add_argument("--export-csv", default=None, help="Export results to CSV")
    ap.add_argument("--config", default=None, help="Path to config YAML/JSON")
    ap.add_argument("--device", default=None, help="Compute device (cpu/cuda)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config) if args.config else CounterfeitConfig()
    if args.ref_path:
        cfg.reference_path = args.ref_path
    if args.backbone:
        cfg.backbone = args.backbone
    if args.top_k:
        cfg.top_k = args.top_k
    if args.device:
        cfg.device = args.device

    ctrl = CounterfeitController(cfg)
    ctrl.load()

    result = ctrl.screen_and_visualise(
        args.source,
        save_path=args.save_grid,
        show=args.show,
        product_filter=args.product,
    )

    # Region heatmap
    if args.save_heatmap:
        ctrl.make_heatmap(args.source, result, save_path=args.save_heatmap)

    # Print results
    print(f"\n{'=' * 60}")
    print(f"  VISUAL SCREENING RESULT")
    print(f"{'=' * 60}")
    print(f"  Suspect:           {result.suspect_path}")
    print(f"  Risk level:        {result.risk_level.upper()}")
    print(f"  Mismatch risk:     {result.mismatch_risk_pct}%")
    print(f"  Best composite:    {result.best_composite:.4f}")
    print(f"  Best reference:    {result.best_reference}")
    print(f"  Best product:      {result.best_product}")
    print(f"{'=' * 60}")

    if result.details:
        print(f"\n{'Ref':<5} {'Product':<20} {'Global':<10} {'Region':<10} "
              f"{'Histo':<10} {'Composite':<10}")
        print("-" * 65)
        for i, d in enumerate(result.details, 1):
            print(f"{i:<5} {(d.reference_product or '-'):<20} "
                  f"{d.global_score:<10.4f} {d.region_score:<10.4f} "
                  f"{d.histogram_score:<10.4f} {d.composite_score:<10.4f}")

    print(f"\n{_DISCLAIMER}\n")

    # Export
    ctrl.export_results(
        result,
        json_path=args.export_json,
        csv_path=args.export_csv,
    )

    ctrl.close()


if __name__ == "__main__":
    main()
