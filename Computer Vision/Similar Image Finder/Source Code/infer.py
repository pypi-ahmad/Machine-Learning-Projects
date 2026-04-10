"""Similar Image Finder — CLI entry point.

Query the index for similar images.

Usage:
    python infer.py --source photo.jpg --top-k 8
    python infer.py --source photo.jpg --save-grid results/grid.jpg
    python infer.py --source photo.jpg --export-json results/out.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from Source Code/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import SimilarityConfig, load_config
from controller import SimilarityController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Find similar images from the index")
    ap.add_argument("--source", required=True, help="Query image path")
    ap.add_argument("--index-path", default=None, help="Override index path")
    ap.add_argument("--top-k", type=int, default=None, help="Number of results")
    ap.add_argument("--backbone", default=None, help="Backbone model name")
    ap.add_argument("--save-grid", default=None, help="Save result grid to path")
    ap.add_argument("--show", action="store_true", help="Display result grid")
    ap.add_argument("--export-json", default=None, help="Export results to JSON")
    ap.add_argument("--export-csv", default=None, help="Export results to CSV")
    ap.add_argument("--config", default=None, help="Path to config YAML")
    ap.add_argument("--device", default=None, help="Compute device (cpu/cuda)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config) if args.config else SimilarityConfig()
    if args.index_path:
        cfg.index_path = args.index_path
    if args.backbone:
        cfg.backbone = args.backbone
    if args.top_k:
        cfg.top_k = args.top_k
    if args.device:
        cfg.device = args.device

    ctrl = SimilarityController(cfg)
    ctrl.load()

    show = args.show
    save_grid = args.save_grid

    result = ctrl.query_and_visualise(
        args.source,
        save_path=save_grid,
        show=show,
    )

    # Print results
    print(f"\nQuery: {result.query_path}")
    print(f"Top match: {result.top_path}  (score={result.top_score:.4f})")
    print(f"\n{'Rank':<6} {'Score':<10} {'Category':<15} Path")
    print("-" * 70)
    for h in result.hits:
        print(f"{h.rank:<6} {h.score:<10.4f} {h.category or '-':<15} {h.path}")

    if result.category_votes:
        print(f"\nCategory votes:")
        for cat, score in sorted(result.category_votes.items(), key=lambda x: -x[1]):
            print(f"  {cat:<20} {score:.4f}")

    # Export
    ctrl.export_results(
        result,
        json_path=args.export_json,
        csv_path=args.export_csv,
    )

    ctrl.close()


if __name__ == "__main__":
    main()
